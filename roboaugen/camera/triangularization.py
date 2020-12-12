from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from roboaugen.camera.transformation_estimator import ProcrustesProblemSolver
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics
from modern_robotics import *
from roboaugen.camera.model import CameraModel
from roboaugen.camera.model import MathUtils

from scipy.linalg import svd
import numpy as np
import cv2
import torch
from colour import Color
import json
import click
import os

class FundamentalMatrixGenerator():

    def __init__(self, camera, camera_topology):
        self.camera = camera
        self.forward_kinematics = RobotForwardKinematics(camera_topology)

    def generate_fundamental_matrix(self, initial_state, final_state):
        initial_transformation = self.forward_kinematics.get_transformation(initial_state)
        final_transformation = self.forward_kinematics.get_transformation(final_state)
        final_viewed_from_initial = np.linalg.inv(initial_transformation) @ final_transformation
        translation = final_viewed_from_initial[0:3, 3]
        rotation = final_viewed_from_initial[0:3, 0:3]
        essential_matrix = MathUtils.skew(translation) @ rotation
        inverse_camera_matrix = np.linalg.inv(self.camera.undistorted_camera_matrix @ self.camera.trans_robot_to_camera_rotation)
        fundamental_matrix = inverse_camera_matrix.T @ essential_matrix @ inverse_camera_matrix
        return fundamental_matrix

    def get_origin(self):
        camera_matrix_transformed = self.undistorted_camera_matrix# @ self.trans_robot_to_camera
        origin = camera_matrix_transformed @ np.array([0, 0, 1])
        return origin

class EpipolarLineGenerator():

    def __init__(self, fundamental_matrix):
        self.fundamental_matrix = fundamental_matrix

    def get_epipolar_line_in_initial_image_from_point_in_final(self, coordinate_x, coordinate_y):
        epipolar_line  = self.fundamental_matrix @ np.array([coordinate_x, coordinate_y, 1])
        return EpipolarLine(epipolar_line)

    def get_epipolar_line_in_final_image_from_point_in_initial(self, coordinate_x, coordinate_y):
        epipolar_line  = np.array([coordinate_x, coordinate_y, 1]).T @ self.fundamental_matrix.numpy()
        return EpipolarLine(epipolar_line)

    def get_epipolar_lines_in_final_image_from_points_in_initial(self, points):
        epipolar_lines  = points.T.double() @ self.fundamental_matrix
        return epipolar_lines

class EpipolarLine():

    def __init__(self, epipolar_line):
        self.a, self.b, self.c = epipolar_line[0], epipolar_line[1], epipolar_line[2]

    def x(self, y):
        return (- ( ( self.b * y ) + self.c ) / self.a)

    def y(self, x):
        return - ( ( self.a * x ) + self.c ) / self.b

    def from_image(self, image):
        height, width = image.shape[0], image.shape[1]
        if abs(self.a) < 1e-5:
            # y is cte
            y_cte = int(self.y(0.))
            return 0, y_cte, width, y_cte
        else:
            y_init = - height
            x_init = int(self.x(y_init))
            y_final = height
            x_final = int(self.x(y_final))
            return x_init, y_init, x_final, y_final

def to_radians(degrees):
    return ( degrees * np.pi ) / 180.


def print_coordinates(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        print(x, y)


class KeypointMatcher():

    def __init__(self, epipolar_line_generator):
        self.config = Config()
        self.epipolar_line_generator = epipolar_line_generator

    def get_predictions_in_image_index(self, image_index, coordinate_predictions):
        indices_image = (coordinate_predictions[:, 0] == image_index).nonzero()[:, 0]
        predictions_image = coordinate_predictions[indices_image, :]
        coordinates_image = predictions_image[:, 2:]
        ones = coordinates_image.new_ones(coordinates_image[:, 0:1].size())
        coordinates_image = torch.cat([coordinates_image, ones], 1).transpose(0, 1)
        return predictions_image, coordinates_image

    def get_probability(self, probability_indices,  predicted_heatmaps, scale):
        heatmap_coordinates = [probability_indices[0].item(), probability_indices[1].item(),
        int(probability_indices[2].item()), int(probability_indices[3].item())]
        probability  = predicted_heatmaps[heatmap_coordinates[0], heatmap_coordinates[1],
            heatmap_coordinates[2], heatmap_coordinates[3]].item()
        return probability

    def scaled_points(self, coordinates_image, scale):
        y = ( (coordinates_image[0, :] ) * scale ).double() + (scale / 2.)
        x = ( (coordinates_image[1, :] ) * scale ).double() + (scale / 2.)
        ones = coordinates_image[2, :]
        scaled_coordinates_image = torch.cat([x.unsqueeze(0), y.unsqueeze(0), ones.unsqueeze(0)], 0)
        return scaled_coordinates_image

    def get_matches_from_predictions(self, predicted_heatmaps, scale, prediction_threshold = 0.1, epipolar_threshold = 3.):
        predicted_heatmaps = predicted_heatmaps * (predicted_heatmaps > prediction_threshold)
        coordinate_predictions = (predicted_heatmaps > 0.).nonzero()

        # assume two images
        predictions_initial_image, coordinates_initial_image = self.get_predictions_in_image_index(0, coordinate_predictions)
        predictions_final_image, coordinates_final_image = self.get_predictions_in_image_index(1, coordinate_predictions)

        scaled_initial_coordinates = self.scaled_points(coordinates_initial_image, scale)
        scaled_final_coordinates = self.scaled_points(coordinates_final_image, scale)

        epipolar_lines_in_final = self.epipolar_line_generator.\
            get_epipolar_lines_in_final_image_from_points_in_initial(scaled_initial_coordinates)

        keypoint_to_matches = {}
        for keypoint_type in range(predicted_heatmaps.size()[1]):
            #print(f'Getting matches for keypoint type index {keypoint_type}')
            indices_initial_keypoint = (predictions_initial_image[:, 1] == keypoint_type).nonzero()
            indices_final_keypoint = (predictions_final_image[:, 1] == keypoint_type).nonzero()
            keypoint_to_matches[keypoint_type] = []
            if indices_initial_keypoint.size()[0] > 0 and indices_final_keypoint.size()[0] > 0:
                indices_initial_keypoint = indices_initial_keypoint[:, 0]
                indices_final_keypoint = indices_final_keypoint[:, 0]
                epipolar_lines = epipolar_lines_in_final[indices_initial_keypoint].double()
                points = scaled_final_coordinates[:, indices_final_keypoint].double()
                distances_initial_to_final = epipolar_lines @ points
                distances_initial_to_final = torch.sqrt(distances_initial_to_final ** 2)
                distances_initial_to_final_indices = (distances_initial_to_final < epipolar_threshold).nonzero()
                matches = distances_initial_to_final_indices.size()[0]
                if matches > 0:
                    #print(f'\t{matches} matches found from original initial {distances_initial_to_final.size()[0]} points and {distances_initial_to_final.size()[1]} final points.')
                    for match_index in range(matches):
                        initial_and_final_indices = distances_initial_to_final_indices[match_index]
                        coordinates_initial = scaled_initial_coordinates[:2, [indices_initial_keypoint[initial_and_final_indices[0]]]]
                        coord_initial = torch.cat([coordinates_initial[0], coordinates_initial[1]], 0).numpy()
                        coordinates_final = scaled_final_coordinates[:2, [indices_final_keypoint[initial_and_final_indices[1]]]]
                        coord_final = torch.cat([coordinates_final[0], coordinates_final[1]], 0).numpy()
                        distance = distances_initial_to_final[initial_and_final_indices[0], initial_and_final_indices[1]].item()

                        probability_initial  = self.get_probability(predictions_initial_image[indices_initial_keypoint[initial_and_final_indices[0]]],
                            predicted_heatmaps, scale)
                        probability_final  = self.get_probability(predictions_final_image[indices_final_keypoint[initial_and_final_indices[1]]],
                            predicted_heatmaps, scale)

                        epipolar_match = EpipolarMatch(coord_initial, coord_final, distance, probability_initial, probability_final)
                        #print(epipolar_match)
                        keypoint_to_matches[keypoint_type].append(epipolar_match)
                else:
                    pass
                    #print(f'\tNo matches found')
        return keypoint_to_matches

    def aggregate_groups(self, keypoint_to_matches_grouped):
        keypoint_to_matches_grouped_aggregated = []
        for groups_in_keypoint in keypoint_to_matches_grouped:
            #print(f'Keypoint {groups_in_keypoint}')
            groupped_matches = []
            for group in groups_in_keypoint:
                if(len(group) > 0):
                    total_probaility_initial = sum([match.probability_initial for match in group])
                    #print('total_probaility_initial', total_probaility_initial)
                    total_probaility_final = sum([match.probability_final for match in group])
                    coord_initial = sum([(match.coord_initial * match.probability_initial) / total_probaility_initial for match in group])
                    coord_final = sum([(match.coord_final * match.probability_final) / total_probaility_final for match in group])
                    grouped_match = EpipolarMatch(\
                                coord_initial,
                                coord_final,
                                0., # we don't care, fill it later
                                total_probaility_initial,
                                total_probaility_initial)
                    groupped_matches.append(grouped_match)
            keypoint_to_matches_grouped_aggregated.append(groupped_matches)
        keypoint_to_matches_grouped_aggregated  = {keypoint: matches for keypoint, matches in enumerate(keypoint_to_matches_grouped_aggregated)}
        return keypoint_to_matches_grouped_aggregated

    def compute_point_distances(self, points):
        points_replicated = np.expand_dims(points, axis=0)
        points_replicated = np.repeat(points_replicated, points.shape[0], axis=0)
        point_distances = points_replicated - np.swapaxes(points_replicated, 0, 1)
        point_distances = point_distances.reshape(point_distances.shape[0] * point_distances.shape[1], points.shape[1])
        point_distances = np.linalg.norm(point_distances, axis=1)
        point_distances = point_distances.reshape(points.shape[0], points.shape[0])
        return point_distances

    def group_matches(self, keypoint_to_matches, grouping_distance_threshold = 10.):
        keypoint_to_matches_grouped = []
        for keypoint in keypoint_to_matches:
            matches_grouped = []
            matches = keypoint_to_matches[keypoint]
            if len(matches) > 1:
                matches_vectors = np.array([np.concatenate((match.coord_initial, match.coord_final)) for match in matches])
                match_distances = self.compute_point_distances(matches_vectors)
                match_distances = match_distances < grouping_distance_threshold
                matched_left_to_group = np.array([True] * match_distances.shape[0])
                for match_index in range(match_distances.shape[0]):
                    matches_to_group = [matches[idx] for idx, is_match in enumerate(match_distances[match_index]) if is_match and matched_left_to_group[idx] ]
                    if len(matches_to_group) > 0:
                        matches_grouped.append(matches_to_group)
                        matched_left_to_group = matched_left_to_group * (1 - match_distances[match_index])
            else:
                matches_grouped = [matches]
            keypoint_to_matches_grouped.append(matches_grouped)

        return self.aggregate_groups(keypoint_to_matches_grouped)

    def draw_keypoint_matches(self, keypoint_to_matches, image_initial_raw, image_final_raw, camera_model):
        image_initial = camera_model.undistort_image(image_initial_raw)
        image_final = camera_model.undistort_image(image_final_raw)
        hues = torch.arange(start=0,end=179., step = 179 / (self.config.num_vertices + 1) )
        colors = [Color(hsl=(hue/180, 1, 0.5)).rgb for hue in hues]
        colors = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]
        for keypoint in keypoint_to_matches:
            color = colors[keypoint]
            for match in keypoint_to_matches[keypoint]:
                epipoloar_line = self.epipolar_line_generator.get_epipolar_line_in_final_image_from_point_in_initial(match.coord_initial[0], match.coord_initial[1])
                x_init, y_init, x_final, y_final = epipoloar_line.from_image(image_final)
                image_final = cv2.line(image_final, (x_init, y_init), (x_final, y_final), color, thickness=2)
                image_initial = cv2.circle(image_initial, (int(match.coord_initial[0]), int(match.coord_initial[1])), 4, color, thickness=2)
                image_final = cv2.circle(image_final, (int(match.coord_final[0]), int(match.coord_final[1])), 4, color, thickness=2)

        return image_initial, image_final

class EpipolarMatch():

    def __init__(self, coord_initial, coord_final, distance, probability_initial, probability_final):
        self.coord_initial = coord_initial
        self.coord_final = coord_final
        self.distance = distance
        self.probability_initial = probability_initial
        self.probability_final = probability_final

    def __repr__(self):
        return str(self.__dict__)

class Triangularizer():

    def __init__(self, camera_model, camera_robot_topology):
        self.config = Config()
        self.camera_model = camera_model
        self.camera_matrix = camera_model.undistorted_camera_matrix @ camera_model.trans_robot_to_camera_rotation
        # TODO: this is a hack to adapt the focals on x and y directions so that there is distance accuracy.
        # This is to be fixed recalibrating properly the camera
        self.camera_matrix[0, 1] *= 0.74
        self.camera_matrix[1, 2] *= 0.74
        self.forward_kinematics = RobotForwardKinematics(camera_robot_topology)

    def triangularize(self, initial_state, final_state, keypoint_to_matches, triangularization_threshold=15.):
        initial_transformation = self.forward_kinematics.get_transformation(initial_state)
        final_transformation = self.forward_kinematics.get_transformation(final_state)
        initial_projection_matrix = self.camera_matrix @ np.linalg.inv(initial_transformation)[:3,:]
        final_projection_matrix = self.camera_matrix @ np.linalg.inv(final_transformation)[:3,:]
        points_predicted = []
        for keypoint in range(self.config.num_vertices):
            points_predicted_keypoint = []
            if keypoint in keypoint_to_matches:
                matches = keypoint_to_matches[keypoint]
                for match in matches:
                    coord_initial = MathUtils.skew([match.coord_initial[0], match.coord_initial[1], 1])
                    coord_final = MathUtils.skew([match.coord_final[0], match.coord_final[1], 1])
                    vector_space_initial = coord_initial @ initial_projection_matrix
                    vector_space_final   = coord_final   @ final_projection_matrix
                    vector_space = np.concatenate((vector_space_initial, vector_space_final), axis=0)
                    #u, s, vh = np.linalg.svd(vector_space)
                    u, s, vh =  svd(vector_space, lapack_driver='gesvd')
                    #S = np.concatenate((np.diag(s), np.array([[0, 0, 0, 0], [0, 0, 0, 0]])), axis=0)
                    #vector_space_r = u @ S @ vh
                    #print('reconstruction error',  np.abs(vector_space - vector_space_r).sum())
                    nullspace = vh[3, :]
                    point = nullspace / nullspace[3]
                    triangularization_error = s[3]
                    point = point[:3]
                    point_extended = np.append(point, np.array([1]), axis=0)
                    initial_point = self.camera_matrix @ (np.linalg.inv(initial_transformation) @ point_extended)[:3]
                    initial_point = initial_point / initial_point[2]
                    # Final
                    final_point = self.camera_matrix @ (np.linalg.inv(final_transformation) @ point_extended)[:3]
                    final_point = final_point / final_point[2]
                    mean_reprojection_error = ( ((initial_point[:2] - match.coord_initial) ** 2).mean() + \
                        ((final_point[:2] - match.coord_final) ** 2).mean() ) / 2.

                    if triangularization_error < triangularization_threshold:
                        #print('IN', keypoint, point, s[3], mean_reprojection_error)
                        points_predicted_keypoint.append(point)
                    else:
                        pass
                        #print('OUT', keypoint, point, s[3], mean_reprojection_error)
            points_predicted.append(points_predicted_keypoint)
        return points_predicted



    def visualize_reprojection(self, points_predicted, image_initial_raw, image_final_raw, initial_state, final_state):
        image_initial = self.camera_model.undistort_image(image_initial_raw)
        image_final = self.camera_model.undistort_image(image_final_raw)

        initial_transformation = np.linalg.inv(self.forward_kinematics.get_transformation(initial_state))
        final_transformation = np.linalg.inv(self.forward_kinematics.get_transformation(final_state))

        hues = torch.arange(start=0,end=179., step = 179 / (self.config.num_vertices + 1) )
        colors = [Color(hsl=(hue / 180, 1, 0.5)).rgb for hue in hues]
        colors = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]
        for idx, points_in_keypoint in enumerate(points_predicted):
            color = colors[idx]
            for point_predicted in points_in_keypoint:
                point_predicted = np.append(point_predicted, np.array([1]), axis=0)
                # Initial
                initial_point = self.camera_matrix @ (initial_transformation @ point_predicted)[:3]
                initial_point = initial_point / initial_point[2]
                image_initial = cv2.circle(image_initial, (int(initial_point[0]), int(initial_point[1])), 4, color, thickness=2)
                # Final
                final_point = self.camera_matrix @ (final_transformation @ point_predicted)[:3]
                final_point = final_point / final_point[2]
                image_final = cv2.circle(image_final, (int(final_point[0]), int(final_point[1])), 4, color, thickness=2)

        cv2.imshow(f'Reprojected | Point in Initial Image', image_initial)
        cv2.imshow(f'Reprojected | Point in Final Image', image_final)

class EndToEndTransformationSolution():

    def __init__(self, solution, image_initial_procrustes, image_final_procrustes, image_initial_ungrouped, image_final_ungrouped,
        image_initial_grouped, image_final_grouped, heatmap_images):
        self.solution = solution
        self.image_initial_procrustes = image_initial_procrustes
        self.image_final_procrustes = image_final_procrustes
        self.image_initial_grouped = image_initial_grouped
        self.image_final_grouped = image_final_grouped
        self.heatmap_images = heatmap_images
        self.image_initial_ungrouped = image_initial_ungrouped
        self.image_final_ungrouped = image_final_ungrouped

    def __repr__(self):
        return str(self.__dict__)

    def to_json(self):
        if self.solution is not None:
            return json.dumps(self.solution.__dict__, indent = 4)
        else:
            return ""

class EndToEndTransformationEstimator():

    def __init__(self):
        self.camera_topology = RobotTopology(l1=142, l2=142, l3=80, h1=50, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)
        self.inferencer = Inferencer(distort=False, keep_dimensions=True, use_cache=False, \
            mode='silco', max_background_objects=1, max_foreground_objects=1)
        supports_folder = 'test/images/'
        self.supports = self.inferencer.get_supports_from_folder(supports_folder)
        self.supports = torch.cat([self.supports] * 2 ).cuda()


    def compute_transformation(self, initial_state, final_state, image_initial_raw, image_final_raw, prediction_threshold = 0.1):
        height, width = image_final_raw.shape[0], image_final_raw.shape[1]
        camera_model = CameraModel(width, height)
        fundamental_matrix_generator = FundamentalMatrixGenerator(camera_model, self.camera_topology)

        fundamental_matrix = fundamental_matrix_generator.generate_fundamental_matrix(initial_state, final_state)
        epipolar_line_generator = EpipolarLineGenerator(torch.from_numpy(fundamental_matrix))

        image_initial = camera_model.undistort_image(image_initial_raw)
        image_final = camera_model.undistort_image(image_final_raw)

        # perform inference
        images = [image_initial, image_final]
        query = self.inferencer.get_queries_from_opencv_images(images)
        original_height, original_width = image_final.shape[0], image_final.shape[1] #(480, 640, 3)


        query_for_view = query.clone()
        query = query.cuda()
        predicted_heatmaps = self.inferencer.get_model_inference(self.supports, query)
        predicted_heatmaps = predicted_heatmaps.cpu()

        target_height, targe_width = predicted_heatmaps.size()[2], predicted_heatmaps.size()[3] # torch.Size([2, 3, 96, 128])
        scale = original_width / targe_width
        #print(f'Original h/w {original_height}, {original_width} => Target h/w {target_height}, {targe_width}. Scales: {original_width / targe_width} | {original_height / target_height}')


        predicted_heatmaps = predicted_heatmaps * (predicted_heatmaps > prediction_threshold)
        heatmap_images = []
        for idx, image in enumerate(images):
            visual_targets, visual_predictions, visual_suports = \
                self.inferencer.display_results(f'Inference {idx}', query_for_view[idx: idx + 1], None, predicted_heatmaps[idx: idx + 1], threshold=0.0)
            heatmap_images.append((visual_targets, visual_predictions, visual_suports))


        keypoint_matcher = KeypointMatcher(epipolar_line_generator)
        keypoint_to_matches = keypoint_matcher.get_matches_from_predictions(predicted_heatmaps,\
            scale, prediction_threshold = prediction_threshold, epipolar_threshold = 1.)
        image_initial_ungrouped, image_final_ungrouped = keypoint_matcher.draw_keypoint_matches(keypoint_to_matches, image_initial_raw, image_final_raw, camera_model)
        keypoint_to_matches = keypoint_matcher.group_matches(keypoint_to_matches)
        image_initial_grouped, image_final_grouped = keypoint_matcher.draw_keypoint_matches(keypoint_to_matches, image_initial_raw, image_final_raw, camera_model)
        triangularizer = Triangularizer(camera_model, self.camera_topology)
        points_predicted = triangularizer.triangularize(initial_state, final_state, keypoint_to_matches)
        ### triangularizer.visualize_reprojection(points_predicted, image_initial_raw, image_final_raw, initial_state, final_state)

        procrustes_problem_solver = ProcrustesProblemSolver()
        solution = procrustes_problem_solver.solve(points_predicted)
        if solution is not None:
            image_initial_procrustes, image_final_procrustes = procrustes_problem_solver.visualize(solution,
                self.camera_topology, initial_state, final_state, image_initial_raw, image_final_raw)
            return EndToEndTransformationSolution(solution, image_initial_procrustes,
                image_final_procrustes, image_initial_ungrouped, image_final_ungrouped,
                image_initial_grouped, image_final_grouped, heatmap_images)
        else:
            return EndToEndTransformationSolution(None, None,
                None, image_initial_ungrouped, image_final_ungrouped,
                image_initial_grouped, image_final_grouped, heatmap_images)


@click.command()
@click.option("--log_idx", default=0, help="Log folder index")
@click.option("--threshold", default=0.1, help="Prediction heatmap threshold")
def triangularize(log_idx, threshold):
    config = Config()
    #image_initial_path = '/home/rusalka/Pictures/Webcam/first.jpg'
    #image_final_path = '/home/rusalka/Pictures/Webcam/second.jpg'
    data_log_dir = f'data/log/{log_idx}'
    if os.path.exists(data_log_dir):
        print(f'\tTriangualizing from {data_log_dir}')
        image_initial_path = f'{data_log_dir}/image_initial.jpg'
        image_final_path = f'{data_log_dir}/image_final.jpg'
        image_initial_raw = config.get_image_from_path(image_initial_path)
        image_final_raw = config.get_image_from_path(image_final_path)
        initial_state_data = None
        final_state_data = None
        with open( f'{data_log_dir}/initial_state.json') as json_file:
            initial_state_data = json.load(json_file)
        with open( f'{data_log_dir}/final_state.json') as json_file:
            final_state_data = json.load(json_file)
        final_state = RobotState(linear_1=initial_state_data['linear_1'],
            angle_1=initial_state_data['angle_1'], angle_2=initial_state_data['angle_2'], angle_3=initial_state_data['angle_3'])
        initial_state = RobotState(linear_1=final_state_data['linear_1'],
            angle_1=final_state_data['angle_1'], angle_2=final_state_data['angle_2'], angle_3=final_state_data['angle_3'])
        end_to_end_transformation_estimator = EndToEndTransformationEstimator()
        end_to_end_solution = \
            end_to_end_transformation_estimator.compute_transformation(initial_state, final_state, image_initial_raw, image_final_raw, threshold)
        visual_targets_initial, visual_predictions_initial, visual_suports_initial = end_to_end_solution.heatmap_images[0]
        visual_targets_final, visual_predictions_final, visual_suports_final = end_to_end_solution.heatmap_images[1]
        if visual_targets_initial is not None:
            cv2.imshow(f'Targets | Image 1', visual_targets_initial)
        if visual_predictions_initial is not None:
            cv2.imshow(f'Heatmap Predictions | Image 1', visual_predictions_initial)
        if visual_suports_initial is not None:
            cv2.imshow(f'Supports | Image 1', visual_suports_initial)
        if visual_targets_final is not None:
            cv2.imshow(f'Targets | Image 2', visual_targets_final)
        if visual_predictions_final is not None:
            cv2.imshow(f'Heatmap Predictions | Image 2', visual_predictions_final)
        if visual_suports_final is not None:
            cv2.imshow(f'Supports | Image 2', visual_suports_final)

        cv2.imshow(f'UnGrouped | Point in image 1', end_to_end_solution.image_initial_ungrouped)
        cv2.namedWindow(f'UnGrouped | Point in image 1')
        cv2.setMouseCallback(f'UnGrouped | Point in image 1', print_coordinates)
        cv2.imshow(f'UnGrouped | Epipolar line in second image', end_to_end_solution.image_final_ungrouped)
        cv2.namedWindow(f'UnGrouped | Epipolar line in second image')
        cv2.setMouseCallback(f'UnGrouped | Epipolar line in second image', print_coordinates)

        cv2.imshow(f'Grouped | Point in image 1', end_to_end_solution.image_initial_grouped)
        cv2.namedWindow(f'Grouped | Point in image 1')
        cv2.setMouseCallback(f'Grouped | Point in image 1', print_coordinates)
        cv2.imshow(f'Grouped | Epipolar line in second image', end_to_end_solution.image_final_grouped)
        cv2.namedWindow(f'Grouped | Epipolar line in second image')
        cv2.setMouseCallback(f'Grouped | Epipolar line in second image', print_coordinates)
        solution = end_to_end_solution.solution
        if solution is not None:
            print("".join(['-'] * 40))
            print('Solution')
            print(solution)

            # Grouped

            # Final solution
            cv2.imshow(f'Transfomation Predicted | Point in Initial Image', end_to_end_solution.image_initial_procrustes)
            cv2.imshow(f'Transfomation Predicted | Point in Final Image', end_to_end_solution.image_final_procrustes)


        else:
            print('\tNo Solution found')
        cv2.waitKey()


if __name__ == '__main__':
    triangularize()
