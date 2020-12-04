from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from roboaugen.camera.transformation_estimator import ProcrustesProblemSolver
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics
from modern_robotics import *

import numpy as np
import cv2
import torch
from colour import Color

class CameraModel():

    def __init__(self, width, height):
        self.trans_robot_to_camera_rotation = np.array([ \
                                            [0., -1., 0.],
                                            [0.,  0.,-1.],
                                            [1.,  0., 0.]])
        self.config = Config()
        self.camera_matrix, self.distortion_coefficients = self.config.load_camera_parameters()
        self.undistorted_camera_matrix, self.region_of_interest = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficients,\
            (width, height), 0, (width, height) )

    def undistort_image(self, image):
        return cv2.undistort(image, \
            self.camera_matrix, \
            self.distortion_coefficients, \
            None, \
            self.undistorted_camera_matrix)


class MathUtils():

    @staticmethod
    def skew(x):
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])

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

    def get_coordinate_predictions(self, predicted_heatmaps, scale):
        coordinate_predictions = (predicted_heatmaps > 0.).nonzero()
        coordinate_predictions_x = coordinate_predictions[:, 3] * scale
        coordinate_predictions_y = coordinate_predictions[:, 2] * scale
        coordinate_predictions[:, 2] = coordinate_predictions_x
        coordinate_predictions[:, 3] = coordinate_predictions_y
        return coordinate_predictions


    def get_predictions_in_image_index(self, image_index, coordinate_predictions):
        indices_image = (coordinate_predictions[:, 0] == image_index).nonzero()[:, 0]
        predictions_image = coordinate_predictions[indices_image, :]
        coordinates_image = predictions_image[:, 2:]
        ones = coordinates_image.new_ones(coordinates_image[:, 0:1].size())
        coordinates_image = torch.cat([coordinates_image, ones], 1).transpose(0, 1)
        return predictions_image, coordinates_image

    def get_matches_from_predictions(self, predicted_heatmaps, scale, prediction_threshold = 0.1, epipolar_threshold = 3.):
        predicted_heatmaps = predicted_heatmaps * (predicted_heatmaps > prediction_threshold)
        coordinate_predictions = self.get_coordinate_predictions(predicted_heatmaps, scale)

        # assume two images
        predictions_initial_image, coordinates_initial_image = self.get_predictions_in_image_index(0, coordinate_predictions)
        predictions_final_image, coordinates_final_image = self.get_predictions_in_image_index(1, coordinate_predictions)

        epipolar_lines_in_final = self.epipolar_line_generator.\
            get_epipolar_lines_in_final_image_from_points_in_initial(coordinates_initial_image)

        keypoint_to_matches = {}
        for keypoint_type in range(predicted_heatmaps.size()[1]):
            print(f'Getting matches for keypoint type index {keypoint_type}')
            indices_initial_keypoint = (predictions_initial_image[:, 1] == keypoint_type).nonzero()
            indices_final_keypoint = (predictions_final_image[:, 1] == keypoint_type).nonzero()
            keypoint_to_matches[keypoint_type] = []
            if indices_initial_keypoint.size()[0] > 0 and indices_final_keypoint.size()[0] > 0:
                indices_initial_keypoint = indices_initial_keypoint[:, 0]
                indices_final_keypoint = indices_final_keypoint[:, 0]
                epipolar_lines = epipolar_lines_in_final[indices_initial_keypoint].double()
                points = coordinates_final_image[:, indices_final_keypoint].double()
                distances_initial_to_final = epipolar_lines @ points
                distances_initial_to_final = torch.sqrt(distances_initial_to_final ** 2)
                distances_initial_to_final_indices = (distances_initial_to_final < epipolar_threshold).nonzero()
                matches = distances_initial_to_final_indices.size()[0]
                if matches > 0:
                    print('distances_initial_to_final_indices', distances_initial_to_final_indices)
                    print(f'\t{matches} matches found from original initial {distances_initial_to_final.size()[0]} points and {distances_initial_to_final.size()[1]} final points.')
                    for match_index in range(matches):
                        initial_and_final_indices = distances_initial_to_final_indices[match_index]
                        #minimum_distances, minimum_initial_index_by_final = torch.min(distances_initial_to_final, dim=0)
                        #distance, final_index = torch.min(minimum_distances, dim=0)
                        #minimum_initial_index = minimum_initial_index_by_final[minimum_final_index]
                        #print(f'\tOptimal match of value {minimum_distance:0.3} from initial index {minimum_initial_index} and final index {minimum_final_index}')
                        coordinates_initial = coordinates_initial_image[:2, [indices_initial_keypoint[initial_and_final_indices[0]]]]
                        #print(coordinates_initial)
                        #print(f'\tCoordinate on initial image: {coordinates_initial}')
                        coord_initial = (coordinates_initial[0], coordinates_initial[1])
                        coordinates_final = coordinates_final_image[:2, [indices_final_keypoint[initial_and_final_indices[1]]]]
                        #print(f'\tCoordinate on final image: {coordinates_final}')
                        coord_final = (coordinates_final[0], coordinates_final[1])
                        distance = distances_initial_to_final[initial_and_final_indices[0], initial_and_final_indices[1]]
                        epipolar_match = EpipolarMatch(coord_initial, coord_final, distance)
                        keypoint_to_matches[keypoint_type].append(epipolar_match)
                else:
                    print(f'\tNo matches found')
        return keypoint_to_matches


    def draw_keypoint_matches(self, label, keypoint_to_matches, image_initial, image_final):
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


        cv2.imshow(f'{label} | Point in image 1', image_initial)
        cv2.namedWindow(f'{label} | Point in image 1')
        cv2.setMouseCallback(f'{label} | Point in image 1', print_coordinates)
        cv2.imshow(f'{label} | Epipolar line in second image', image_final)

class EpipolarMatch():

    def __init__(self, coord_initial, coord_final, minimum_distance):
        self.coord_initial = coord_initial
        self.coord_final = coord_final
        self.minimum_distance = minimum_distance

    def __repr__(self):
        return str(self.__dict__)

class Triangularizer():

    def __init__(self, camera_model, camera_robot_topology):
        self.config = Config()
        self.camera_matrix = camera_model.undistorted_camera_matrix @ camera_model.trans_robot_to_camera_rotation
        self.forward_kinematics = RobotForwardKinematics(camera_robot_topology)

    def triangularize(self, initial_state, final_state, keypoint_to_matches, triangularization_threshold=40.):

        initial_transformation = self.forward_kinematics.get_transformation(initial_state)
        final_transformation = self.forward_kinematics.get_transformation(final_state)
        initial_projection_matrix = self.camera_matrix @ initial_transformation[:3,:]
        final_projection_matrix = self.camera_matrix @ final_transformation[:3,:]
        points_predicted = []
        for keypoint in range(self.config.num_vertices):
            points_predicted_keypoint = []
            match_found = False
            if keypoint in keypoint_to_matches:
                matches = keypoint_to_matches[keypoint]
                for match in matches:
                    print('match', match)
                    coord_initial = MathUtils.skew([match.coord_initial[0], match.coord_initial[1], 1])
                    coord_final = MathUtils.skew([match.coord_final[0], match.coord_final[1], 1])
                    vector_space_initial = coord_initial @ initial_projection_matrix
                    vector_space_final   = coord_final   @ final_projection_matrix
                    vector_space = np.concatenate((vector_space_initial, vector_space_final), axis=0)
                    u, s, vh = np.linalg.svd(vector_space)
                    nullspace = vh[3, :]
                    point = nullspace / nullspace[3]

                    triangularization_error = s[3]
                    if triangularization_error < triangularization_threshold:
                        #print(point, s[3])
                        point = point[:3]
                        point_extended = np.append(point, np.array([1]), axis=0)
                        initial_point = self.camera_matrix @ (initial_transformation @ point_extended)[:3]
                        initial_point = initial_point / initial_point[2]
                        # Final
                        final_point = self.camera_matrix @ (final_transformation @ point_extended)[:3]
                        final_point = final_point / final_point[2]
                        mean_reprojection_error = ((initial_point[:2] - match.coord_initial) ** 2).mean() + \
                            ((final_point[:2] - match.coord_final) ** 2).mean()
                        print(point, s[3], mean_reprojection_error)

                        points_predicted_keypoint.append(point)
                        match_found = True
            points_predicted.append(points_predicted_keypoint)
            #if not match_found:
            #    points_predicted_keypoint.append(None)

        return points_predicted


def test():
    config = Config()

    image_initial_path = '/home/rusalka/Pictures/Webcam/first.jpg'
    image_final_path = '/home/rusalka/Pictures/Webcam/second.jpg'
    image_initial_raw = config.get_image_from_path(image_initial_path)
    image_final_raw = config.get_image_from_path(image_final_path)
    height, width = image_final_raw.shape[0], image_final_raw.shape[1]

    camera_topology = RobotTopology(l1=142, l2=142, l3=60, h1=50, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)


    #nothing_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(0.), angle_3=to_radians(0.))
    #nothing_transformation = robot_forward_kinamatics.get_transformation(nothing_state)
    #initial_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(0.), angle_3=to_radians(0.))
    #final_state   = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(-90.), angle_3=to_radians(0.))

    initial_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(70.), angle_3=to_radians(-20.))
    final_state   = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(-90.), angle_3=to_radians(20.))


    camera_model = CameraModel(width, height)
    fundamental_matrix_generator = FundamentalMatrixGenerator(camera_model, camera_topology)
    fundamental_matrix = fundamental_matrix_generator.generate_fundamental_matrix(initial_state, final_state)
    epipolar_line_generator = EpipolarLineGenerator(torch.from_numpy(fundamental_matrix))


    coords_initial = []
    coords_final = []
    # A
    #coords.append((371, 367))
    #coords.append((373, 428))
    #coords.append((292, 446))
    #coords.append((289, 380))
    #coords.append((270, 355))
    #coords.append((338, 344))
    #coords.append((271, 417))

    # B
    #coords.append((405, 368))
    #coords.append((408, 427))
    #coords.append((419, 400))
    #coords.append((328, 419))
    #coords.append((327, 362))
    #coords.append((347, 342))
    #coords.append((416, 345))



    image_initial = camera_model.undistort_image(image_initial_raw)
    image_final = camera_model.undistort_image(image_final_raw)
    threshold = 0.1
    # perform inference

    inferencer = Inferencer(distort=False, keep_dimensions=True, use_cache=False, \
        mode='silco', max_background_objects=1, max_foreground_objects=1)
    supports_folder = 'test/images/'
    supports = inferencer.get_supports_from_folder(supports_folder)
    images = [image_initial, image_final]
    supports = torch.cat([supports] * len(images) )
    query = inferencer.get_queries_from_opencv_images(images)
    original_height, original_width = image_final.shape[0], image_final.shape[1] #(480, 640, 3)


    visualize_query = query.clone()
    predicted_heatmaps = inferencer.get_model_inference(supports, query)
    target_height, targe_width = predicted_heatmaps.size()[2], predicted_heatmaps.size()[3] # torch.Size([2, 3, 96, 128])
    scale = original_width / targe_width
    print(f'Original h/w {original_height}, {original_width} => Target h/w {target_height}, {targe_width}. Scales: {original_width / targe_width} | {original_height / target_height}')


    keypoint_matcher = KeypointMatcher(epipolar_line_generator)
    keypoint_to_matches = keypoint_matcher.get_matches_from_predictions(predicted_heatmaps,\
        scale, prediction_threshold = 0.04, epipolar_threshold = 1.)
    print(keypoint_to_matches)
    keypoint_matcher.draw_keypoint_matches('Initial', keypoint_to_matches, image_initial, image_final)


    '''
    keypoint_to_matches = {0: [], 1: [], \
        2: [EpipolarMatch((torch.tensor(270), torch.tensor(350)), (torch.tensor(320), torch.tensor(335)), torch.tensor(0.0434, dtype=torch.float64))],
        3: [],
        4: [EpipolarMatch((torch.tensor(375), torch.tensor(435)), (torch.tensor(410), torch.tensor(430)), torch.tensor(0.0203, dtype=torch.float64))],
        5: [EpipolarMatch((torch.tensor(285), torch.tensor(440)), (torch.tensor(410), torch.tensor(430)), torch.tensor(0.0739, dtype=torch.float64))],
        6: [EpipolarMatch((torch.tensor(285), torch.tensor(380)), (torch.tensor(315), torch.tensor(360)), torch.tensor(0.1840, dtype=torch.float64))],
        7: [EpipolarMatch((torch.tensor(370), torch.tensor(370)), (torch.tensor(410), torch.tensor(365)), torch.tensor(0.5986, dtype=torch.float64))]
        }
    '''
    '''
    triangularizer = Triangularizer(camera_model, camera_topology)
    points_predicted = triangularizer.triangularize(initial_state, final_state, keypoint_to_matches)
    stats = {}
    topology = {
        'front': {
            'indices': set([4,5,6,7]),
            'f': lambda x: x[0]
        },
        'back': {
            'indices': set([0,1,2,3]),
            'f': lambda x: x[0]
        },
        'up': {
            'indices': set([0,1,4,5]),
            'f': lambda x: x[2]
        },
        'down': {
            'indices': set([2,3,6,7]),
            'f': lambda x: x[2]
        },
        'left': {
            'indices': set([1,2,5,6]),
            'f': lambda x: x[1]
        },
        'right': {
            'indices': set([0,3,4,7]),
            'f': lambda x: x[1]
        }
    }

    image_initial = camera_model.undistort_image(image_initial_raw)
    image_final = camera_model.undistort_image(image_final_raw)


    config = Config()
    camera_matrix = camera_model.undistorted_camera_matrix @ camera_model.trans_robot_to_camera_rotation
    forward_kinematics = RobotForwardKinematics(camera_topology)
    initial_transformation = forward_kinematics.get_transformation(initial_state)
    final_transformation = forward_kinematics.get_transformation(final_state)

    hues = torch.arange(start=0,end=179., step = 179 / (config.num_vertices + 1) )
    colors = [Color(hsl=(hue / 180, 1, 0.5)).rgb for hue in hues]
    colors = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]
    print('points_predicted', len(points_predicted))
    for idx, points_in_keypoint in enumerate(points_predicted):
        color = colors[idx]
        for point_predicted in points_in_keypoint:
            point_predicted = np.append(point_predicted, np.array([1]), axis=0)
            # Initial
            initial_point = camera_matrix @ (initial_transformation @ point_predicted)[:3]
            initial_point = initial_point / initial_point[2]
            image_initial = cv2.circle(image_initial, (int(initial_point[0]), int(initial_point[1])), 4, color, thickness=2)
            # Final
            final_point = camera_matrix @ (final_transformation @ point_predicted)[:3]
            final_point = final_point / final_point[2]
            print((int(final_point[0]), int(final_point[1])))
            image_final = cv2.circle(image_final, (int(final_point[0]), int(final_point[1])), 4, color, thickness=2)

    cv2.imshow(f'Reprojected | Point in Initial Image', image_initial)
    cv2.imshow(f'Reprojected | Point in Final Image', image_final)

    for idx, point_predicted in enumerate(points_predicted):
        if point_predicted is not None:
            for side in topology:
                if idx in topology[side]['indices']:
                    if side not in stats:
                        stats[side] = []
                    stats[side].append( ( topology[side]['f'](point_predicted), idx) )
    print(stats)

    #print('points_predicted: ', points_predicted)
    procrustes_problem_solver = ProcrustesProblemSolver()
    solution = procrustes_problem_solver.solve(points_predicted)
    if solution is not None:
        #print('solution: ', solution)
        #print(solution.transformation)
        rotation = solution.transformation[0:3, 0:3]

        z_vector = np.array([0, 0, 1])
        z_vector_rotated = rotation @ z_vector
        z_vector_rotated = z_vector_rotated
        z_vector_rotated[0] = 0
        z_vector_rotated = z_vector_rotated / np.linalg.norm(z_vector_rotated)
        angle_x = np.arccos(z_vector @ z_vector_rotated.T)  * 180. / np.pi
        print('Angle rotation in X: ', angle_x)

        x_vector = np.array([1, 0, 0])
        x_vector_rotated = rotation @ x_vector
        x_vector_rotated = x_vector_rotated
        x_vector_rotated[1] = 0
        x_vector_rotated = x_vector_rotated / np.linalg.norm(x_vector_rotated)
        angle_y = np.arccos(x_vector @ x_vector_rotated.T)  * 180. / np.pi
        print('Angle rotation in Y: ', angle_y)

        y_vector = np.array([0, 1, 0])
        y_vector_rotated = rotation @ y_vector
        y_vector_rotated = y_vector_rotated
        y_vector_rotated[2] = 0
        y_vector_rotated = y_vector_rotated / np.linalg.norm(y_vector_rotated)
        angle_z = np.arccos(y_vector @ y_vector_rotated.T)  * 180. / np.pi
        print('Angle rotation in Z: ', angle_z)

        #print(y_vector, y_vector_projected_to_xy_plane, )
        #rotation_axis, angle = AxisAng3(so3ToVec(MatrixLog3(rotation)))
        #angle = angle * 180. / np.pi
        #print('Rotation with angle and vector', angle, rotation_axis)
        print('solution.points',  solution.points)
        keypoint_to_matches = { keypoint: matches  if solution.points[keypoint] is not None else [] for keypoint, matches in keypoint_to_matches.items()}
        image_initial = camera_model.undistort_image(image_initial_raw)
        image_final = camera_model.undistort_image(image_final_raw)
        keypoint_matcher.draw_keypoint_matches('From procrustes',  keypoint_to_matches, image_initial, image_final)


    #cv2.imshow(f'Point in image 1', image_initial)
    #cv2.namedWindow(f'Point in image 1')
    #cv2.setMouseCallback(f'Point in image 1', print_coordinates)
    #cv2.imshow(f'Epipolar line in second image', image_final)


    #predicted_heatmaps = predicted_heatmaps * (predicted_heatmaps > 0.05)
    #for idx, image in enumerate(images):
    #    inferencer.display_results(f'Inference {idx}', visualize_query[idx: idx + 1], None, predicted_heatmaps[idx: idx + 1], threshold=0.0)

    '''
    cv2.waitKey(0)

if __name__ == '__main__':
    test()

