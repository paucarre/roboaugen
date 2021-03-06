from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics
from roboaugen.camera.model import CameraModel

import itertools
from enum import Enum, auto
import numpy as np
import cv2
import torch
from colour import Color
from scipy.linalg import svd
import json

'''
Coordinate basis for Model (same as robot)
    Z
    |
    |
    |_______ Y
   /
  /
 X
'''

class ProcrustesSolutionType(Enum):

    NO_SOLUTION_FOUND = auto()
    NEW_SET_OF_POINTS_FOUND = auto()
    SOLUTION_FOUND = auto()

class ProcrustesTripletSolution():

    def __init__(self, solution_type, points=[], keypoint_indices=[], rotation=None, translation=None):
        self.solution_type = solution_type
        self.points = points
        self.rotation = rotation
        self.translation = translation
        self.keypoint_indices = keypoint_indices

    def get_transformation(self):
        transformation = np.concatenate((self.rotation, np.expand_dims(self.translation, axis=0).T), axis=1)
        transformation = np.concatenate((transformation, np.array([[0, 0, 0, 1]])), axis=0)
        return transformation

    def get_degree_rotations_around_axis(self):
        z_vector = np.array([0, 0, 1])
        z_vector_rotated = self.rotation @ z_vector
        z_vector_rotated = z_vector_rotated
        z_vector_rotated[0] = 0
        z_vector_rotated = z_vector_rotated / np.linalg.norm(z_vector_rotated)
        angle_x = np.arccos(z_vector @ z_vector_rotated.T)  * 180. / np.pi

        x_vector = np.array([1, 0, 0])
        x_vector_rotated = self.rotation @ x_vector
        x_vector_rotated = x_vector_rotated
        x_vector_rotated[1] = 0
        x_vector_rotated = x_vector_rotated / np.linalg.norm(x_vector_rotated)
        angle_y = np.arccos(x_vector @ x_vector_rotated.T)  * 180. / np.pi

        y_vector = np.array([0, 1, 0])
        y_vector_rotated = self.rotation @ y_vector
        y_vector_rotated = y_vector_rotated
        y_vector_rotated[2] = 0
        y_vector_rotated = y_vector_rotated / np.linalg.norm(y_vector_rotated)
        angle_z = np.arccos(y_vector @ y_vector_rotated.T)  * 180. / np.pi

        return [angle_x, angle_y, angle_z]

    def __repr__(self):
        return str(self.__dict__)

    def to_json_string(self):
        angle_x, angle_y, angle_z = self.get_degree_rotations_around_axis()
        data = {
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist(),
            'angle_x': angle_x,
            'angle_y': angle_y,
            'angle_z': angle_z
        }
        return json.dumps(data,  indent = 4)

class ProcrustesSolution():

    def __init__(self, triplet_solutions):
        self.triplet_solutions = triplet_solutions

    def __repr__(self):
        return str(self.__dict__)

class ProcrustesProblemSolver():

    '''

    Cube
         [2]__________[3]
        /|            /|
       / |           / |
      /  |          /  |
     /   |         /   |
   [6]------------[7]  |
    |    [1]_______|__[0]
    |   /          |  /
    |  /           | /
    | /            |/
   [5]------------[4]

    '''
    def __init__(self):
        self.shape_points = np.array([ \
            [ -40., 40.,  0.],  # 5 - Front-Bottom-Left
            [ -40.,-40.,  0.],  # 4 - Front-Bottom-Right
            [ -40.,-40., 55.],
            [ -40., 40., 55.],  # 6 - Front-Top-Left
            [40., 40.,  0.],  # 1 - Back-Bottom-Left
            [40.,  -40.,  0.],  # 0 - Back-Bottom-Right
            [40.,  -40., 55.] , # 3
            [40., 40., 55.],  # 2 - Back-Top-Left
                ])
        center = self.shape_points.mean(0)
        self.shape_points = self.shape_points - center
        projection_angle = -np.pi / 4
        rot =  np.array(
            [
                [ np.cos(projection_angle),-np.sin(projection_angle),  0],
                [ np.sin(projection_angle), np.cos(projection_angle),  0.],
                [0, 0.,  1]
            ])
        #self.shape_points = (rot @ self.shape_points.T).T
        #self.shape_length = np.linalg.norm(self.shape_points, axis=1)
        self.shape_point_distances = self.compute_point_distances(self.shape_points)


    def compute_point_distances(self, points):
        points_replicated = np.expand_dims(points, axis=0)
        points_replicated = np.repeat(points_replicated, points.shape[0], axis=0)
        point_distances = points_replicated - np.swapaxes(points_replicated, 0, 1)
        point_distances = point_distances.reshape(point_distances.shape[0] * point_distances.shape[1], 3)
        point_distances = np.linalg.norm(point_distances, axis=1)
        point_distances = point_distances.reshape(points.shape[0], points.shape[0])
        # Add identity to remove diagonal zeros and the distance matrix can be used
        # for division without getting numerical problems
        point_distances = point_distances + np.identity(points.shape[0])
        return point_distances

    def create_orthonormal_basis(self, points):
        v_1 = points[0] - points[1]
        v_1 = v_1 / np.linalg.norm(v_1)
        v_2 = points[2] - points[1]
        v_2 = v_2 / np.linalg.norm(v_2)
        # reject v_1 from v_2
        v_2 = v_2 - (v_1 * (v_1.T @ v_2))
        v_2 = v_2 / np.linalg.norm(v_2)
        v_3 = np.cross(v_1, v_2)
        v_1 = np.expand_dims(v_1, axis=0)
        v_2 = np.expand_dims(v_2, axis=0)
        v_3 = np.expand_dims(v_3, axis=0)
        basis =  np.concatenate((v_1, v_2, v_3), axis=0)
        #basis = basis / np.linalg.det(basis)
        return basis

    def solve(self, points_predicted):
        solutions = self.solve_proposals(points_predicted)
        if(len(solutions.triplet_solutions) > 0):
            keypoint_indices = set(itertools.chain(*[solution.keypoint_indices for solution in solutions.triplet_solutions]))
            keypoint_indices = [idx for idx, points in enumerate(points_predicted) if idx in keypoint_indices]
            points = [points for idx, points in enumerate(points_predicted) if idx in keypoint_indices]
            translation_mean = np.array([np.expand_dims(solution.translation, axis=0) for solution in solutions.triplet_solutions]).mean(0)[0]
            rotation_mean = np.array([np.expand_dims(solution.rotation, axis=0) for solution in solutions.triplet_solutions]).mean(0)[0]
            solution = ProcrustesTripletSolution(ProcrustesSolutionType.SOLUTION_FOUND, points=points,
                keypoint_indices=keypoint_indices, rotation=rotation_mean, translation=translation_mean)
            return solution
        else:
            return None

    def solve_proposals(self, points_predicted):
        solutions = []
        keypoints_with_values = [idx for idx, points in enumerate(points_predicted) if len(points) > 0]
        for keypoint_1_idx in range(len(keypoints_with_values)):
            keypoint_1 = keypoints_with_values[keypoint_1_idx]
            for point_in_keypoint_1 in points_predicted[keypoint_1]:
                for keypoint_2_idx in range(keypoint_1_idx + 1, len(keypoints_with_values)):
                    keypoint_2 = keypoints_with_values[keypoint_2_idx]
                    for point_in_keypoint_2 in points_predicted[keypoint_2]:
                        for keypoint_3_idx in range(keypoint_2_idx + 1, len(keypoints_with_values)):
                            keypoint_3 = keypoints_with_values[keypoint_3_idx]
                            for point_in_keypoint_3 in points_predicted[keypoint_3]:
                                proposal_points = [None] * len(points_predicted)
                                #print(f'\t\tkeypoints: {keypoint_1}, {keypoint_2}, {keypoint_3}')
                                proposal_points[keypoint_1] = point_in_keypoint_1
                                proposal_points[keypoint_2] = point_in_keypoint_2
                                proposal_points[keypoint_3] = point_in_keypoint_3
                                #print(f'keypoints: {point_in_keypoint_1}, {point_in_keypoint_2}, {point_in_keypoint_3}')
                                solution = self.solve_for_point_triplet(proposal_points)
                                if solution is not None:
                                    solutions.append(solution)
        return ProcrustesSolution(solutions)

    def solution_attempt(self, predicted_points, length_threshold, standard_deviation_center_threshold=10):
        keypoints_with_values = [idx for idx, point in enumerate(predicted_points) if point is not None]
        if len(keypoints_with_values) == 3:
            print('solution attempt', keypoints_with_values)
            shape_points_with_matching_predictions = self.shape_points[keypoints_with_values]
            predicted_points_with_values = np.array([point for idx, point in enumerate(predicted_points) if point is not None])
            '''
            predicted_points_with_values = shape_points_with_matching_predictions # np.array([point for idx, point in enumerate(predicted_points) if point is not None])
            angle_distortion = 0.1
            rot = np.array(
                [
                    [ 1., 0., 0.],
                    [ 0., np.cos(angle_distortion), -np.sin(angle_distortion)],
                    [ 0., np.sin(angle_distortion), np.cos(angle_distortion)],
                ])
            predicted_points_with_values = (rot @ shape_points_with_matching_predictions.T).T + 20
            '''
            basis_for_shape = self.create_orthonormal_basis(shape_points_with_matching_predictions)
            basis_for_predicted = self.create_orthonormal_basis(predicted_points_with_values)
            rotation = basis_for_predicted.T @ basis_for_shape
            predicted_vectors_to_center = (rotation @ -shape_points_with_matching_predictions.T).T
            displacement = predicted_points_with_values + predicted_vectors_to_center
            standard_deviation_center = displacement.std(0).sum()
            if standard_deviation_center < standard_deviation_center_threshold:
                displacement = displacement.mean(0)
                centered_predictions = predicted_points_with_values - displacement
                shape_distances = self.compute_point_distances(shape_points_with_matching_predictions)
                perdictions_distances = self.compute_point_distances(centered_predictions)
                difference = np.abs(shape_distances - perdictions_distances) / (perdictions_distances + 0.001)
                mean_difference = difference.mean(0)
                keypoints_with_wrong_lengths = mean_difference > length_threshold
                if keypoints_with_wrong_lengths.sum() == 0:

                    return ProcrustesTripletSolution(ProcrustesSolutionType.SOLUTION_FOUND, predicted_points, keypoints_with_values, rotation, displacement)
        return ProcrustesTripletSolution(ProcrustesSolutionType.NO_SOLUTION_FOUND, predicted_points)

    def solve_for_point_triplet(self, predicted_points, length_threshold = 0.1):
        solution = None
        current_solution_type = ProcrustesSolutionType.NEW_SET_OF_POINTS_FOUND
        while current_solution_type == ProcrustesSolutionType.NEW_SET_OF_POINTS_FOUND:
            solution = self.solution_attempt(predicted_points, length_threshold)
            current_solution_type = solution.solution_type
        if current_solution_type == ProcrustesSolutionType.NO_SOLUTION_FOUND:
            return None
        else:
            return solution

    def visualize(self, solution, robot_topology, camera_holder_transformation, initial_state, final_state, image_initial_raw, image_final_raw):
        height, width = image_final_raw.shape[0], image_final_raw.shape[1]
        camera_model = CameraModel(width, height)
        config = Config()
        image_initial = camera_model.undistort_image(image_initial_raw)
        image_final = camera_model.undistort_image(image_final_raw)
        if solution is not None:
            hues = torch.arange(start=0,end=179., step = 179 / (config.num_vertices + 1) )
            colors = [Color(hsl=(hue/180, 1, 0.5)).rgb for hue in hues]
            colors = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]
            points_transformed = (self.shape_points @ solution.rotation) + solution.translation
            ones = np.expand_dims( np.array([1.] * points_transformed.shape[0]).T, axis=1)
            points_transformed = np.concatenate((points_transformed, ones), axis=1).T

            camera_matrix = camera_model.undistorted_camera_matrix @ camera_model.trans_robot_to_camera_rotation
            # TODO: this is a hack to adapt the focals on x and y directions so that there is distance accuracy.
            # This is to be fixed recalibrating properly the camera
            camera_matrix[0, 1] *= 0.78
            camera_matrix[1, 2] *= 0.68
            forward_kinematics = RobotForwardKinematics(robot_topology)
            initial_transformation = forward_kinematics.get_transformation(initial_state) @ camera_holder_transformation
            final_transformation = forward_kinematics.get_transformation(final_state) @ camera_holder_transformation
            initial_projection_matrix = camera_matrix @ np.linalg.inv(initial_transformation)[:3,:]
            initial_poits = initial_projection_matrix @ points_transformed
            initial_poits = initial_poits / initial_poits[2, :]
            final_projection_matrix = camera_matrix @ np.linalg.inv(final_transformation)[:3,:]
            final_points = final_projection_matrix @ points_transformed
            final_points = final_points / final_points[2, :]
            for keypoint in range(final_points.shape[1]):
                color = colors[keypoint]
                thickness = 1 if keypoint in solution.keypoint_indices else 1
                radius = 4 if keypoint in solution.keypoint_indices else 2
                initial_point = initial_poits[:, keypoint]
                final_point = final_points[:, keypoint]
                image_initial = cv2.circle(image_initial, (int(initial_point[0]), int(initial_point[1])), radius, color, thickness=thickness)
                image_final = cv2.circle(image_final, (int(final_point[0]), int(final_point[1])), radius, color, thickness=thickness)
        return image_initial, image_final



if __name__ == '__main__':
    solver = ProcrustesProblemSolver()
    solution  = solver.solve([ \
            [-40.,  40.,   0.],
            [-41., -40.,   0.],
            [-40., -43.,  155.],
            [-40.,  40.,  48.],
            None,
            None,
            None,
            None,
            None])
    print(solution)