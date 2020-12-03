from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics

from enum import Enum, auto
import numpy as np
import cv2
import torch
from colour import Color


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

class ProcrustesSolution():

    def __init__(self, solution_type, points=[], transformation=None):
        self.solution_type = solution_type
        self.points = points
        self.transformation = transformation

    def __repr__(self):
        return str(self.__dict__)

class ProcrustesProblemSolver():

    def __init__(self):
        self.shape_points = np.array([ \
                [-40.,  40.,   0.],
                [-40., -40.,   0.],
                [-40., -40.,  55.],
                [-40.,  40.,  55.],
                [ 40.,  40.,   0.],
                [ 40., -40.,   0.],
                [ 40., -40.,  55.],
                [ 40.,  40.,  55.]])
        center = self.shape_points.mean(0)
        self.shape_points = self.shape_points - center
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


    def solution_attempt(self, predicted_points, length_threshold):
        #
        # predicted_points =  R @ points
        # R = U @ V.T where U @ S @ V.T = SVD( predicted_points @ points.T )
        #
        keypoints_with_values = [idx for idx, point in enumerate(predicted_points) if point is not None]
        if len(keypoints_with_values) >= 3:
            shape_points_with_matching_predictions = self.shape_points[keypoints_with_values]
            #shape_length_with_matching_predictions = self.shape_length[keypoints_with_values]
            predicted_points_with_values = np.array([point for idx, point in enumerate(predicted_points) if point is not None])
            predicted_points_mean = predicted_points_with_values.mean(0)
            shape_points_mean = shape_points_with_matching_predictions.mean(0)
            displacement = predicted_points_mean - shape_points_mean
            displacement = np.expand_dims(displacement, axis=0)
            displacements = np.repeat(displacement, shape_points_with_matching_predictions.shape[0], axis=0)
            centered_predictions = predicted_points_with_values - displacements
            shape_distances = self.compute_point_distances(shape_points_with_matching_predictions)
            perdictions_distances = self.compute_point_distances(centered_predictions)
            mean_difference = (np.abs(shape_distances - perdictions_distances) / perdictions_distances).mean(0)
            print('\tMean Difference:', mean_difference)

            #predictions_length = np.linalg.norm(centered_predictions, axis=1)
            #percentage_error_length = np.abs((predictions_length - shape_length_with_matching_predictions) / shape_length_with_matching_predictions)
            #keypoints_with_wrong_lengths = percentage_error_length > length_perc_threshold
            #print('\tPercentage_error_length', percentage_error_length)
            #print('\tCentered_predictions', centered_predictions)
            #print('\tSelf.shape_points', shape_points_with_matching_predictions)
            keypoints_with_wrong_lengths = mean_difference > length_threshold
            if keypoints_with_wrong_lengths.sum() > 0 or len(mean_difference) > 3:
                # remove the point with largest error and try again
                local_index_with_max_error = np.argmax(mean_difference)
                global_index_with_max_error = keypoints_with_values[local_index_with_max_error]
                predicted_points[global_index_with_max_error] = None
                return ProcrustesSolution(ProcrustesSolutionType.NEW_SET_OF_POINTS_FOUND, predicted_points)
            else:
                u, _, vh = np.linalg.svd(centered_predictions @ shape_points_with_matching_predictions.T)
                rotation = u @ vh
                transformation = np.concatenate((rotation, displacement.T), axis=1)
                transformation = np.concatenate((transformation, np.array([[0, 0, 0, 1]])), axis=0)
                return ProcrustesSolution(ProcrustesSolutionType.SOLUTION_FOUND, predicted_points, transformation)
        else:
            return ProcrustesSolution(ProcrustesSolutionType.NO_SOLUTION_FOUND, predicted_points)

    def solve(self, predicted_points, length_threshold = 0.1):
        solution = None
        current_solution_type = ProcrustesSolutionType.NEW_SET_OF_POINTS_FOUND
        while current_solution_type == ProcrustesSolutionType.NEW_SET_OF_POINTS_FOUND:
            solution = self.solution_attempt(predicted_points, length_threshold)
            current_solution_type = solution.solution_type
        if current_solution_type == ProcrustesSolutionType.NO_SOLUTION_FOUND:
            return None
        else:
            return solution

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