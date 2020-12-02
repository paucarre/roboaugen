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
        self.model_points = np.array([ \
                [-40.,  40.,   0.],
                [-40., -40.,   0.],
                [-40., -40.,  55.],
                [-40.,  40.,  55.],
                [ 40.,  40.,   0.],
                [ 40., -40.,   0.],
                [ 40., -40.,  55.],
                [ 40.,  40.,  55.]])
        center = self.model_points.mean(0)
        self.model_points = self.model_points - center
        self.shape_lenght = np.linalg.norm(self.model_points, axis=1)

    def solution_attempt(self, predicted_points, lenght_perc_threshold = 0.1):
        #
        # predicted_points =  R @ points
        # R = U @ V.T where U @ S @ V.T = SVD( predicted_points @ points.T )
        #
        keypoints_with_values = [idx for idx, point in enumerate(predicted_points) if point is not None]
        if len(keypoints_with_values) >= 3:
            shape_points_with_matching_predictions = self.model_points[keypoints_with_values]
            shape_lenght_with_matching_predictions = self.shape_lenght[keypoints_with_values]
            predicted_points_with_values = np.array([point for idx, point in enumerate(predicted_points) if point is not None])
            predicted_points_mean = predicted_points_with_values.mean(0)
            shape_points_mean = shape_points_with_matching_predictions.mean(0)
            displacement = predicted_points_mean - shape_points_mean
            displacement = np.expand_dims(displacement, axis=0)
            displacements = np.repeat(displacement, shape_points_with_matching_predictions.shape[0], axis=0)
            centered_predictions = predicted_points_with_values - displacements
            predictions_lenght = np.linalg.norm(centered_predictions, axis=1)
            percentage_error_lenght = np.abs((predictions_lenght - shape_lenght_with_matching_predictions) / shape_lenght_with_matching_predictions)
            keypoints_with_wrong_lenghts = percentage_error_lenght > lenght_perc_threshold
            if keypoints_with_wrong_lenghts.sum() > 0:
                print(percentage_error_lenght)
                # remove the point with largest error and try again
                local_index_with_max_error = np.argmax(percentage_error_lenght)
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

    def solve(self, predicted_points, lenght_perc_threshold = 0.1):
        solution = None
        current_solution_type = ProcrustesSolutionType.NEW_SET_OF_POINTS_FOUND
        while current_solution_type == ProcrustesSolutionType.NEW_SET_OF_POINTS_FOUND:
            solution = self.solution_attempt(predicted_points, lenght_perc_threshold)
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