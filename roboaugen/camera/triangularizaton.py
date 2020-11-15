from roboaugen.core.config import Config
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics
import numpy as np

def to_radians(degrees):
    return degrees * np.pi / 180.

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

config = Config()
camera_matrix, distortion_coefficients = config.load_camera_parameters()
#print(camera_matrix)

camera_topology = RobotTopology(l1=142, l2=142, l3=60, h1=30, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)
robot_forward_kinamatics = RobotForwardKinematics(camera_topology)

state_1 = RobotState(linear_1=0, angle_1=to_radians(73.6583), angle_2=to_radians(63.6454), angle_3=to_radians(333.4349))
robot_transformation_1 = robot_forward_kinamatics.get_transformation(state_1)
robot_end_effector_1 = robot_transformation_1 @ np.array([0, 0, 0, 1])
#print(robot_transformation_1)

state_2 = RobotState(linear_1=0, angle_1=to_radians(296.3546), angle_2=to_radians(286.3417), angle_3=to_radians(26.5651))
robot_transformation_2 = robot_forward_kinamatics.get_transformation(state_2)
robot_end_effector_2 = robot_transformation_2 @ np.array([0, 0, 0, 1])
#print(robot_transformation_2)

point_2_seen_by_point_1 = np.linalg.inv(robot_transformation_1) @ robot_end_effector_2

viewer_1_as_orgin_to_viewer_2 = np.linalg.inv(robot_transformation_1) @ robot_transformation_2

rotation = viewer_1_as_orgin_to_viewer_2[0:3, 0:3]
translation = skew(viewer_1_as_orgin_to_viewer_2[0:3, 3])
essential_matrix = translation @ rotation
inverse_camera_matrix = np.linalg.inv(camera_matrix)
fundamental_matrix = inverse_camera_matrix.T @ essential_matrix @ inverse_camera_matrix


inferencer = Inferencer(sampleid, file, distort, keep_dimensions, use_cache, \
    threshold, supports, heatmap, mode, max_background_objects, max_foreground_objects)
predicted_heatmaps, targets, visualize_query, visualize_suports = inferencer.get_model_inference()
predictions = inferencer.display_heatmap('Predictions', visualize_query, predicted_heatmaps)

print(fundamental_matrix)
print(fundamental_matrix @ np.array([0, 0, 1]))
