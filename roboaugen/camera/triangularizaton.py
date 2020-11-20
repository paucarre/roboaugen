from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics

import numpy as np
import cv2

def to_radians(degrees):
    return degrees * np.pi / 180.

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

trans_robot_to_camera = np.array([ [0., -1., 0.],
                                    [0.,  0.,-1.],
                                    [1.,  0., 0.]])

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
#robot_end_effector_2 = robot_transformation_2 @ np.array([0, 0, 0, 1])
#print(robot_transformation_2)
#point_2_seen_by_point_1 = np.linalg.inv(robot_transformation_1) @ robot_end_effector_2

viewer_1_as_orgin_to_viewer_2 = np.linalg.inv(robot_transformation_1) @ robot_transformation_2

rotation = viewer_1_as_orgin_to_viewer_2[0:3, 0:3]
translation = viewer_1_as_orgin_to_viewer_2[0:3, 3]
print('rotation', rotation)
print('translation', translation)


essential_matrix = skew(translation) @ rotation
camera_matrix_transformed = camera_matrix @ trans_robot_to_camera
inverse_camera_matrix = np.linalg.inv(camera_matrix_transformed)
fundamental_matrix = inverse_camera_matrix.T @ essential_matrix @ inverse_camera_matrix

print('camera_matrix', camera_matrix)
point_at_origin = camera_matrix @ np.array([0, 0, 1])
point_at_origin = point_at_origin / point_at_origin[2]
print('point at origin', point_at_origin)

'''
extended_camera_matrix = np.c_[ camera_matrix, np.zeros(3) ]
print('camera matrix', extended_camera_matrix)

point_in_3d = np.array([0, 0, 100, 1])
print('point_in_3d', point_in_3d)
point_in_2d = extended_camera_matrix @ point_in_3d
point_in_2d = point_in_2d / point_in_2d[2]
print('point_in_2d', point_in_2d)

'''

supports_folder = '/home/rusalka/Pictures/Webcam/'
threshold = 0.05
inferencer = Inferencer()

image_1 = '/home/rusalka/Pictures/Webcam/230_200_1_m05.jpg'

'''
supports_1, query_1, target_heatmaps_1, spatial_penalty_1 = inferencer.get_supports_and_query(sampleid=None, file=image_1, supports=supports_folder)
visualize_query_1 = query_1.clone()
visualize_suports_1 = supports_1.clone()
predicted_heatmaps_1 = inferencer.get_model_inference(supports_1, query_1)
inferencer.display_results('1', visualize_query_1, None, predicted_heatmaps_1, target_heatmaps_1, spatial_penalty_1, threshold)
'''

image_2 = '/home/rusalka/Pictures/Webcam/230_m200_1_05.jpg'

'''
supports_2, query_2, target_heatmaps_2, spatial_penalty_2 = inferencer.get_supports_and_query(sampleid=None, file=image_2, supports=supports_folder)
visualize_query_2 = query_2.clone()
visualize_suports_2 = supports_2.clone()
predicted_heatmaps_2 = inferencer.get_model_inference(supports_2, query_2)
inferencer.display_results('2', visualize_query_2, None, predicted_heatmaps_2, target_heatmaps_2, spatial_penalty_2, threshold)
cv2.waitKey(0)
'''

# working (0, 200)
image_2_full = config.get_image_from_path(image_2)
height, width = image_2_full.shape[0], image_2_full.shape[1]
def change_coordinates_with_zero_at_center(x, y, height, width):
    return int((width / 2) + x), int((height / 2) - y)


coordinate_x, coordinate_y = 0, -200
#coordinate_x, coordinate_y = -200, -200
#coordinate_x, coordinate_y = 0, 0
coordinate_x, coordinate_y = change_coordinates_with_zero_at_center(coordinate_x, coordinate_y, height, width)
print('Pixel coordinates: ', coordinate_x, coordinate_y)
epipolar_line_in_second_image  = fundamental_matrix @ np.array([coordinate_x, coordinate_y, 1])
a = epipolar_line_in_second_image[0]
b = epipolar_line_in_second_image[1]
c = epipolar_line_in_second_image[2]
print(a, b, c)
# ax + by + c = 0 -> x = (-by -c) / a | y = (-ax - c) / b

x_compute = lambda y: ( ( -b * y ) - c ) / a
#y_compute = lambda x: ( ( -a * x ) - c ) / b

supports_folder = '/home/rusalka/Pictures/Webcam/'

'''
threshold = 0.05
inferencer = Inferencer()
supports_2, query_2, target_heatmaps_2, spatial_penalty_2 = inferencer.get_supports_and_query(sampleid=None, file=image_2, supports=supports_folder)
'''


image = image_2_full
y_init = - height
x_init = int(x_compute(y_init))
y_final = height
x_final = int(x_compute(y_final))


#x_init, y_init = change_coordinates_with_zero_at_center(x_init, y_init, height, width )
#x_final, y_final = change_coordinates_with_zero_at_center(x_final, y_final, height, width )
#print(height, width )
#print(y_init, x_init)
#print(y_final, x_final)


image_1_full = config.get_image_from_path(image_1)
#print(image_1_full.shape)
image_1_full = cv2.circle(image_1_full,(int(coordinate_x), int(coordinate_y)), 4, (0, 255, 0), thickness=2)
cv2.imshow(f'Point in image 1', image_1_full)


cv2.line(image_2_full, (x_init, y_init), (x_final, y_final), (0, 255, 0), thickness=2)
cv2.imshow(f'Epipolar line in second image', image_2_full)

cv2.waitKey(0)
