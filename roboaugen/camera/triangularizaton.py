from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics

import numpy as np
import cv2

class FundamentalMatrixGenerator():

    def __init__(self, camera_topology):
        self.camera_topology = camera_topology
        self.trans_robot_to_camera = np.array([ [0., 1., 0.],
                                    [0.,  0.,-1.],
                                    [1.,  0., 0.]])
        self.config = Config()
        self.camera_matrix, self.distortion_coefficients = self.config.load_camera_parameters()

    @staticmethod
    def skew(x):
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])

    def generate_fundamental_matrix(self, initial_state, final_state):
        robot_transformation_initial = robot_forward_kinamatics.get_transformation(initial_state)
        robot_transformation_final = robot_forward_kinamatics.get_transformation(final_state)
        viewer_initial_as_orgin_to_viewer_final = np.linalg.inv(robot_transformation_initial) @ robot_transformation_final
        rotation = viewer_initial_as_orgin_to_viewer_final[0:3, 0:3]
        translation = viewer_initial_as_orgin_to_viewer_final[0:3, 3]
        fundamental_matrix = self._get_fundamental_matrix(rotation, translation)
        return fundamental_matrix


    def _get_fundamental_matrix(self, rotation, translation):
        essential_matrix = FundamentalMatrixGenerator.skew(translation) @ rotation
        camera_matrix_transformed = self.camera_matrix @ self.trans_robot_to_camera
        inverse_camera_matrix = np.linalg.inv(camera_matrix_transformed)
        fundamental_matrix = inverse_camera_matrix.T @ essential_matrix @ inverse_camera_matrix
        return fundamental_matrix


class EpipolarLineGenerator():

    def __init__(self, fundamental_matrix):
        self.fundamental_matrix = fundamental_matrix

    def get_epipolar_line_in_initial_image_from_point_in_second(self, coordinate_x, coordinate_y):
        epipolar_line  = fundamental_matrix @ np.array([coordinate_x, coordinate_y, 1])
        return EpipolarLine(epipolar_line[0], epipolar_line[1], epipolar_line[2])

    def get_epipolar_line_in_final_image_from_point_in_first(self, coordinate_x, coordinate_y):
        epipolar_line  = np.array([coordinate_x, coordinate_y, 1]) @ fundamental_matrix
        return EpipolarLine(epipolar_line[0], epipolar_line[1], epipolar_line[2])

class EpipolarLine():

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def x(self, y):
        return - ( ( self.b * y ) + self.c ) / self.a

    def y(self, x):
        return - ( ( self.a * x ) + self.c ) / self.b

    def from_image(self, image):
        height, width = image.shape[0], image.shape[1]
        y_init = - height
        x_init = int(self.x(y_init))
        y_final = height
        x_final = int(self.x(y_final))
        return x_init, y_init, x_final, y_final



config = Config()

def change_coordinates_with_zero_at_center(x, y, height, width):
    return int((width / 2) + x), int((height / 2) - y)

def to_radians(degrees):
    return degrees * np.pi / 180.


camera_topology = RobotTopology(l1=142, l2=142, l3=60, h1=30, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)
robot_forward_kinamatics = RobotForwardKinematics(camera_topology)
state_1 = RobotState(linear_1=0, angle_1=to_radians(73.6583), angle_2=to_radians(63.6454), angle_3=to_radians(333.4349))
state_2 = RobotState(linear_1=0, angle_1=to_radians(296.3546), angle_2=to_radians(286.3417), angle_3=to_radians(26.5651))

image_initial_path = '/home/rusalka/Pictures/Webcam/230_200_1_m05.jpg'
image_final_path = '/home/rusalka/Pictures/Webcam/230_m200_1_05.jpg'
image_initial = config.get_image_from_path(image_initial_path)
image_final = config.get_image_from_path(image_final_path)
height, width = image_final.shape[0], image_final.shape[1]

coordinate_x, coordinate_y = 30, -147
coordinate_x, coordinate_y = change_coordinates_with_zero_at_center(coordinate_x, coordinate_y, height, width)

fundamental_matrix_generator = FundamentalMatrixGenerator(camera_topology)
fundamental_matrix = fundamental_matrix_generator.generate_fundamental_matrix(state_1, state_2)
epipolar_line_generator = EpipolarLineGenerator(fundamental_matrix)
epipoloar_line = epipolar_line_generator.get_epipolar_line_in_final_image_from_point_in_first(coordinate_x, coordinate_y)
x_init, y_init, x_final, y_final = epipoloar_line.from_image(image_final)


image_initial = cv2.circle(image_initial, (int(coordinate_x), int(coordinate_y)), 4, (0, 255, 0), thickness=2)
cv2.imshow(f'Point in image 1', image_initial)

cv2.line(image_final, (x_init, y_init), (x_final, y_final), (0, 255, 0), thickness=2)
cv2.imshow(f'Epipolar line in second image', image_final)

cv2.waitKey(0)

'''

supports_folder = '/home/rusalka/Pictures/Webcam/'
threshold = 0.05
inferencer = Inferencer()

supports_1, query_1, target_heatmaps_1, spatial_penalty_1 = inferencer.get_supports_and_query(sampleid=None, file=image_1, supports=supports_folder)
visualize_query_1 = query_1.clone()
visualize_suports_1 = supports_1.clone()
predicted_heatmaps_1 = inferencer.get_model_inference(supports_1, query_1)
inferencer.display_results('1', visualize_query_1, None, predicted_heatmaps_1, target_heatmaps_1, spatial_penalty_1, threshold)


supports_2, query_2, target_heatmaps_2, spatial_penalty_2 = inferencer.get_supports_and_query(sampleid=None, file=image_2, supports=supports_folder)
visualize_query_2 = query_2.clone()
visualize_suports_2 = supports_2.clone()
predicted_heatmaps_2 = inferencer.get_model_inference(supports_2, query_2)
inferencer.display_results('2', visualize_query_2, None, predicted_heatmaps_2, target_heatmaps_2, spatial_penalty_2, threshold)
cv2.waitKey(0)

threshold = 0.05
inferencer = Inferencer()
supports_2, query_2, target_heatmaps_2, spatial_penalty_2 = inferencer.get_supports_and_query(sampleid=None, file=image_2, supports=supports_folder)

'''

