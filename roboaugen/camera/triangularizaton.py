from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics

import numpy as np
import cv2
import torch
from colour import Color

class FundamentalMatrixGenerator():

    def __init__(self, camera_topology, width, height):
        self.camera_topology = camera_topology
        self.trans_robot_to_camera_rotation = np.array([ \
                                            [0., -1., 0.],
                                            [0.,  0.,-1.],
                                            [1.,  0., 0.]])
        self.config = Config()
        self.camera_matrix, self.distortion_coefficients = self.config.load_camera_parameters()
        self.undistorted_camera_matrix, self.region_of_interest = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficients,\
            (width, height), 0, (width, height) )#, centerPrincipalPoint= 1)
        print(self.region_of_interest)

    @staticmethod
    def skew(x):
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])

    def generate_fundamental_matrix(self, initial_state, final_state):
        initial_transformation = robot_forward_kinamatics.get_transformation(initial_state)
        print(initial_transformation)
        final_transformation = robot_forward_kinamatics.get_transformation(final_state)
        print(final_transformation)
        final_viewed_from_initial = np.linalg.inv(initial_transformation) @ final_transformation
        translation = final_viewed_from_initial[0:3, 3]
        rotation = final_viewed_from_initial[0:3, 0:3]
        print(translation)
        print(rotation)
        essential_matrix = FundamentalMatrixGenerator.skew(translation) @ rotation
        print(essential_matrix)
        inverse_camera_matrix = np.linalg.inv(self.undistorted_camera_matrix @ self.trans_robot_to_camera_rotation)
        fundamental_matrix = inverse_camera_matrix.T @ essential_matrix @ inverse_camera_matrix
        return fundamental_matrix

    def get_origin(self):
        camera_matrix_transformed = self.undistorted_camera_matrix# @ self.trans_robot_to_camera
        origin = camera_matrix_transformed @ np.array([0, 0, 1])
        return origin

    def undistort_image(self, image):
        return cv2.undistort(image, \
            self.camera_matrix, \
            self.distortion_coefficients, \
            None, \
            self.undistorted_camera_matrix)


class EpipolarLineGenerator():

    def __init__(self, fundamental_matrix):
        self.fundamental_matrix = fundamental_matrix

    def get_epipolar_line_in_initial_image_from_point_in_final(self, coordinate_x, coordinate_y):
        epipolar_line  = fundamental_matrix @ np.array([coordinate_x, coordinate_y, 1])
        return EpipolarLine(epipolar_line)

    def get_epipolar_line_in_final_image_from_point_in_initial(self, coordinate_x, coordinate_y):
        epipolar_line  = np.array([coordinate_x, coordinate_y, 1]).T @ fundamental_matrix
        return EpipolarLine(epipolar_line)

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

config = Config()

image_final_path = '/home/rusalka/Pictures/Webcam/first.jpg'
image_initial_path = '/home/rusalka/Pictures/Webcam/second.jpg'
image_initial = config.get_image_from_path(image_initial_path)
image_final = config.get_image_from_path(image_final_path)
height, width = image_final.shape[0], image_final.shape[1]

camera_topology = RobotTopology(l1=142, l2=142, l3=60, h1=50, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)
robot_forward_kinamatics = RobotForwardKinematics(camera_topology)

#nothing_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(0.), angle_3=to_radians(0.))
#nothing_transformation = robot_forward_kinamatics.get_transformation(nothing_state)

#initial_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(0.), angle_3=to_radians(0.))
#final_state   = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(-90.), angle_3=to_radians(0.))

final_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(70.), angle_3=to_radians(-20.))
initial_state   = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(-90.), angle_3=to_radians(20.))



fundamental_matrix_generator = FundamentalMatrixGenerator(camera_topology, width, height)
fundamental_matrix = fundamental_matrix_generator.generate_fundamental_matrix(initial_state, final_state)
epipolar_line_generator = EpipolarLineGenerator(fundamental_matrix)


coords = []
# A
#coords.append((371, 367))
#coords.append((373, 428))
#coords.append((292, 446))
#coords.append((289, 380))
#coords.append((270, 355))
#coords.append((338, 344))
#coords.append((271, 417))

# B
coords.append((405, 368))
coords.append((408, 427))
coords.append((419, 400))
coords.append((328, 419))
coords.append((327, 362))
coords.append((347, 342))
coords.append((416, 345))



image_initial = fundamental_matrix_generator.undistort_image(image_initial)
image_final = fundamental_matrix_generator.undistort_image(image_final)



hues = torch.arange(start=0,end=179., step = 179 / (len(coords) + 1) )  # H: 0-179, S: 0-255, V: 0-255.
colors = [Color(hsl=(hue/180, 1, 0.5)).rgb for hue in hues]
colors = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]
for idx, coord in enumerate(coords):
    color = colors[idx]
    coordinate_x, coordinate_y = coord
    epipoloar_line = epipolar_line_generator.get_epipolar_line_in_final_image_from_point_in_initial(coordinate_x, coordinate_y)
    x_init, y_init, x_final, y_final = epipoloar_line.from_image(image_final)
    image_final = cv2.line(image_final, (x_init, y_init), (x_final, y_final), color, thickness=2)
    image_initial = cv2.circle(image_initial, (int(coordinate_x), int(coordinate_y)), 4, color, thickness=2)


cv2.imshow(f'Point in image 1', image_initial)
cv2.namedWindow(f'Point in image 1')
cv2.setMouseCallback(f'Point in image 1', print_coordinates)
cv2.imshow(f'Epipolar line in second image', image_final)

cv2.waitKey(0)

