from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics

import numpy as np
import cv2
import torch

class FundamentalMatrixGenerator():

    def __init__(self, camera_topology, height, width):
        self.camera_topology = camera_topology
        #self.trans_robot_to_camera = np.array([ [0., -1., 0.],
        #                                        [0.,  0.,-1.],
        #                                        [1.,  0., 0.]])
        self.trans_robot_to_camera2 = np.array([ [0., -1., 0., 0.],
                                                [0.,  0.,-1., 0.],
                                                [1.,  0., 0., 0.],
                                                [0.,  0., 0., 1.]])

        self.config = Config()
        self.camera_matrix, self.distortion_coefficients = self.config.load_camera_parameters()
        self.undistorted_camera_matrix, self.region_of_interest = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficients,\
            (width, height), 0, (width, height))
        #print(self.distortion_coefficients)
        #print(height, width)

    @staticmethod
    def skew(x):
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])


    def generate_essential_matrix(self, initial_state, final_state):
        robot_transformation_initial =  robot_forward_kinamatics.get_transformation(initial_state) @ np.linalg.inv(self.trans_robot_to_camera2)
        robot_transformation_final = robot_forward_kinamatics.get_transformation(final_state) @ np.linalg.inv(self.trans_robot_to_camera2)
        print(np.linalg.inv(robot_transformation_initial) @ robot_transformation_initial)
        final_coordinates_viewed_from_initial = np.linalg.inv(robot_transformation_initial) @ robot_transformation_final
        print(final_coordinates_viewed_from_initial)

    def generate_fundamental_matrix(self, initial_state, final_state):
        robot_transformation_initial =  robot_forward_kinamatics.get_transformation(initial_state) @ np.linalg.inv(self.trans_robot_to_camera2)
        print('robot_transformation_initial\n', robot_transformation_initial)
        print('End effector position\n', robot_transformation_initial @ np.array([0, 0, 0, 1]))
        print('End effector orientation\n', robot_transformation_initial @ np.array([1, 0, 0, 0]))
        robot_transformation_final = robot_forward_kinamatics.get_transformation(final_state) @ np.linalg.inv(self.trans_robot_to_camera2)
        print('robot_transformation_final\n', robot_transformation_final)
        print('End effector position\n', robot_transformation_final @ np.array([0, 0, 0, 1]))
        print('End effector orientation\n', robot_transformation_final @ np.array([1, 0, 0, 0]))

        #viewer_initial_as_orgin_to_viewer_final = np.linalg.inv(robot_transformation_initial) @ robot_transformation_final
        #rotation = viewer_initial_as_orgin_to_viewer_final[0:3, 0:3]
        viewer_initial_as_orgin_to_viewer_final = self.trans_robot_to_camera2 @ np.linalg.inv(robot_transformation_initial) @ robot_transformation_final
        translation = viewer_initial_as_orgin_to_viewer_final[0:3, 3]

        print('viewer_initial_as_orgin_to_viewer_final\n', viewer_initial_as_orgin_to_viewer_final)
        #viewer_initial_as_orgin_to_viewer_final = np.linalg.inv(self.trans_robot_to_camera2) @ viewer_initial_as_orgin_to_viewer_final
        #viewer_initial_as_orgin_to_viewer_final = viewer_initial_as_orgin_to_viewer_final
        #print(viewer_initial_as_orgin_to_viewer_final)
        rotation = np.eye(3)#viewer_initial_as_orgin_to_viewer_final[0:3, 0:3]
        translation =  np.array([0, 1., 0.])  #viewer_initial_as_orgin_to_viewer_final[0:3, 3]
        print('rotation\n', rotation)
        print('translation\n', translation)

        print('[0, 0, 1]: ', self.undistorted_camera_matrix @ np.array([0, 0, 1]))
        print('[0, 1, 1]: ', self.undistorted_camera_matrix @ np.array([0, 1, 1]))
        print('[0, -1, 1]: ', self.undistorted_camera_matrix @ np.array([0, -1, 1]))
        print('[ 1, 0, 1]: ', self.undistorted_camera_matrix @ np.array([1, 0, 1]))
        print('[-1, 0, 1]: ', self.undistorted_camera_matrix @ np.array([-1, 0, 1]))


        fundamental_matrix = self._get_fundamental_matrix(rotation, translation)
        return fundamental_matrix

    def get_origin(self):
        camera_matrix_transformed = self.undistorted_camera_matrix# @ self.trans_robot_to_camera
        origin = camera_matrix_transformed @ np.array([0, 0, 1])
        return origin

    def _get_fundamental_matrix(self, rotation, translation):
        essential_matrix = FundamentalMatrixGenerator.skew(translation) @ rotation
        camera_matrix_transformed = self.undistorted_camera_matrix #@ self.trans_robot_to_camera
        inverse_camera_matrix = np.linalg.inv(camera_matrix_transformed)
        fundamental_matrix = inverse_camera_matrix.T @ essential_matrix @ inverse_camera_matrix
        return fundamental_matrix

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
        return EpipolarLine(epipolar_line[0], epipolar_line[1], epipolar_line[2])

    def get_epipolar_line_in_final_image_from_point_in_initial(self, coordinate_x, coordinate_y):
        epipolar_line  = np.array([coordinate_x, coordinate_y, 1]).T @ fundamental_matrix
        return EpipolarLine(epipolar_line[0], epipolar_line[1], epipolar_line[2])

class EpipolarLine():

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        print(a, b, c)

    def x(self, y):
        #if abs(self.a)> 1e-10:
        return - ( ( self.b * y ) + self.c ) / self.a
        #else:
        #    return 1e-10

    def y(self, x):
        return - ( ( self.a * x ) + self.c ) / self.b

    def from_image(self, image):
        height, width = image.shape[0], image.shape[1]
        if abs(self.a) < 1e-10:
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


config = Config()

image_initial_path = '/home/rusalka/Pictures/Webcam/first.jpg'
image_final_path = '/home/rusalka/Pictures/Webcam/second.jpg'
image_initial = config.get_image_from_path(image_initial_path)
image_final = config.get_image_from_path(image_final_path)
height, width = image_final.shape[0], image_final.shape[1]

trans_robot_to_camera = np.array([  [0., -1., 0., 0.],
                                    [0.,  0.,-1., 0.],
                                    [1.,  0., 0., 0.],
                                    [0.,  0., 0., 1.]])

trans_robot_to_camera_rotation = np.array([  [0., -1., 0.],
                                    [0.,  0.,-1.],
                                    [1.,  0., 0.]])


camera_topology = RobotTopology(l1=142, l2=142, l3=60, h1=50, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)
robot_forward_kinamatics = RobotForwardKinematics(camera_topology)

nothing_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(0.), angle_3=to_radians(0.))
nothing_transformation = robot_forward_kinamatics.get_transformation(nothing_state)
translation = nothing_transformation[0:4, 3]
rotation = nothing_transformation[0:3, 0:3]
#print(nothing_transformation)

translation_from_camera = trans_robot_to_camera @ translation
#print(translation_from_camera)
rotation_from_camera = trans_robot_to_camera_rotation @ rotation @ np.linalg.inv(trans_robot_to_camera_rotation)
#print(rotation_from_camera)



initial_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(90.), angle_3=to_radians(0.))
initial_transformation = robot_forward_kinamatics.get_transformation(initial_state)
print(initial_transformation)

final_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(-90.), angle_3=to_radians(0.))
final_transformation = robot_forward_kinamatics.get_transformation(final_state)
print(final_transformation)

final_viewed_from_initial = np.linalg.inv(initial_transformation) @ final_transformation
print(final_viewed_from_initial)

translation_from_camera = trans_robot_to_camera @ final_viewed_from_initial[0:4, 3]
rotation_from_camera = trans_robot_to_camera_rotation @ final_viewed_from_initial[0:3, 0:3] @ np.linalg.inv(trans_robot_to_camera_rotation)
print(translation_from_camera)
print(rotation_from_camera)

camera_matrix, distortion_coefficients = config.load_camera_parameters()
essential_matrix = FundamentalMatrixGenerator.skew(translation_from_camera[0:3]) @ rotation_from_camera
print(essential_matrix)
inverse_camera_matrix = np.linalg.inv(camera_matrix)
fundamental_matrix = inverse_camera_matrix.T @ essential_matrix @ inverse_camera_matrix
print(fundamental_matrix)

origin = camera_matrix @ np.array([0, 0, 1])
coordinate_x, coordinate_y = origin[0], origin[1]
coordinate_x, coordinate_y = coordinate_x + 260, coordinate_y + 185
epipolar_line = np.array([coordinate_x, coordinate_y , 1]).T @ fundamental_matrix
print(epipolar_line)
epipolar_line = EpipolarLine(epipolar_line[0], epipolar_line[1], epipolar_line[2])

undistorted_camera_matrix, region_of_interest = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients,\
            (width, height), 0, (width, height))


image_initial = cv2.undistort(image_initial, \
            camera_matrix, \
            distortion_coefficients, \
            None, \
            undistorted_camera_matrix)
image_final = cv2.undistort(image_final, \
            camera_matrix, \
            distortion_coefficients, \
            None, \
            undistorted_camera_matrix)

x_init, y_init, x_final, y_final = epipolar_line.from_image(image_final)
image_final = cv2.line(image_final, (x_init, y_init), (x_final, y_final), (0, 255, 0), thickness=2)
image_initial = cv2.circle(image_initial, ( int(coordinate_x), int(coordinate_y)     ), 4, (0, 255, 0), thickness=2)

cv2.imshow(f'Point in image 1', image_initial)
cv2.imshow(f'Epipolar line in second image', image_final)

cv2.waitKey(0)



#final_viewed_from_initial = np.linalg.inv(initial_transformation) @ final_transformation
#print(final_viewed_from_initial)



#fundamental_matrix_generator = FundamentalMatrixGenerator(camera_topology, height, width)
#e = fundamental_matrix_generator.generate_essential_matrix(state_1, state_2)
#robot_transformation_initial =  robot_forward_kinamatics.get_transformation(initial_state) @ np.linalg.inv(self.trans_robot_to_camera2)
#robot_transformation_final = robot_forward_kinamatics.get_transformation(final_state) @ np.linalg.inv(self.trans_robot_to_camera2)
#print(np.linalg.inv(robot_transformation_initial) @ robot_transformation_initial)
#final_coordinates_viewed_from_initial = np.linalg.inv(robot_transformation_initial) @ robot_transformation_final
#print(final_coordinates_viewed_from_initial)


'''


camera_topology = RobotTopology(l1=142, l2=142, l3=60, h1=50, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)
robot_forward_kinamatics = RobotForwardKinematics(camera_topology)
#state_1 = RobotState(linear_1=20, angle_1=to_radians(24.4819), angle_2=to_radians(94.050), angle_3=to_radians(338.198))
#state_2 = RobotState(linear_1=20, angle_1=to_radians(334.5181), angle_2=to_radians(265.9496), angle_3=to_radians(21.801))

state_1 = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(0.), angle_3=to_radians(0.))
state_2 = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(90.), angle_3=to_radians(0.))


fundamental_matrix_generator = FundamentalMatrixGenerator(camera_topology, height, width)
origin = fundamental_matrix_generator.get_origin()
print(f'Origin: {origin}')

# Top Left
#coordinate_x, coordinate_y = int(origin[0]) + 30, int(origin[1]) + 155
# Top Right
#coordinate_x, coordinate_y = int(origin[0]) + 100, int(origin[1]) + 145

coords = []

# Top Left
#coordinate_x, coordinate_y = int(origin[0]) , int(origin[1])
#coords.append((coordinate_x, coordinate_y ))
# Top Right
coordinate_x, coordinate_y = int(origin[0]) - 100, int(origin[1]) + 165
coords.append((coordinate_x, coordinate_y ))
#coordinate_x, coordinate_y = int(origin[0]) + 165, int(origin[1]) + 100
#coords.append((coordinate_x, coordinate_y ))

# Bottom Right
#coordinate_x, coordinate_y = int(origin[0]) + 132, int(origin[1]) + 178
#coords.append((coordinate_x, coordinate_y ))
# Bottom Left
#coordinate_x, coordinate_y = int(origin[0]) + 55, int(origin[1]) + 190
#coords.append((coordinate_x, coordinate_y ))



#coordinate_x, coordinate_y = int(origin[0]) - 260, int(origin[1]) + 195

#coordinate_x, coordinate_y = int(origin[0]) + 20, int(origin[1]) + 140

fundamental_matrix = fundamental_matrix_generator.generate_fundamental_matrix(state_1, state_2)
epipolar_line_generator = EpipolarLineGenerator(fundamental_matrix)

#cv2.imshow(f'Original Initial Image', image_initial)
#cv2.imshow(f'Original Final Image', image_final)

image_initial = fundamental_matrix_generator.undistort_image(image_initial)
image_final = fundamental_matrix_generator.undistort_image(image_final)

#image_initial = cv2.undistort(image_initial, fundamental_matrix_generator.camera_matrix, fundamental_matrix_generator.distortion_coefficients)
#image_final = cv2.undistort(image_final, fundamental_matrix_generator.camera_matrix, fundamental_matrix_generator.distortion_coefficients)

hues = torch.arange(start=0,end=179., step = 179 / (len(coords) + 1) )
for idx, coord in enumerate(coords):
    hue = hues[idx]
    coordinate_x, coordinate_y = coord
    epipoloar_line = epipolar_line_generator.get_epipolar_line_in_final_image_from_point_in_initial(coordinate_x, coordinate_y)
    x_init, y_init, x_final, y_final = epipoloar_line.from_image(image_final)
    image_final = cv2.line(image_final, (x_init, y_init), (x_final, y_final), (0, 255, 0), thickness=2)
    image_initial = cv2.circle(image_initial, (int(coordinate_x), int(coordinate_y)), 4, (0, 255, 0), thickness=2)

cv2.imshow(f'Point in image 1', image_initial)
cv2.imshow(f'Epipolar line in second image', image_final)

cv2.waitKey(0)
'''

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

