from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from robotcontroller.kinematics import RobotTopology, RobotState
from robotcontroller.ik import IkSolver, RobotForwardKinematics

import numpy as np
import cv2
import torch
from colour import Color

class Camera():

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

class FundamentalMatrixGenerator():

    def __init__(self, camera, camera_topology):
        self.camera = camera
        self.camera_topology = camera_topology

    @staticmethod
    def skew(x):
        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])

    def generate_fundamental_matrix(self, initial_state, final_state):
        initial_transformation = robot_forward_kinamatics.get_transformation(initial_state)
        final_transformation = robot_forward_kinamatics.get_transformation(final_state)
        final_viewed_from_initial = np.linalg.inv(initial_transformation) @ final_transformation
        translation = final_viewed_from_initial[0:3, 3]
        rotation = final_viewed_from_initial[0:3, 0:3]
        essential_matrix = FundamentalMatrixGenerator.skew(translation) @ rotation
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
        epipolar_line  = fundamental_matrix @ np.array([coordinate_x, coordinate_y, 1])
        return EpipolarLine(epipolar_line)

    def get_epipolar_line_in_final_image_from_point_in_initial(self, coordinate_x, coordinate_y):
        epipolar_line  = np.array([coordinate_x, coordinate_y, 1]).T @ fundamental_matrix
        return EpipolarLine(epipolar_line)

    def get_epipolar_lines_in_final_image_from_points_in_initial(self, points):
        epipolar_lines  = points.T @ fundamental_matrix
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


if __name__ == '__main__':
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


    camera = Camera(width, height)
    fundamental_matrix_generator = FundamentalMatrixGenerator(camera, camera_topology)
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


    image_initial = camera.undistort_image(image_initial)
    image_final = camera.undistort_image(image_final)
    threshold = 0.05
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
    factor = original_width / targe_width
    print(f'Original h/w {original_height}, {original_width} => Target h/w {target_height}, {targe_width}. Factors: {original_width / targe_width} | {original_height / target_height}')

    #mean_probabilities = predicted_heatmaps.view(predicted_heatmaps.size()[0], -1).transpose(1, 0).mean(0)
    #std_probabilities = predicted_heatmaps.view(predicted_heatmaps.size()[0], -1).transpose(1, 0).std(0)
    #size = predicted_heatmaps.size()[2] * predicted_heatmaps.size()[3]
    #mean_probabilities = mean_probabilities[:, None, None, None]
    #std_probabilities = std_probabilities[:, None, None, None]
    predicted_heatmaps = predicted_heatmaps * (predicted_heatmaps > threshold)# * mean_probabilities * (size * torch.sqrt(std_probabilities))))

    coordinate_predictions = (predicted_heatmaps > 0.).nonzero()
    coordinate_predictions_x = coordinate_predictions[:, 3] * factor
    coordinate_predictions_y = coordinate_predictions[:, 2] * factor
    coordinate_predictions[:, 2] = coordinate_predictions_x
    coordinate_predictions[:, 3] = coordinate_predictions_y
    #print(coordinate_predictions)

    # assume two images
    indices_initial_image = (coordinate_predictions[:, 0] == 0).nonzero()[:, 0]
    predictions_initial_image = coordinate_predictions[indices_initial_image, :]
    coordinates_initial_image = predictions_initial_image[:, 2:]
    ones = coordinates_initial_image.new_ones(coordinates_initial_image[:, 0:1].size())
    coordinates_initial_image = torch.cat([coordinates_initial_image, ones], 1).transpose(0, 1)

    indices_final_image = (coordinate_predictions[:, 0] == 1).nonzero()[:, 0]
    predictions_final_image = coordinate_predictions[indices_final_image, :]
    coordinates_final_image = predictions_final_image[:, 2:]
    ones = coordinates_final_image.new_ones(coordinates_final_image[:, 0:1].size())
    coordinates_final_image = torch.cat([coordinates_final_image, ones], 1).transpose(0, 1)

    epipolar_lines_in_final = epipolar_line_generator.\
        get_epipolar_lines_in_final_image_from_points_in_initial(coordinates_initial_image)
    #print(coordinates_initial_image.numpy().T)
    #print(epipolar_lines_in_final.shape)

    #print(predictions_initial_image)
    threshold = 10.
    for keypoint_type in range(predicted_heatmaps.size()[1]):
        print(f'Getting matches for keypoint type index {keypoint_type}')
        indices_initial_keypoint = (predictions_initial_image[:, 1] == keypoint_type).nonzero()
        indices_final_keypoint = (predictions_final_image[:, 1] == keypoint_type).nonzero()
        if indices_initial_keypoint.size()[0] > 0 and indices_final_keypoint.size()[0] > 0:
            indices_initial_keypoint = indices_initial_keypoint[:, 0]
            indices_final_keypoint = indices_final_keypoint[:, 0]
            epipolar_lines = epipolar_lines_in_final[indices_initial_keypoint].double()
            points = coordinates_final_image[:, indices_final_keypoint].double()
            print(points[0,:].max())
            print(points[1,:].max())
            distances_initial_to_final = epipolar_lines @ points
            distances_initial_to_final = torch.sqrt(distances_initial_to_final ** 2)
            distances_initial_to_final_indices = (distances_initial_to_final < threshold).nonzero()
            matches = distances_initial_to_final_indices.size()[0]
            if matches > 0:
                print(f'\t{matches} matches found from original initial {distances_initial_to_final.size()[0]} points and {distances_initial_to_final.size()[1]} final points.')
                minimum_distances, minimum_initial_index_by_final = torch.min(distances_initial_to_final, dim=0)
                minimum_distance, minimum_final_index = torch.min(minimum_distances, dim=0)
                minimum_initial_index = minimum_initial_index_by_final[minimum_final_index]
                print(f'\tOptimal match of value {minimum_distance:0.3} from initial index {minimum_initial_index} and final index {minimum_final_index}')
                coordinates_initial = predictions_initial_image[indices_initial_keypoint[minimum_initial_index], 2:]
                print(f'\tCoordinate on initial image: {coordinates_initial}')
                coords_initial.append((coordinates_initial[0], coordinates_initial[1]))
                coordinates_final = predictions_final_image[indices_final_keypoint[minimum_final_index], 2:]
                print(f'\tCoordinate on final image: {coordinates_final}')
                coords_final.append((coordinates_final[0], coordinates_final[1]))
            else:
                print(f'\tNo matches found')






    for idx, image in enumerate(images):
        inferencer.display_results(f'Inference {idx}', visualize_query[idx: idx + 1], None, predicted_heatmaps[idx: idx + 1], threshold=0.0)
    cv2.waitKey(0)

    hues = torch.arange(start=0,end=179., step = 179 / (len(coords_initial) + 1) )  # H: 0-179, S: 0-255, V: 0-255.
    colors = [Color(hsl=(hue/180, 1, 0.5)).rgb for hue in hues]
    colors = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]
    for idx, coord in enumerate(coords_initial):
        color = colors[idx]
        coordinate_x, coordinate_y = coord
        epipoloar_line = epipolar_line_generator.get_epipolar_line_in_final_image_from_point_in_initial(coordinate_x, coordinate_y)
        x_init, y_init, x_final, y_final = epipoloar_line.from_image(image_final)
        image_final = cv2.line(image_final, (x_init, y_init), (x_final, y_final), color, thickness=2)
        image_initial = cv2.circle(image_initial, (int(coordinate_x), int(coordinate_y)), 4, color, thickness=2)
        coordinate_x, coordinate_y = coords_final[idx]
        image_final = cv2.circle(image_final, (int(coordinate_x), int(coordinate_y)), 4, color, thickness=2)


    cv2.imshow(f'Point in image 1', image_initial)
    cv2.namedWindow(f'Point in image 1')
    cv2.setMouseCallback(f'Point in image 1', print_coordinates)
    cv2.imshow(f'Epipolar line in second image', image_final)

    cv2.waitKey(0)
