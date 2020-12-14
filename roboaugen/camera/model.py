from roboaugen.core.config import Config
import numpy as np
import cv2
import base64
import struct

from robotcontroller.kinematics import RobotState

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
    #skew  = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0]
    @staticmethod
    def skew(x):
        return np.array([[0., -x[2], x[1]],
                        [x[2], 0., -x[0]],
                        [-x[1], x[0], 0.]])

