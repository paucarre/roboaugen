import numpy as np
import cv2
import glob
import os
import math
import sys
import click

from roboaugen.core.config import Config

class CameraCalibrator():

    def __init__(self):
        self.config = Config()

    @staticmethod
    def to_degrees(radians):
        degrees = ( radians * 180 ) / np.pi
        if(degrees < 0):
            degrees = degrees + 360
        return degrees

    @staticmethod
    def compute_euler_angles_from_rodrigues_vector(rvec):
        rotation_matrix, jacobian = cv2.Rodrigues(rvec)
        #Extract Euler angles from: https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
        angle_x = math.atan2(rotation_matrix[2,1],
            rotation_matrix[2,2])
        angle_y = math.atan2(-rotation_matrix[2,0],
            math.sqrt( (rotation_matrix[2,1] ** 2) + (rotation_matrix[2,2] ** 2) ))
        angle_z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
        return CameraCalibrator.to_degrees(angle_x), CameraCalibrator.to_degrees(angle_y), CameraCalibrator.to_degrees(angle_z)

    def draw(self, image, corners, image_points):
        image_points = np.int32(image_points).reshape(-1,2)
        image = cv2.drawContours(image, [image_points[:4]],-1,(0,255,0),-3)
        for i,j in zip(range(4),range(4,8)):
            image = cv2.line(image, tuple(image_points[i]), tuple(image_points[j]),(255),3)
        image = cv2.drawContours(image, [image_points[4:]],-1,(0,0,255),3)
        return image

    def get_camera_matrix(self, image):
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = (np.mgrid[0:7,0:6].T.reshape(-1,2) * 5) - 15
        #print(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imagepoints = []
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imagepoints.append(corners2)
            frame = cv2.drawChessboardCorners(image, (7,6), corners2,ret)
            ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = \
                cv2.calibrateCamera(objpoints, imagepoints, gray.shape[::-1], None, None)
            self.config.save_camera_parameters(camera_matrix, distortion_coefficients)
            return camera_matrix, distortion_coefficients, frame
        else:
            print('No chessboard corners found in image')
        return None, None, None

    def get_position_and_orientation(self, image, camera_matrix, distortion_coefficients):
        object_points = np.zeros(( 6 * 7, 3 ), np.float32)
        object_points[:,:2] = (np.mgrid [0:7, 0:6 ].T.reshape(-1, 2) * 5) - 15
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                        [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        chessboard_is_found, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        if chessboard_is_found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
            retval, rotation_vectors, translation_vectors, inliers = \
                cv2.solvePnPRansac(object_points, corners2, camera_matrix, distortion_coefficients)
            #imagepts, jac = cv2.projectPoints(axis, rotation_vectors, translation_vectors, camera_matrix, distortion_coefficients)
            #image = draw(image,corners2,imagepts)
            angle_x, angle_y, angle_z = CameraCalibrator.compute_euler_angles_from_rodrigues_vector(rotation_vectors)
            return angle_x, angle_y, angle_z, translation_vectors
        else:
            return None, None, None, None



@click.command()
@click.option("--camera", default=0, help="OpenCV Camera index")
def calibrate(camera):
    Q_KEY_CODE = 113
    video_capture = cv2.VideoCapture(camera)
    camera_matrix, distortion_coefficients = None, None
    camera_calibrator = CameraCalibrator()
    while video_capture.isOpened():
        ret, image = video_capture.read()
        cv2.imshow('image', image)
        if camera_matrix is None or distortion_coefficients is None:
            print(f'Neither camera matrix nor distortion coefficients found. Generating them')
            camera_matrix, distortion_coefficients, image_with_keypoints = camera_calibrator.get_camera_matrix(image)
            if image_with_keypoints is not None:
                cv2.imshow('image_with_keypoints', image_with_keypoints)
        if camera_matrix is not None or distortion_coefficients is not None:
            angle_x, angle_y, angle_z, translation = camera_calibrator.get_position_and_orientation(image, camera_matrix, distortion_coefficients)
            print(f'Orientation: {angle_x}, {angle_y}, {angle_z}. Translation: {translation}')
        print('Press any key to take another photo or press "q" to quit.')
        key = cv2.waitKey(0) & 0xff
        if key == Q_KEY_CODE:
            sys.exit(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
  calibrate()