import numpy as np
import cv2
import glob
import os
import math
import sys
import click
from roboaugen.camera.triangularizaton import Camera


@click.command()
@click.option("--idx", default=0, help="OpenCV Camera index")
@click.option("--destination_folder", default='data/backgrounds', help="Folder where images are stored")
def take_shot(idx, destination_folder):
    Q_KEY_CODE = 113
    S_KEY_CODE = 115
    video_capture = cv2.VideoCapture(idx)
    files = os.listdir(f'{destination_folder}')
    filenames = [os.path.splitext(file)[0] for file in files]
    files_with_numbers = [int(filename) for filename in filenames if filename.isnumeric()]
    max_image_id = 0
    if len(files_with_numbers) > 0:
        max_image_id = max(files_with_numbers) + 1
    while video_capture.isOpened():
        ret, image = video_capture.read()
        height, width = image.shape[0], image.shape[1]
        camera = Camera(width, height)
        image = camera.undistort_image(image)
        cv2.imshow('image', image)
        key = cv2.waitKey(1) & 0xff
        if key == Q_KEY_CODE:
            sys.exit(1)
        elif key == S_KEY_CODE:
            path = f'{destination_folder}/{max_image_id}.jpg'
            cv2.imwrite(path, image)
            print(f'Image saved in {path}')
            max_image_id += 1
    cv2.destroyAllWindows()


if __name__ == '__main__':
  take_shot()