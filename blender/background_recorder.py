import numpy as np
import cv2
import os
import time
from pathlib import Path

backgrounds_folder = f'{Path.home()}/work/roboaugen/data/backgrounds'
test_image_folder = f'{Path.home()}/work/roboaugen/test/images'

def get_next_id_in_folder(folder):
    ids = os.listdir(folder)
    if len(ids) > 0:
        ids = [int(id.replace('.jpg', '')) for id in ids]
        ids.sort(reverse = True)
        return ids[0] + 1
    else:
        return 0

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_EXPOSURE, 0.25)
background_id = get_next_id_in_folder(backgrounds_folder)
test_id = get_next_id_in_folder(test_image_folder)
while video_capture.isOpened():
    is_success, frame = video_capture.read()
    if is_success:
        cv2.imshow('Current Frame', frame)
        key_pressed = cv2.waitKey(10)
        if key_pressed == ord('n'):
            print('Skipping image, going to next image...')
        elif key_pressed == ord('s'):
            file_path = f'{backgrounds_folder}/{background_id}.jpg'
            print(f'Saving background image as {file_path}...')
            cv2.imwrite(file_path, frame)
            background_id += 1
        elif key_pressed == ord('t'):
            file_path = f'{test_image_folder}/{test_id}.jpg'
            print(f'Saving test as {file_path}...')
            cv2.imwrite(file_path, frame)
            test_id += 1
        elif key_pressed == ord('q'):
            break
cv2.destroyAllWindows()

