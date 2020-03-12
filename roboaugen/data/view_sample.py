import sys
import cv2
import numpy as np
from roboaugen.core.config import Config

def draw(sample_id):
    config = Config()
    image = config.get_image_sample(sample_id)
    target_image = config.get_target_image_sample(sample_id)
    data = config.get_data_sample(sample_id)
    transformed_points = data['projected_visible_vertices']
    for transformed_point in transformed_points:
        if transformed_point is not None:
            x, y = int(transformed_point[0]), int(transformed_point[1])
            cv2.circle(image, (x, y), 1, (0, 0, 255), cv2.FILLED, 4, 0)
            cv2.circle(target_image, (x, y), 1, (0, 0, 255), cv2.FILLED, 4, 0)
    cv2.imshow('image', image)
    cv2.imshow('target image', target_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    sample_id = sys.argv[1]
    draw(sample_id)