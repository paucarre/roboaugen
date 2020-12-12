
from pathlib import Path
import os
import cv2
import json
import traceback
import logging
import torch
import shutil
import numpy as np
import random

class Config():

  def __init__(self):
    self.project_root = f'{Path.home()}/work/roboaugen'
    self.train_dataset_dir = f"{self.project_root}/data/train"
    self.backgrounds_dir = f"{self.project_root}/data/backgrounds"
    self.cache_train_dir = f"{self.project_root}/data/cache_train"
    self.model_dir = f"{self.project_root}/models"
    self.train_state_path = f"{self.model_dir}/train_state"
    self.camera_dir = f"{self.project_root}/data/camera"

    self.camera_matrix_path = f"{self.camera_dir}/camera_matrix.numpy"
    self.distortion_coefficients_path = f"{self.camera_dir}/distortion_coefficients.numpy"

    self.channels_blocks = [24, 32, 64, 160]
    self.input_width = 256 # 512
    self.input_height = 256 #512
    self.dimensions_blocks = [(self.input_width / (4 * (2 ** index) ) , self.input_width / (4 * (2 ** index) )) for index in range(4)]
    self.save_each = 100
    self.save_each_epoch = 10
    self.channels_pointset = 16
    self.points_per_vertex = 10
    self.num_vertices = 8
    self.supports = 5
    self.context_classes = 2
    self.logger = self.get_logger(self)

  def save_camera_parameters(self, camera_matrix, distortion_coefficients):
    with open(self.camera_matrix_path, "wb") as camera_matrix_file:
      np.save(camera_matrix_file, camera_matrix)
    with open(self.distortion_coefficients_path, "wb") as dist_file:
      np.save(dist_file, distortion_coefficients)

  def load_camera_parameters(self):
    camera_matrix = None
    distortion_coefficients = None
    if os.path.exists(self.camera_matrix_path) and os.path.exists(self.distortion_coefficients_path):
        with open(self.camera_matrix_path, "rb") as camera_matrix_file:
            camera_matrix = np.load(camera_matrix_file)
        with open(self.distortion_coefficients_path, "rb") as dist_file:
            distortion_coefficients = np.load(dist_file)
    return camera_matrix, distortion_coefficients

  def save_mobilenet_model(self, model):
    self._save_model('mobilenet_model', model)

  def save_silco_model(self, model):
    self._save_model('silco_model', model)

  def save_higher_resolution_model(self, model):
    self._save_model('higher_resolution_model', model)

  def _save_model(self, model_type, model):
    self.logger.info(f'Saving model {model_type}...')
    torch.save(model, f'{self.model_dir}/{model_type}')
    self.logger.info(f'...{model_type} model saved.')

  def load_higher_resolution_model(self):
    return self._load_model('higher_resolution_model')

  def load_silco_model(self):
    return self._load_model('silco_model')

  def load_mobilenet_model(self):
    return self._load_model('mobilenet_model')

  def _load_model(self, model_type):
    path = f'{self.model_dir}/{model_type}'
    if os.path.exists(path):
      return torch.load(path, map_location=lambda storage, loc: storage)
    return None

  def get_logger(self, object_instance):
    log = logging.getLogger(type(object_instance).__name__)
    if(not log.hasHandlers()):
      log.setLevel(level=logging.INFO)
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      handler = logging.StreamHandler()
      handler.setLevel(level=logging.DEBUG)
      handler.setFormatter(formatter)
      log.addHandler(handler)
    return  log

  def get_sample_ids(self, object_type):
    sample_ids = os.listdir(f'{self.train_dataset_dir}/{object_type}')
    sample_ids = [int(id) for id in sample_ids]
    sample_ids.sort()
    return sample_ids

  def get_object_type_to_ids(self):
    object_types = os.listdir(self.train_dataset_dir)
    object_type_to_ids = {object_type : self.get_sample_ids(object_type) for object_type in object_types}
    return object_type_to_ids

  def sample_dir(self, object_type, sample_id):
    return f"{self.project_root}/data/train/{object_type}/{sample_id}"

  def get_background_sample(self, background_id=None):
    folder = f'{self.backgrounds_dir}/'
    background_ids = os.listdir(f'{self.backgrounds_dir}')
    if background_id is not None:
      background_id = random.choice(range(len(background_ids)))
    background_id = background_ids[background_id % len(background_ids)]
    image = self.get_image_from_path(f'{folder}{background_id}')
    if image is None:
      self.logger.error(f'Error loading image file in folder: {folder}{background_id}')
    width  = image.shape[0]
    height = image.shape[1]
    new_height, width = height, width
    if height > width:
      new_height, new_width = width, width
    else:
      new_height, new_width = height, height
    start_x = (max(new_width, width) - min(new_width, width)) // 2
    start_y = (max(new_height, height) - min(new_height, height)) // 2
    image = image[start_x : start_x + new_width, start_y :  start_y + new_height]
    image = cv2.resize(image, (self.input_height, self.input_width), interpolation = cv2.INTER_AREA)
    return image

  def get_image_sample_alpha(self, object_type, sample_id):
    folder = f'{self.sample_dir(object_type, sample_id)}/'
    image = self.get_image_from_path(f'{folder}data_no_background.png', True)
    if image is None:
      self.logger.error(f'Error loading image file in folder: {folder}')
      try:
        if os.path.exists(folder):
          shutil.rmtree(folder)
      except:
        traceback.print_exc()
    return image

  def get_image_from_path(self, image_sample_path, alpha=False):
    try:
      if alpha:
        return cv2.imread(image_sample_path, cv2.IMREAD_UNCHANGED)
      else:
        return cv2.imread(image_sample_path)
    except:
      self.logger.error(f'Error loading image file: {image_sample_path}')
      traceback.print_exc()
      return None

  def get_data_sample(self, object_type, sample_id):
    try:
      data_sample_path = f'{self.sample_dir(object_type, sample_id)}/data.json'
      if(os.path.exists(data_sample_path)):
        with open(data_sample_path, 'r') as f:
            json_data = f.read()
            json_data = json.loads(json_data)
            return json_data
      self.logger.error(f'Data file not found: {data_sample_path}')
      return None
    except:
      self.logger.error(f'Error loading data file: {data_sample_path}')
      traceback.print_exc()
      return None
