from roboaugen.data.dataset import ProjectedMeshDataset
from roboaugen.core.config import Config
from roboaugen.model.models import HigherResolutionNetwork, MobileNetV2
from roboaugen.model.silco import Silco
from roboaugen.train.trainer import Trainer

import cv2
import sys
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import click


class Inferencer():

  def __init__(self, distort=False, keep_dimensions=False, use_cache=False, \
      mode='silco', max_background_objects=1, max_foreground_objects=1, object_type=None):
    self.config = Config()
    self.distort = distort
    self.keep_dimensions = keep_dimensions
    self.use_cache = use_cache
    self.mode = mode
    self.max_background_objects = max_background_objects
    self.max_foreground_objects = max_foreground_objects
    self.object_type = object_type
    self.dataset = ProjectedMeshDataset(self.config.input_height, self.config.input_width, self.config.num_vertices,
      self.max_background_objects, self.max_foreground_objects, distort=self.distort,
      keep_dimensions=self.keep_dimensions, use_cache=self.use_cache, forced_object_type=object_type)
    self.higher_resolution = self.config.load_higher_resolution_model().cuda()
    self.silco = self.config.load_silco_model().cuda()
    self.backbone = self.config.load_mobilenet_model().cuda()

  @staticmethod
  def to_numpy_image(tensor):
    return tensor.transpose(0,2).transpose(0,1).numpy()

  def display_heatmap(self, label, input_images, predicted_heatmaps, threshold):
    colored_predictions = torch.zeros(predicted_heatmaps.size()[0], predicted_heatmaps.size()[1], 3, input_images.size()[2], input_images.size()[3])
    hues = torch.arange(start=0,end=179., step = 179 / (self.config.num_vertices + 1) )  # H: 0-179, S: 0-255, V: 0-255.
    colored_predictions[:, :, 1, : ,:] = 128
    predicted_heatmaps[:, :, : , :] = (predicted_heatmaps[:, :, : , :] > threshold)
    for sample in range(predicted_heatmaps.size()[0]):
      predictions_in_sample = np.zeros((input_images.size()[2], input_images.size()[3], 3))
      for vertex in range(predicted_heatmaps.size()[1]):
        colored_predictions[sample, vertex, 0, : , :] = int(hues[vertex])
        colored_predictions[sample, vertex, 2, : , :] = (predicted_heatmaps[sample, vertex, : , :] * 255.).byte()
        current_prediction = colored_predictions[sample, vertex, :, : , :].transpose(0,2).transpose(0,1).numpy()
        current_prediction = cv2.cvtColor(current_prediction, cv2.COLOR_HSV2BGR)
        predictions_in_sample += current_prediction
      predictions_in_sample = predictions_in_sample
      predictions_in_sample_mask = predictions_in_sample.sum(2)
      predictions_in_sample_mask = (predictions_in_sample_mask == 0).astype(np.float32)
      predictions_in_sample_mask = np.expand_dims(predictions_in_sample_mask, 2)
      input_image = Inferencer.to_numpy_image(input_images[sample])
      new_input_image = (predictions_in_sample_mask * input_image[:, :, :] * 255) + ((1.- predictions_in_sample_mask) * predictions_in_sample)
      new_input_image = (new_input_image).astype(np.uint8)
      #cv2.imshow(f'Input_Heatmap_{label}_{sample}', new_input_image)
      #if self.heatmap:
      #  predictions_in_sample = predictions_in_sample.astype(np.uint8)
        #cv2.imshow(f'Heatmap_{label}_{sample}', predictions_in_sample)
      return new_input_image

  def get_supports_and_query(self, sampleid, file, supports_path):
    target_heatmaps = None
    spatial_penalty = None
    supports = None
    if file is None or file == '':
      query, supports, target, spatial_penalty, _, _ = self.dataset.__getitem__(sampleid)
      if self.mode == 'keypoints':
        supports = None
      else:
        supports = supports.unsqueeze(0)
      query = torch.cat([query.unsqueeze(0)], dim=0)
      target_heatmaps = torch.cat([target.unsqueeze(0)], dim=0)
      spatial_penalty = torch.cat([spatial_penalty.unsqueeze(0)], dim=0)
    else:
      if self.mode != 'keypoints':
        if supports_path is not None and supports_path != '':
          supports = self.get_supports_from_folder(supports_path)
        else:
          supports = self.dataset.get_supports(sampleid, self.dataset.forced_object_type).unsqueeze(0)
      query = self.get_queries_from_opencv_images([cv2.imread(file)])
    return supports, query, target_heatmaps, spatial_penalty

  def get_supports_from_folder(self, supports_folder):
    support_ids = os.listdir(f'{supports_folder}')
    support_ids = random.choices(support_ids, k=5)
    supports = [self.dataset.image_to_torch(cv2.imread(f'{supports_folder}/{support_id}')).unsqueeze(0).unsqueeze(1) for support_id in support_ids]
    supports = torch.cat(supports, 1)
    return supports

  def get_queries_from_opencv_images(self, query_images):
    queries = [self.dataset.image_to_torch(image).unsqueeze(0) for image in query_images]
    queries = torch.cat(queries, dim=0)
    return queries

  def get_model_inference(self, supports, query):
    w = query.size()[2]
    h = query.size()[3]
    if supports is not None:
      supports = supports[:, :, :, :w, :h]
    query_features, support_features = Silco.backbone_features(self.backbone, query, supports)
    if self.mode != 'keypoints':
      query_features, spatial_classifier, feature_classifier, spatial_supports = self.silco(query_features, support_features)
    predicted_heatmaps = self.higher_resolution(query_features)
    return predicted_heatmaps

  def display_results(self, label, visualize_query, visualize_suports, predicted_heatmaps, threshold, target_heatmap=None, spatial_penalty=None):
    visual_targets, visual_predictions, visual_suports = None, None, None

    predicted_heatmaps = F.interpolate(predicted_heatmaps, size=(visualize_query.size()[2], visualize_query.size()[3]), mode='bilinear')
    predicted_heatmaps = predicted_heatmaps.detach().cpu()
    visual_predictions = self.display_heatmap('Predictions', visualize_query, predicted_heatmaps, threshold)
    if target_heatmap is not None:
      targets = self.display_heatmap('Targets', visualize_query, target_heatmap, threshold)
      spatial_penalty = self.display_heatmap('Spatial Penalty', visualize_query, spatial_penalty, threshold)
      images = np.hstack((targets, spatial_penalty, visual_predictions))
      visual_targets = images
    if visualize_suports is not None:
      visual_suports = visualize_suports.squeeze(0)
      visual_suports = [Inferencer.to_numpy_image(visual_suports[sample_idx]) for sample_idx in range(visual_suports.size()[0])]
      visual_suports = np.hstack(visual_suports)

    return visual_targets, visual_predictions, visual_suports

@click.command()
@click.option("--sampleid", default=1, help="Sample ID.")
@click.option("--distort", default=False, help="Distort sample image.")
@click.option("--keep_dimensions", default=True, help="Keep original image dimensions.")
@click.option("--use_cache", default=False, help="Use image cache.")
@click.option("--file", default='', help="Path to image file.")
@click.option("--threshold", default=0.1, help="Positive threshold.")
@click.option("--supports_path", default='', help="Path to support images.")
@click.option("--heatmap", default=False, help="Display heatmap without image.")
@click.option("--mode", default='keypoints', help="Training mode: keypoints, silco.")
@click.option("--max_background_objects", default=1, help="Maximum number of background objects not in target for keypoints")
@click.option("--max_foreground_objects", default=1, help="Maximum number of foreground objects not in target for keypoints.")
@click.option("--object_type", default='', help="Supports object type.")
def inference(sampleid, distort, keep_dimensions, use_cache, file, threshold, supports_path, heatmap, mode, max_background_objects, max_foreground_objects, object_type):
  object_type = None if object_type == '' else object_type
  inferencer = Inferencer(distort, keep_dimensions, use_cache, \
      mode, max_background_objects, max_foreground_objects, object_type)
  supports, query, target_heatmaps, spatial_penalty = inferencer.get_supports_and_query(sampleid, file, supports_path)
  if distort:
    query = inferencer.dataset.distort_image(query)
  #if distort:
  #  supports = torch.cat([inferencer.dataset.distort_image(supports[0, support_idx]).unsqueeze(0) for support_idx in range(supports.size()[1])], 0).unsqueeze(0)
  visualize_query = query.clone()

  visualize_suports = None
  if mode != 'keypoints':
    visualize_suports = supports.clone()
    supports = supports.cuda()
  query = query.cuda()
  predicted_heatmaps = inferencer.get_model_inference(supports, query)
  visual_targets, visual_predictions, visual_suports = inferencer.display_results(sampleid, visualize_query, visualize_suports, predicted_heatmaps, threshold, target_heatmaps, spatial_penalty)
  if visual_targets is not None:
    cv2.imshow(f'{sampleid} - Targets - Spatial Penalty - Predictions', visual_targets)
  #cv2.imshow(f'{sampleid} - Targets - Spatial Penalty - Predictions', visualize_query)
  #cv2.imshow(f'{sampleid} Supports', visual_suports)
  cv2.imshow(f'{sampleid} Predictions', visual_predictions)
  cv2.waitKey(0)

if __name__ == '__main__':
  inference()

