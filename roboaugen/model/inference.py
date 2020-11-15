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

config = Config()

def to_numpy_image(tensor):
  return tensor.transpose(0,2).transpose(0,1).numpy()

def display_heatmap(label, input_images, predicted_heatmaps, heatmap, threshold):
  colored_predictions = torch.zeros(predicted_heatmaps.size()[0], predicted_heatmaps.size()[1], 3, input_images.size()[2], input_images.size()[3])
  hues = torch.arange(start=0,end=179., step = 179 / (config.num_vertices + 1) )  # H: 0-179, S: 0-255, V: 0-255.
  colored_predictions[:, :, 1, : ,:] = 128
  for sample in range(predicted_heatmaps.size()[0]):
    predictions_in_sample = np.zeros((input_images.size()[2], input_images.size()[3], 3))
    for vertex in range(predicted_heatmaps.size()[1]):
      colored_predictions[sample, vertex, 0, : , :] = int(hues[vertex])
      predicted_heatmaps[sample, vertex, : , :] = predicted_heatmaps[sample, vertex, : , :] #/ predicted_heatmaps[sample, vertex, : , :].max()
      predicted_heatmaps[sample, vertex, : , :] = (predicted_heatmaps[sample, vertex, : , :] > threshold) #* predicted_heatmaps[sample, vertex, : , :]
      colored_predictions[sample, vertex, 2, : , :] = (predicted_heatmaps[sample, vertex, : , :] * 255.).byte()
      current_prediction = colored_predictions[sample, vertex, :, : , :].transpose(0,2).transpose(0,1).numpy()
      current_prediction = cv2.cvtColor(current_prediction, cv2.COLOR_HSV2BGR)
      predictions_in_sample += current_prediction
    predictions_in_sample = predictions_in_sample
    predictions_in_sample_mask = predictions_in_sample.sum(2)
    predictions_in_sample_mask = (predictions_in_sample_mask == 0).astype(np.float32)
    predictions_in_sample_mask = np.expand_dims(predictions_in_sample_mask, 2)
    input_image = to_numpy_image(input_images[sample])
    new_input_image = (predictions_in_sample_mask * input_image[:, :, :] * 255) + ((1.- predictions_in_sample_mask) * predictions_in_sample)
    new_input_image = (new_input_image).astype(np.uint8)
    #cv2.imshow(f'Input_Heatmap_{label}_{sample}', new_input_image)
    if heatmap:
      predictions_in_sample = predictions_in_sample.astype(np.uint8)
      #cv2.imshow(f'Heatmap_{label}_{sample}', predictions_in_sample)
    return new_input_image


@click.command()
@click.option("--sampleid", default=1, help="Sample ID.")
@click.option("--distort", default=False, help="Distort sample image.")
@click.option("--keep_dimensions", default=True, help="Keep original image dimensions.")
@click.option("--use_cache", default=False, help="Use image cache.")
@click.option("--file", default='', help="Path to image file.")
@click.option("--threshold", default=0.05, help="Positive threshold.")
@click.option("--supports", default='', help="Path to support images.")
@click.option("--heatmap", default=False, help="Display heatmap without image.")
@click.option("--mode", default='keypoints', help="Training mode: keypoints, silco.")
@click.option("--max_background_objects", default=0, help="Maximum number of background objects not in target for keypoints")
@click.option("--max_foreground_objects", default=0, help="Maximum number of foreground objects not in target for keypoints.")

def inference(sampleid, file, distort, keep_dimensions, use_cache, threshold, supports, heatmap, mode, max_background_objects, max_foreground_objects):
  targets = None
  dataset = ProjectedMeshDataset(config.input_height, config.input_width, config.num_vertices,
    max_background_objects, max_foreground_objects, distort=distort, keep_dimensions=keep_dimensions, use_cache=use_cache)
  if file == '':
    query, supports, target, spatial_penalty, _, _ = dataset.__getitem__(sampleid)
    targets = torch.cat([target.unsqueeze(0)], dim=0)
    spatial_penalty = torch.cat([spatial_penalty.unsqueeze(0)], dim=0)
    supports = supports.unsqueeze(0)
  else:
    support_ids = os.listdir(f'{supports}')
    support_ids = random.choices(support_ids, k=5)
    supports = [dataset.image_to_torch(cv2.imread(f'{supports}/{support_id}')).unsqueeze(0).unsqueeze(1) for support_id in support_ids]
    supports = torch.cat(supports, 1)
    query = dataset.image_to_torch(cv2.imread(file))
  query = torch.cat([query.unsqueeze(0)], dim=0)
  visualize_query = query.clone()
  visualize_suports = supports.clone()

  higher_resolution = config.load_higher_resolution_model().cpu()
  silco = config.load_silco_model().cpu()
  backbone = config.load_mobilenet_model().cpu()
  query_features, support_features = Silco.backbone_features(backbone, query, supports)
  if mode == 'silco':
    query_features, spatial_classifier, feature_classifier, spatial_supports = silco(query_features, support_features)

  #print([s.size() for s in spatial_supports], support_features.size())
  # scale, support, in coordinate, out coordinate
  #print(len(spatial_classifier), len(spatial_classifier[0]), [s[0].size() for s in spatial_classifier])

  '''
  scale_to_support_to_spatial = {}
  for scale_idx, spatial_classifier_scale in enumerate(spatial_classifier):
    scale_to_support_to_spatial[scale_idx] = []
    for support_idx, spatial_classifier_support in enumerate(spatial_classifier_scale):
      spatial_classifier_support = spatial_classifier_support.squeeze()
      mask = torch.zeros(spatial_classifier_support.size()[0])
      res = torch.argmax(spatial_classifier_support, 0).detach().cpu()
      #print(spatial_classifier_support.size(), res.size(), mask.size())
      mask[res] = spatial_classifier_support[res, :].sum().detach().cpu()
      mask = mask / mask.max()
      size = int(math.sqrt(mask.size()[0]))
      scale_to_support_to_spatial[scale_idx].append(mask.view(size, size))
    masks = np.hstack(scale_to_support_to_spatial[scale_idx])
    cv2.imshow(f'Mask {scale_idx}', masks)
  '''

    #object_alphas
  #print(object_alphas.shape)
  #cv2.imshow(f'Object Alphas', to_numpy_image(object_alphas))
  #print('Spatial Classifier', Trainer.compute_classifiers_entropy(spatial_classifier).item())
  #print('Feature Classifier', Trainer.compute_classifiers_entropy(feature_classifier).item())
  predicted_heatmaps = higher_resolution(query_features) ## updated_query_features)

  predicted_heatmaps = predicted_heatmaps.detach().cpu()
  predicted_heatmaps = F.interpolate(predicted_heatmaps, size=(visualize_query.size()[2], visualize_query.size()[3]), mode='bilinear')
  predictions = display_heatmap('Predictions', visualize_query, predicted_heatmaps, heatmap, threshold)
  if targets is not None:
    targets = display_heatmap('Targets', visualize_query, targets, heatmap, threshold)
    spatial_penalty = display_heatmap('Spatial Penalty', visualize_query, spatial_penalty, heatmap, threshold)
    images = np.hstack((targets, spatial_penalty, predictions))
    cv2.imshow(f'Targets - Spatial Penalty - Predictions', images)
  else:
    cv2.imshow(f'Predictions', predictions)
  #print(supports.size())#
  #print(len(spatial_supports), len(spatial_supports[0]), spatial_supports[0][0].size())
  #print(len(support_features), len(support_features[0]), support_features[0][0].size())
  #support_features.size(), spatial_supports.size())
  visualize_suports = visualize_suports.squeeze(0)
  visualize_suports = [to_numpy_image(visualize_suports[sample_idx]) for sample_idx in range(visualize_suports.size()[0])]
  visualize_suports = np.hstack(visualize_suports)
  cv2.imshow(f'Supports', visualize_suports)


  cv2.waitKey(0)

if __name__ == '__main__':
  inference()

