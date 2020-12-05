from roboaugen.core.config import Config
from roboaugen.data.dataset import ProjectedMeshDataset
from roboaugen.model.models import HigherResolutionNetwork, MobileNetWrapper
from roboaugen.model.silco import Silco
from  torch_optimizer import DiffGrad
import torchvision.models as models
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import cv2
import os
from tqdm import tqdm

import click

epsillon = 1e-8
torch.autograd.set_detect_anomaly(True)
class Trainer():

  def __init__(self, backbone, silco, higher_resolution, batch_size=16, learning_rate=0.001):
    self.config = Config()
    self.logger = self.config.get_logger(self)
    self.backbone = backbone
    self.silco = silco
    self.higher_resolution = higher_resolution
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.roboaugen_writer = SummaryWriter('logdir/roboaugen')

  @staticmethod
  def entropy(classifier):
    return - (classifier * torch.log(classifier + epsillon)).mean()

  @staticmethod
  def compute_classifiers_entropy(classifiers):
    spatial_classifiers_entropy = 0.0
    for current_classifiers in classifiers:
      classifiers_entropy = 0.0
      for classifier in current_classifiers:
        classifiers_entropy += Trainer.entropy(classifier)
      classifiers_entropy /= len(current_classifiers)
      spatial_classifiers_entropy += classifiers_entropy
    classifiers_entropy /= len(classifiers)
    return classifiers_entropy

  def mse(self, target_heatmaps, predicted_heatmaps, spatial_penalty):
    positives = target_heatmaps > 0.
    negatives = target_heatmaps == 0.
    error = ((target_heatmaps - predicted_heatmaps) ** 2)
    positive_error = error * positives
    positive_area =  positives.permute(2, 3, 0, 1).view(-1, positives.size()[0], positives.size()[1]).sum(0) + 1
    negative_error = error * negatives
    negative_area =  negatives.permute(2, 3, 0, 1).view(-1, negatives.size()[0], negatives.size()[1]).sum(0) + 1

    sum_positive_error = positive_error.permute(2, 3, 0, 1).view(-1, positive_error.size()[0], positive_error.size()[1]).sum(0)
    sum_negative_error = negative_error.permute(2, 3, 0, 1).view(-1, negative_error.size()[0], negative_error.size()[1]).sum(0)
    mean_positive_error = sum_positive_error / positive_area
    mean_negative_error = sum_negative_error / negative_area
    #print(sum_positive_error, sum_negative_error)
    #print(positive_area, negative_area)
    #print(mean_positive_error.size(), mean_negative_error.size())

    error_normalized = (mean_positive_error + mean_negative_error) / 2.
    return error_normalized.mean()
    '''
    print('START')
    print(target_heatmaps.size())
    print(target_heatmaps.permute(2, 3, 0, 1).view(-1,target_heatmaps.size()[0], target_heatmaps.size()[1]).sum(0).size())
    non_zero = torch.nonzero(target_heatmaps > 0.).cuda()
    print(non_zero[0])
    print(non_zero.size())
    print(target_heatmaps[non_zero[0][0], non_zero[0][1], non_zero[0][2], non_zero[0][3]])
    print('END')
    heatmap_target = ((target_heatmaps - predicted_heatmaps) ** 2)# * (1.0 - spatial_penalty)
    return ( (heatmap_target.data + 1.0) * heatmap_target).mean()
    '''

  def save_train_state(self, epoch, current_iteration, optimizer_higher_resolution, optimizer_silco):
    torch.save({
                'epoch': epoch,
                'current_iteration': current_iteration,
                'optimizer_higher_resolution_state_dict': optimizer_higher_resolution.state_dict(),
                'optimizer_silco_state_dict': optimizer_silco.state_dict()
                }, self.config.train_state_path)

  def load_train_state(self):
    #optimizer = DiffGrad(self.backbone.parameters(), lr = self.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer_higher_resolution = DiffGrad(self.higher_resolution.parameters(), lr = self.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer_silco = DiffGrad(self.silco.parameters(), lr = self.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    current_iteration = 0
    epoch = -1
    if os.path.exists(self.config.train_state_path):
      checkpoint = torch.load(self.config.train_state_path)
      #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      optimizer_higher_resolution.load_state_dict(checkpoint['optimizer_higher_resolution_state_dict'])
      optimizer_silco.load_state_dict(checkpoint['optimizer_silco_state_dict'])
      epoch = checkpoint['epoch']
      current_iteration = checkpoint['current_iteration'] if 'current_iteration' in checkpoint else 0
    #return optimizer, optimizer_higher_resolution, optimizer_silco, epoch
    return optimizer_higher_resolution, optimizer_silco, epoch, current_iteration

  def log(self, current_iteration, losses):
    #self.logger.info(f'Epoch {epoch + 1}/{epochs} | Batch {batch_index + 1}/{batches}')
    self.roboaugen_writer.add_scalar('Training Loss', losses['total_loss'], current_iteration )
    self.roboaugen_writer.add_scalar('MSE Loss', losses['mse_loss'], current_iteration )
    self.roboaugen_writer.add_scalar('Entropy Loss', losses['entropy_loss'], current_iteration )
    #self.roboaugen_writer.add_scalar('Spatial Entropy Loss', losses['spatial_location_entropy_loss'], current_iteration )
    #self.roboaugen_writer.add_scalar('Feature Entropy Loss', losses['feature_entropy_loss'], current_iteration )

  def save_models(self):
    self.config.save_mobilenet_model(self.backbone)
    self.config.save_silco_model(self.silco)
    self.config.save_higher_resolution_model(self.higher_resolution)

  def compute_spatial_loss(spatial_classifier, object_mask):
    spatial_loss = 0
    for scale_idx, spatial_classifier_scale in enumerate(spatial_classifier):
      for support_idx, spatial_classifier_support in enumerate(spatial_classifier_scale):
        spatial_classifier_support_sum = spatial_classifier_support.sum(2)
        object_mask_scale = object_mask.view(object_mask.size()[0], -1)
        object_mask_scale = (object_mask_scale > 0).float()
        spatial_classifier_support_sum = spatial_classifier_support_sum * (1. - object_mask_scale)
        spatial_loss = spatial_loss + spatial_classifier_support_sum.sum()
    return spatial_loss

  def train(self, epochs, mode, max_background_objects, max_foreground_objects, distort):
    self.backbone.train()
    self.backbone.cuda()
    if mode == 'silco' or mode == 'both':
      self.silco.train()
      self.silco.cuda()
    self.higher_resolution.train()
    self.higher_resolution.cuda()

    optimizer_higher_resolution, optimizer_silco, epoch_start, current_iteration = self.load_train_state()
    train_loader = None
    batches = None
    epoch_start = 0
    print(f'Starting with epoch {epoch_start + 1} with a total of {epochs} epochs')
    for epoch in range(epoch_start + 1, epochs):
      if train_loader is None or (epoch + 1) % self.config.save_each_epoch == 0:
        train_dataset = ProjectedMeshDataset(self.config.input_height,
          self.config.input_width, self.config.num_vertices,
          max_background_objects, max_foreground_objects,
          distort=distort, keep_dimensions=True, use_cache=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=8)
        batches = len(train_loader)
      for batch_index, (queries, supports, target_heatmaps, spatial_penalty, _, _) in tqdm(enumerate(train_loader)):
        losses = {}
        queries = queries.cuda(non_blocking=True)
        supports = supports.cuda(non_blocking=True)
        query_features, support_features = Silco.backbone_features(self.backbone, queries, supports)
        if mode == 'silco' or mode == 'both':
          query_features, spatial_classifiers, feature_classifiers, _ = self.silco(query_features, support_features)

        #losses['feature_entropy_loss'] = 0.#Trainer.compute_classifiers_entropy(feature_classifiers) * 1e-12
        #losses['spatial_location_entropy_loss'] = Trainer.compute_classifiers_entropy(spatial_classifiers) * 1e-12
        predicted_heatmaps = self.higher_resolution(query_features) #updated_query_features)
        target_heatmaps = F.interpolate(target_heatmaps, size=(self.config.input_width // 2, self.config.input_height // 2), mode='bilinear').cuda(non_blocking=True)
        spatial_penalty = F.interpolate(spatial_penalty, size=(self.config.input_width // 2, self.config.input_height // 2), mode='bilinear').cuda(non_blocking=True)
        losses['mse_loss'] = self.mse(target_heatmaps, predicted_heatmaps, spatial_penalty)
        losses['entropy_loss'] = Trainer.entropy(predicted_heatmaps) * 1e-3
        losses['total_loss'] = losses['mse_loss'] + losses['entropy_loss']#+ losses['spatial_location_entropy_loss'] + losses['feature_entropy_loss']
        if mode == 'silco' or mode == 'both':
          optimizer_silco.zero_grad()
        if mode == 'keypoints' or mode == 'both':
          optimizer_higher_resolution.zero_grad()
        losses['total_loss'].backward()
        if mode == 'keypoints' or mode == 'both':
          optimizer_higher_resolution.step()
        if mode == 'silco' or mode == 'both':
          optimizer_silco.step()
        self.log(current_iteration, losses)
        current_iteration += 1
        if (batch_index + 1) % self.config.save_each == 0:
          self.save_models()
          self.save_train_state(epoch_start, current_iteration, optimizer_higher_resolution, optimizer_silco)
      if (epoch + 1) % self.config.save_each_epoch == 0:
        self.save_models()

@click.command()
@click.option("--learning_rate", default=0.001, help="Learning rate")
@click.option("--batch_size", default=8, help="Batch size.")
@click.option("--epochs", default=1000, help="Epochs.")
@click.option("--mode", default='keypoints', help="Training mode: keypoints, silco, both.")
@click.option("--max_background_objects", default=0, help="Maximum number of background objects not in target for keypoints")
@click.option("--max_foreground_objects", default=0, help="Maximum number of foreground objects not in target for keypoints.")
@click.option("--distort", default=False, help="Wether to apply distortions.")

def train(learning_rate, batch_size, epochs, mode, max_background_objects, max_foreground_objects, distort):
  config = Config()
  higher_resolution = config.load_higher_resolution_model()
  if higher_resolution is None:
    print("Higher Resolution Network not found. Creating one from scratch.")
    higher_resolution = HigherResolutionNetwork(config.channels_blocks, config.dimensions_blocks, config.num_vertices)
  silco = config.load_silco_model()
  if silco is None:
    print("SILCO Network not found. Creating one from scratch.")
    silco = Silco()
  backbone = config.load_mobilenet_model()
  if backbone is None:
    print("MobileNet Network not found. Creating one from scratch.")
    backbone = models.mobilenet_v2(pretrained=True)
    backbone = MobileNetWrapper(backbone.features)
  trainer = Trainer(backbone, silco, higher_resolution, batch_size, learning_rate)
  trainer.train(epochs, mode, max_background_objects, max_foreground_objects, distort)


if __name__ == '__main__':
  train()


