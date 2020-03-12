import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

torch.backends.cudnn.deterministic = True


from torch_optimizer import DiffGrad
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchvision import transforms, utils
import torchvision.models as models
import torch.autograd as autograd

from apex import amp

epsillon = 1e-6

class NetUtils():

    @staticmethod
    def xavier_sequential(layer):
      if type(layer) == nn.Sequential:
        for module in layer:
          NetUtils.xavier_sequential(module)
      elif type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d or type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
          torch.nn.init.zeros_(layer.bias.data)

    @staticmethod
    def zero_sequential(layer):
      if type(layer) == nn.Sequential:
        for module in layer:
          NetUtils.zero_sequential(module)
      elif type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d or type(layer) == nn.Linear:
        torch.nn.init.zeros_(layer.weight.data)
        if layer.bias is not None:
          torch.nn.init.zeros_(layer.bias.data)

    @staticmethod
    def zeros_sigmoid_sequential(sequential):
      for module in sequential:
        if type(module) == nn.Sequential:
          NetUtils.zeros_sigmoid_sequential(module)
        elif type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
          nn.init.zeros_(module.weight.data)
          if module.bias is not None:
            module.bias.data[:] = -8

    @staticmethod
    def conv_3x3_with_batch_norm(in_channels, out_channels, stride=2):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        NetUtils.xavier_sequential(conv)
        return conv



class HigherResolutionNetwork(nn.Module):

    def __init__(self,
                 channels_blocks,
                 dimensions_blocks,
                 num_vertices):
        super(HigherResolutionNetwork, self).__init__()
        #self.backbone = MobileNetV2(3)
        self.channels_blocks = channels_blocks
        self.num_vertices = num_vertices
        self._init_layers()

    def _init_layers(self):
      self.block_to_first_phase = self._init_sequential_phase()
      self._init_signal_cross_sharing_phase()
      self.block_to_second_phase = self._init_sequential_phase()
      self._init_signal_aggregation()
      self._init_high_resolution_upsample()

    def _init_signal_cross_sharing_phase(self):
      self.source_to_destination_convolution = torch.nn.ModuleList()
      for source_block_index, source_channels in enumerate(self.channels_blocks):
        self.source_to_destination_convolution.append(torch.nn.ModuleList())
        for destination_block_index, destination_channels in enumerate(self.channels_blocks):
          self.source_to_destination_convolution[source_block_index].append( \
            NetUtils.conv_3x3_with_batch_norm(source_channels, destination_channels, stride=1) )

    def _init_sequential_phase(self):
      block_to_phase = torch.nn.ModuleList()
      for channels in self.channels_blocks:
        conv = nn.Sequential(\
          NetUtils.conv_3x3_with_batch_norm(channels, channels, stride=1),
          NetUtils.conv_3x3_with_batch_norm(channels, channels, stride=1))
        block_to_phase.append(conv)
      return block_to_phase

    def _init_signal_aggregation(self):
      self.source_convolution_aggregation = torch.nn.ModuleList()
      for source_channels in self.channels_blocks:
        self.source_convolution_aggregation.append( \
          NetUtils.conv_3x3_with_batch_norm(source_channels, self.channels_blocks[0], stride=1) )

    def _init_high_resolution_upsample(self):
      padding = 1
      kernel_size = 2 + ( 2 * padding )
      self.high_resolution_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.channels_blocks[0], self.num_vertices,
              kernel_size=kernel_size, stride=2, padding=padding),
            nn.Sigmoid()
      )
      NetUtils.zeros_sigmoid_sequential(self.high_resolution_upsample)

    def forward_signal_cross_sharing(self, input_block):
      outputs = []
      #[torch.Size([18, 24, 128, 128]), torch.Size([18, 32, 64, 64]), torch.Size([18, 64, 32, 32]), torch.Size([18, 160, 16, 16])]
      for destination_block_index, _ in enumerate(self.channels_blocks):
        #destination_dimensions = self.dimensions_blocks[destination_block_index]
        current_aggregated_output = input_block[destination_block_index]
        destination_dimensions = current_aggregated_output.size()[2]
        #print('destination size: ', current_aggregated_output.size(), destination_dimensions)
        for source_block_index, _ in enumerate(self.channels_blocks):
          input_interpolated = F.interpolate(input_block[source_block_index], size=destination_dimensions, mode='bilinear')
          #print('\t currnet: ', input_interpolated.size())
          convolution = self.source_to_destination_convolution[source_block_index][destination_block_index]
          current_aggregated_output = current_aggregated_output + convolution(input_interpolated)
        outputs.append(current_aggregated_output)
      return outputs

    def forward_signal_aggregation(self, input_block):
      output = input_block[0]
      destination_dimensions = input_block[0].size()[2]
      for source_block_index, _ in enumerate(self.channels_blocks):
          input_interpolated = F.interpolate(input_block[source_block_index], size=destination_dimensions, mode='bilinear')
          convolution = self.source_convolution_aggregation[source_block_index]
          output = output + convolution(input_interpolated)
      return output

    def forward_sequential(self, inputs, convolutions):
      return [convolutions[index](inputs[index]) for index in range(len(convolutions))]

    def forward(self, backbone_features):
        first_phase = self.forward_sequential(backbone_features, self.block_to_first_phase)
        cross_sharing = self.forward_signal_cross_sharing(first_phase)

        second_phase = self.forward_sequential(cross_sharing, self.block_to_second_phase)
        aggregated_signal = self.forward_signal_aggregation(second_phase)
        heatmap_prediction = self.high_resolution_upsample(aggregated_signal)
        return heatmap_prediction

    def get_coordinates(self, heatmaps, top_k=100):
        heatmaps = heatmaps.data
        coords = []
        for sample in range(heatmaps.size()[0]):
          sample_coords = []
          for vertex in range(self.num_vertices):
            if heatmaps[sample, vertex, :, :].max() > 0.7:
              heatmaps[sample, vertex, :, :] = (heatmaps[sample, vertex, :, :] / heatmaps[sample, vertex, :, :].max())
            current_heatmap = heatmaps[sample, vertex, :, :].view(-1)
            values, indices = torch.topk(current_heatmap, k=top_k, dim=0)
            vertex_coords = []
            for index_flatten in range(indices.size()[0]):
              if values[index_flatten].item() > 0.5:
                output_height = heatmaps[sample, vertex, :, :].size()[0]
                x = (indices[index_flatten] % output_height).item()
                y = (indices[index_flatten] // output_height).item()
                prob = values[index_flatten].item()
                vertex_coords.append([x, y, prob])
              else:
                vertex_coords.append([0.0, 0.0, 0.0])
              #print( (x, y), values[index_flatten].item(), heatmaps[0, vertex, y, x].item() )
            sample_coords.append(vertex_coords)
          coords.append(sample_coords)
        return torch.FloatTensor(coords)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        NetUtils.xavier_sequential(self.conv)


    def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)


'''
    MobileNet V2 head

    Paper:
        MobileNetV2: Inverted Residuals and Linear Bottlenecks
        https://arxiv.org/pdf/1801.04381.pdf


    The code is borrowed from https://github.com/tonylins/pytorch-mobilenet-v2

'''
class MobileNetV2(nn.Module):

    def __init__(self,
                input_channel=32,
                interverted_residual_setting=[
                                                # t, c, n, s
                                                [1, 16, 1, 1],
                                                [6, 24, 2, 2],
                                                [6, 32, 3, 2],
                                                [6, 64, 4, 2],
                                                [6, 96, 3, 1],
                                                [6, 160, 3, 2],
                                                [6, 320, 1, 1],
                                            ],
                width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.input_channels = int(input_channel * width_mult)
        self.interverted_residual_setting = interverted_residual_setting
        self.width_mult = width_mult
        self._init_layers()

    def forward(self, x):
        outputs = []
        for feature in self.features:
            x = feature(x)
            outputs.append(x)
        outputs = [outputs[end_of_block] for end_of_block in self.end_of_block_indices]
        return outputs

    def _init_layers(self):
        current_input_channels = self.input_channels
        self.features = [NetUtils.conv_3x3_with_batch_norm(3, current_input_channels)]
        # building inverted residual blocks
        self.end_of_block_indices = []
        for expand_ratio, channels, repetitions, stride in self.interverted_residual_setting:
            output_channel = int(channels * self.width_mult)
            for i in range(repetitions):
                if i == 0:
                    self.features.append(InvertedResidual(current_input_channels, output_channel, stride, expand_ratio=expand_ratio))
                else:
                    self.features.append(InvertedResidual(current_input_channels, output_channel, 1, expand_ratio=expand_ratio))
                    if i == repetitions - 1 and stride > 1:
                        # end of block detected
                        self.end_of_block_indices.append(len(self.features) - 1)
                current_input_channels = output_channel
        self.features = nn.Sequential(*self.features)

