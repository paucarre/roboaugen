from roboaugen.core.config import Config

import copy
import logging
import random
import math
import traceback
import os
import shutil

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF


class ObjectData():

    def __init__(self, object_type, sample_id):
        self.object_type = object_type
        self.sample_id = sample_id


class ProjectedMeshDataset(Dataset):

    def __init__(self, height, width, num_vertices, max_background_objects, max_foreground_objects, distort=True, keep_dimensions=False, use_cache=True):
        self.height = height
        self.width = width
        self.max_background_objects = max_background_objects
        self.max_foreground_objects = max_foreground_objects
        self.num_vertices = num_vertices
        self.config = Config()
        #self.sample_ids = self.config.get_sample_ids()
        self.num_vertices = num_vertices
        self.target_generator = TargetGenerator(height, width, num_vertices)
        self.distort = distort
        self.logger = self.config.get_logger(self)
        self.use_cache = use_cache
        self.object_type_to_ids = self.config.get_object_type_to_ids()
        self.keep_dimensions = False

    def __len__(self,):
        return sum([len(ids) for ids in self.object_type_to_ids.values()])

    def distort_image(self, input_image):
        input_image = TF.adjust_brightness(input_image, 1.0 + (torch.rand(1).item() - 0.5)) # from 0.5 to 1.5
        input_image = TF.adjust_contrast(input_image, 1.0 + (torch.rand(1).item() - 0.5)) # from 0.5 to 1.5
        input_image = TF.adjust_saturation(input_image, 1.0 + (torch.rand(1).item() - 0.5)) # from 0.5 to 1.5
        if torch.rand(1).item() < 0.1:
            kernel_size = int(torch.rand(1).item() * 30) + 3
            if kernel_size % 2 == 0:
                kernel_size -= 1
            input_image = torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))(input_image)
        return input_image

    def image_to_torch(self, input_image):
        input_image = torch.from_numpy(input_image / 255)
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]
        input_image = input_image.transpose(0, 2).transpose(1, 2).float()
        if self.keep_dimensions == True:
            if input_image_width != input_image_height or \
                input_image_height != self.config.input_height or input_image_width != self.config.input_width:
                input_image = input_image.unsqueeze(1)
                input_image = F.interpolate(input_image, size=(self.config.input_height, self.config.input_width), mode='bilinear')
                input_image = input_image.squeeze(1)
        elif max(input_image_height, input_image_width) > max(self.config.input_height, self.config.input_width):
            aspect_ratio = input_image_height / input_image_width
            if input_image_height > input_image_width:
                input_image_height = max(self.config.input_height, self.config.input_width)
                input_image_width = int(input_image_height / aspect_ratio)
            else:
                input_image_width = max(self.config.input_height, self.config.input_width)
                input_image_height = int(input_image_width * aspect_ratio)
            input_image = input_image.unsqueeze(1)
            input_image = F.interpolate(input_image, size=(input_image_height, input_image_width), mode='bilinear')
            input_image = input_image.squeeze(1)
        return input_image

    def save_in_cache(self, object_type, index, input_image, supports, target, spatial_penalty, coordinates, coordinates_probs):
        folder = f'{self.config.cache_train_dir}/{object_type}/{index}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        try:
            torch.save(input_image, f'{folder}/input_image')
            torch.save(supports, f'{folder}/supports')
            torch.save(target, f'{folder}/target')
            torch.save(spatial_penalty, f'{folder}/spatial_penalty')
            torch.save(coordinates, f'{folder}/coordinates')
            torch.save(coordinates_probs, f'{folder}/coordinates_probs')
            #torch.save(object_alphas, f'{folder}/object_alphas')
        except:
            self.logger.error(f'Error saving sample at index {index}')
            self.logger.error(traceback.format_exc())
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                except:
                    self.logger.error(traceback.format_exc())

    def load_from_cache(self, object_type, index):
        try:
            cache_elements = os.listdir(self.config.cache_train_dir)
            cache_elements = [element for element in cache_elements if element.isdigit()]
            if len(cache_elements) == 0:
                return None
            index = index % len(cache_elements)
            index = cache_elements[index]
            folder = f'{self.config.cache_train_dir}/{object_type}/{index}'
            input_image = torch.load(f'{folder}/input_image')
            supports = torch.load(f'{folder}/supports')
            target = torch.load(f'{folder}/target')
            spatial_penalty = torch.load(f'{folder}/spatial_penalty')
            coordinates = torch.load(f'{folder}/coordinates')
            coordinates_probs = torch.load(f'{folder}/coordinates_probs')
            object_alphas = torch.load(f'{folder}/object_alphas')
            return input_image, supports, target, spatial_penalty, coordinates, coordinates_probs, object_alphas
        except:
            self.logger.error(f'Error loading sample at index {index}')
            self.logger.error(traceback.format_exc())
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                except:
                    self.logger.error(traceback.format_exc())
            return None

    def get_objects(self, index, object_types):
        sampled_objects = []
        for sample_index, object_type in enumerate(object_types):
            ids_in_object_type = len(self.object_type_to_ids[object_type])
            sample_id = index + sample_index # TODO: add some entropy/randomness here
            sample_id = self.object_type_to_ids[object_type][sample_id % ids_in_object_type]
            sampled_objects.append(ObjectData(object_type, sample_id))
        return sampled_objects

    def get_images_and_alpha_from_objects(self, objects):
        if objects is not None and len(objects) > 0:
            objects = [self.config.get_image_sample_alpha(sample.object_type, sample.sample_id) for sample in objects]
            objects = [current_object for current_object in objects if current_object is not None]
            object_alphas = [ self.image_to_torch(sample[:, :, 3:4]).float() for sample in objects]
            object_alphas = torch.cat([alpha for alpha in object_alphas], 0)
            object_images = [ self.image_to_torch(sample[:, :, 0:3]) for sample in objects ]
            return object_alphas, object_images
        return torch.zeros((self.config.input_width, self.config.input_width)), None

    def get_random_objects(self, index, from_count, to_count):
        count = random.randint(from_count, to_count)
        object_types = random.sample(list(self.object_type_to_ids.keys()), count)
        objects = self.get_objects(index, object_types)
        object_alphas, object_images = self.get_images_and_alpha_from_objects(objects)
        return objects, object_alphas, object_images

    def handle_data_loading_error(self, index, object_type, support_ids, object_to_detect):
        self.logger.error(f'Error loading sample at index {index}')
        self.logger.error(traceback.format_exc())
        support_folders_cache = []
        support_folders_data = []
        if support_ids is not None:
            support_folders_cache = [ f'{self.config.cache_train_dir}/{object_type}/{support_id}' for support_id in support_ids]
            support_folders_data = [ f'{self.config.train_dataset_dir}/{object_type}/{support_id}' for support_id in support_ids]
        if object_to_detect is not None:
            folder_data = f'{self.config.train_dataset_dir}/{object_type}/{object_to_detect.sample_id}'
            folder_cache = f'{self.config.cache_train_dir}/{object_type}/{object_to_detect.sample_id}'
            for folder in support_folders_cache + support_folders_data + [folder_data, folder_cache]:
                if os.path.exists(folder):
                    try:
                        #shutil.rmtree(folder)
                        pass
                    except:
                        self.logger.error(traceback.format_exc())

    def get_supports(self, index, object_type):
        supports = []
        for support_id in range(self.config.supports):
            object_types =  [object_type]
            object_index = random.choice(range(len(self.object_type_to_ids[object_type])))
            objects_background = self.get_objects(object_index, object_types)
            object_alphas_background, object_images_background = self.get_images_and_alpha_from_objects(objects_background)
            support = self.image_to_torch(self.config.get_background_sample(index))
            if object_images_background is not None:
                for idx, object_image_background in enumerate(object_images_background):
                    support = support * (1.0 - object_alphas_background[idx])
                    support += (object_alphas_background[idx] * object_image_background)
                if self.distort:
                    support = self.distort_image(support)
                supports.append(support.unsqueeze(0))
        return torch.cat(supports, 0)

    def generate_target(self, data, std):
        return self.target_generator.generate_target(data['projected_visible_vertices'], data['resolution'][1], data['resolution'][0], std)

    def get_input_image(self, background_image, object_to_detect, object_images_foreground, object_alphas_foreground):
        input_image = self.config.get_image_sample_alpha(object_to_detect.object_type, object_to_detect.sample_id)
        object_alphas = self.image_to_torch(input_image[:, :, 3:4]).float()
        object_image = self.image_to_torch(input_image[:, :, 0:3])
        background_image = background_image * (1.0 - object_alphas)
        input_image = background_image + (object_image * object_alphas)
        if self.distort:
            input_image = self.distort_image(input_image)
        if object_images_foreground is not None:
            for idx, object_image_foreground in enumerate(object_images_foreground):
                input_image = input_image * (1.0 - object_alphas_foreground[idx])
                object_alphas = object_alphas * (1.0 - object_alphas_foreground[idx])
                input_image += (object_alphas_foreground[idx] * object_image_foreground)
        return input_image, object_alphas

    def get_only_background(self, index, input_image, object_images_foreground, object_alphas_foreground):
        if self.distort:
            input_image = self.distort_image(input_image)
        if object_images_foreground is not None:
            for idx, object_image_foreground in enumerate(object_images_foreground):
                input_image = input_image * (1.0 - object_alphas_foreground[idx])
                input_image += (object_alphas_foreground[idx] * object_image_foreground)
        target = torch.zeros(self.config.num_vertices, self.config.input_width, self.config.input_height)
        spatial_penalty = torch.zeros(self.config.num_vertices, self.config.input_width, self.config.input_height)
        coordinates = torch.tensor([[0., 0.]] * self.config.num_vertices)
        coordinates_probs = torch.tensor([0.] * self.config.num_vertices)
        object_alphas = torch.zeros(self.config.input_width, self.config.input_height)
        return input_image, target, spatial_penalty, coordinates, coordinates_probs

    def __getitem__(self, index):
        sample = None
        support_ids = None
        object_to_detect = None
        object_type = None
        #if self.use_cache and torch.rand(1).item() < 0.90:
        #    object_type = random.choice(\
        #            list([object_type for object_type in self.object_type_to_ids.keys()]))
        #    sample = self.load_from_cache(object_type, index)
        if sample is None:
            try:
                # Add background image and objects
                objects_background, object_alphas_background, object_images_background = self.get_random_objects(index, 0, self.max_background_objects)
                background_image = self.image_to_torch(self.config.get_background_sample(index))
                if object_images_background is not None:
                    for idx, object_image_background in enumerate(object_images_background):
                        background_image = background_image * (1.0 - object_alphas_background[idx])
                        background_image += (object_alphas_background[idx] * object_image_background)

                # Select foreground objects
                objects_foreground, object_alphas_foreground, object_images_foreground = self.get_random_objects(index, 0, self.max_foreground_objects)
                foreground_and_background_object_types = set(map(lambda x: x.object_type, objects_background)) | set(map(lambda x: x.object_type, objects_foreground))
                object_type = random.choice(\
                    list([object_type for object_type in self.object_type_to_ids.keys() \
                        if object_type not in foreground_and_background_object_types]))


                # There will always be supports => TODO: actually during only keypoint training/inference there are no supports!!
                supports = self.get_supports(index, object_type)
                #print(supports.size())

                # Add image, if applicable
                object_to_detect = self.get_objects(index, [object_type])[0]
                object_in_image = torch.rand(1).item() > 0.1
                if object_in_image:
                    input_image, object_alphas = self.get_input_image(background_image, object_to_detect, object_images_foreground, object_alphas_foreground)
                    data = self.config.get_data_sample(object_type, object_to_detect.sample_id)
                    target = self.generate_target(data, 1)
                    spatial_penalty = self.generate_target(data, 10).clamp(0.0, 0.9)
                    target = target * (1.0 - object_alphas_foreground)
                    spatial_penalty = spatial_penalty * (1.0 - object_alphas_foreground)
                    coordinates, coordinates_probs = self.get_coordinates_and_probs(data)
                else:
                    input_image, target, spatial_penalty, coordinates, coordinates_probs = self.get_only_background(index, background_image, object_images_foreground, object_alphas_foreground)
                self.save_in_cache(object_type, index, input_image, supports, target, spatial_penalty, coordinates, coordinates_probs)
                sample = input_image, supports, target, spatial_penalty, coordinates, coordinates_probs
            except:
                self.handle_data_loading_error(index, object_type, support_ids, object_to_detect)
                return self.__getitem__( (index + 1000) % len(self)) # TODO: add entropy here
        return sample

    def get_coordinates_and_probs(self, data):
        coordinates = []
        coordinates_probs = []
        projected_visible_vertices = data['bbox_vertices']
        for coord in projected_visible_vertices:
            if coord is not None:
                coordinates.append(coord)
                coordinates_probs.append(1.)
            else:
                coordinates.append([0., 0.])
                coordinates_probs.append(0.)
        return torch.Tensor(coordinates), torch.Tensor(coordinates_probs)


class HeatmapDataset(ProjectedMeshDataset):

    def __init__(self, batch_size):
        self.config = Config()
        super().__init__(self.config.input_height,
            self.config.input_width, self.config.num_vertices)
        self.batch_size = batch_size
        self.model_heatmap = self.config.load_model()

    def __getitem__(self, index):
        input_images_batch, target_coordinates_batch, target_coordinates_probs_batch = [], [] ,[]
        for _ in range(self.batch_size):
            input_images, _, target_coordinates, target_coordinates_probs = super().__getitem__(index)
            input_images_batch.append(input_images.unsqueeze(0))
            target_coordinates_batch.append(target_coordinates.unsqueeze(0))
            target_coordinates_probs_batch.append(target_coordinates_probs.unsqueeze(0))
        input_images_batch = torch.cat(input_images_batch, dim=0)
        target_coordinates_batch = torch.cat(target_coordinates_batch, dim=0)
        target_coordinates_probs_batch = torch.cat(target_coordinates_probs_batch, dim=0)
        predicted_heatmap, _ = self.model_heatmap(input_images_batch)
        predicted_coordinates = self.model_heatmap.get_coordinates(predicted_heatmap, self.config.points_per_vertex)
        return predicted_coordinates, target_coordinates_batch, target_coordinates_probs_batch

    def __len__(self,):
        return super().__len__() // self.batch_size


class TargetGenerator():

    def __init__(self, target_height, target_width, num_vertices):
        self.target_height = target_height
        self.target_width = target_width
        self.num_vertices = num_vertices
        #self.sigma = 1 # this if fixed, we can change if afterwards iff needed

    def generate_gaussian_kernel(self, sigma):
        kernel_size = (6 * sigma) + 3
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        coord_center = (3 * sigma) + 1
        gaussian_kernel = (1. / (2. * math.pi * sigma ) ) * \
                  torch.exp(
                      -torch.sum( (xy_grid - coord_center) ** 2., dim=-1) / \
                      (2 * float(sigma))
                  )
        gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
        return gaussian_kernel

    def generate_target(self, projected_visible_vertices, vertex_height, vertex_width, sigma):
        gaussian_kernel = self.generate_gaussian_kernel(sigma)
        heatmap = torch.zeros([self.num_vertices, self.target_height, self.target_width])
        for vertex_index, vertex in enumerate(projected_visible_vertices):
            if vertex is not None:
                x, y = vertex
                x = (x / vertex_width ) * self.target_width
                y = (y / vertex_height) * self.target_height
                if x < self.target_width and y < self.target_height and x >=0 and y >= 0:
                    upper_left   = int( np.round(x - (3 * sigma) - 1) ), int( np.round(y - (3 * sigma) - 1) )
                    bottom_right = int( np.round(x + (3 * sigma) + 2) ), int( np.round(y + (3 * sigma) + 2) )

                    x_init, x_end = max(0, upper_left[0]), min(bottom_right[0], self.target_width)
                    if (x_end - x_init) > gaussian_kernel.size()[1]:
                        x_end -= 1
                    y_init, y_end = max(0, upper_left[1]), min(bottom_right[1], self.target_height)
                    if (y_end - y_init) > gaussian_kernel.size()[0]:
                        y_end -= 1

                    x_init_gaussian = -upper_left[0] if upper_left[0] < 0 else 0
                    x_end_gaussian  = x_init_gaussian + (x_end - x_init)
                    y_init_gaussian = -upper_left[1] if upper_left[1] < 0 else 0
                    y_end_gaussian  = y_init_gaussian + (y_end - y_init)
                    '''
                    print("--------------------------")
                    print(self.sigma, upper_left[1], bottom_right[1], self.height)
                    print('ob', x_end - x_init, y_end - y_init)
                    print('ga', x_end_gaussian - x_init_gaussian, y_end_gaussian - y_init_gaussian)
                    print(self.gaussian_kernel.size())
                    '''
                    heatmap[vertex_index, y_init:y_end, x_init:x_end] = \
                        gaussian_kernel[y_init_gaussian:y_end_gaussian, x_init_gaussian:x_end_gaussian]
        return heatmap

if __name__ == '__main__':
    config = Config()
    sample_id = argv[1]
    image = config.get_image_sample(sample_id)
    height, width, channels = image.shape
    print(height, width, channels)
    dataset = ProjectedMeshDataset(height, width, config.num_vertices)
    input_image, target = dataset.__getitem__(sample_id)
    data = config.get_data_sample(sample_id)
    transformed_points = data['bbox_vertices']
    cv2.imshow('target', target.sum(0).numpy())
    cv2.imshow('input_image', input_image.squeeze().transpose(0,2).transpose(0,1).numpy())
    cv2.waitKey(0)