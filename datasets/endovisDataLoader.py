'''
Class for loading the miccaiSeg dataset
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os
import json

class endovisDataset(Dataset):
    '''
        Endovis Dataset
    '''

    def __init__(self, root_dir, transform=None, json_path=None, training=False):
        '''
        Args:
            root_dir (string): Directory with all the images
            transform(callable, optional): Optional transform to be applied on a sample
            json_path (string, optional): Optional json path for dataset classes
            training (bool): True when training, False for testing. If testing, skips loading
                         the groundtruth
        '''

        self.training = training
        self.root_dir = root_dir
        self.sub_dirs = [os.path.join(self.root_dir, sd) for sd in os.listdir(self.root_dir) \
                        if os.path.isdir(os.path.join(self.root_dir, sd))]
        self.img_dirs = [os.path.join(sd, 'left_images') for sd in self.sub_dirs]
        if self.training:
            self.gt_dirs = [os.path.join(sd, 'labels') for sd in self.sub_dirs]
        self.image_list = []
        for img_dir in self.img_dirs:
            self.image_list.append([f for f in os.listdir(img_dir) if (f.endswith('.png') or f.endswith('.jpg'))])
        self.transform = transform

        if json_path:
            # Read the json file containing classes information
            # This is later used to generate masks from the segmented images
            self.classes = json.load(open(json_path))['classes']

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        if not self.training:
            gt_file_name = img_name.split('/')[-1]
            gt_name = os.path.join(self.gt_dir, gt_file_name)
            gt = Image.open(gt_name)
            gt = gt.convert('RGB')

        if self.transform:
            image = self.transform(image)
            if self.training:
                gt = self.transform(gt)

        if self.training:
            return image, gt
        else:
            return image
