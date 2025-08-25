import os
import random
import deepinv as dinv
import torch.nn.functional as F

from torch.utils.data import Dataset
from typing import Callable
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self,
                 folder: str,
                 physics: dinv.physics.Physics,
                 device: str,
                 transform: Callable = None,
                 num_images=None,):
        self.folder = folder
        self.physics = physics
        self.device = device
        self.transform = transform
        self.files = [f for f in os.listdir(folder) if f.endswith((
                      '.png', '.jpg', '.jpeg'))]

        if num_images is not None:
            self.files.sort()
            self.files = random.sample(self.files, num_images)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder, self.files[idx])
        x = Image.open(img_name)

        if self.transform:
            x = self.transform(x)
        
        x = x.to(self.device)

        y = self.physics(x.unsqueeze(0))
        y = y.squeeze(0)

        #_, x_h, x_w = x.shape
        #_, y_h, y_w = y.shape
        
        #diff_h = x_h - y_h
        #diff_w = x_w - y_w
        
        #pad_top = diff_h // 2
        #pad_bottom = diff_h - pad_top
        #pad_left = diff_w // 2
        #pad_right = diff_w - pad_left
        
        #y = F.pad(y, pad=(pad_left, pad_right, pad_top, pad_bottom), value=0)

        return x, y
