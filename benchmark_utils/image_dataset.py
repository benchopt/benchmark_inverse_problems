import os
import random

from torch.utils.data import Dataset
from typing import Callable
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, folder: str, transform: Callable = None, num_images=None):
        self.folder = folder
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
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
