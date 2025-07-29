import torch
from torch.utils.data import Dataset
from deepinv.datasets import FastMRISliceDataset
import torch.nn.functional as F
import deepinv as dinv

class FastMRIDataset(Dataset):
    def __init__(self, dataset: FastMRISliceDataset, mask, max_coils=32):
        self.dataset = dataset
        self.max_coils = max_coils
        self.mask = mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x, y = x.to(device=self.mask.device), y.to(device=self.mask.device)
        
        # Pad the width
        target_width = 400
        pad_total = target_width - y.shape[3]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = F.pad(y, (pad_left, pad_right, 0, 0), mode='constant', value=0)
        
        # Pad the height
        target_height = 700
        pad_total = target_height - y.shape[2]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = F.pad(y, (0, 0, pad_left, pad_right), mode='constant', value=0)
        
        # Transform the mask to match the kspace shape
        mask = self.mask.repeat(y.shape[0], y.shape[1], 1, 1)
        
        # Apply the mask to the k-space data
        y = y * mask

        # Add an imaginary part of zeros
        x = torch.cat([x, torch.zeros_like(x)], dim=0)

        # Pad the coil dimension if necessary
        coil_dim = y.shape[1]
        if coil_dim < self.max_coils:
            pad_size = self.max_coils - coil_dim
            y = F.pad(y, (0, 0, 0, 0, 0, pad_size))

        return x, y
