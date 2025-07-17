import torch
from torch.utils.data import Dataset
from deepinv.datasets import FastMRISliceDataset
import torch.nn.functional as F

class FastMRIDataset(Dataset):
    def __init__(self, dataset: FastMRISliceDataset, mask):
        self.dataset = dataset
        self.mask = mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load and return a sample from the dataset
        x, y = self.dataset[idx]
        x, y = x.to(self.mask.device), y.to(self.mask.device)
        
        target_width = 400
        pad_total = target_width - y.shape[2]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = F.pad(y, (pad_left, pad_right))
        
        y *= self.mask.squeeze(0)
        x = torch.cat([x, torch.zeros_like(x)], dim=0)

        return x, y
