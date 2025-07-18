import torch
from torch.utils.data import Dataset
from deepinv.datasets import FastMRISliceDataset
import torch.nn.functional as F

class FastMRIDataset(Dataset):
    def __init__(self, dataset: FastMRISliceDataset):
        self.dataset = dataset
        self.mask = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #if self.mask is None:
        #    raise ValueError("Mask must be set before getting items.")
        
        # Load and return a sample from the dataset
        x, y = self.dataset[idx]
        
        #target_width = 400
        #pad_total = target_width - y.shape[2]
        #pad_left = pad_total // 2
        #pad_right = pad_total - pad_left
        #y = F.pad(y, (pad_left, pad_right))

        if self.mask is not None:
            x, y = x.to(self.mask.device), y.to(self.mask.device)
            y = y * self.mask.squeeze(0)
        x = torch.cat([x, torch.zeros_like(x)], dim=0)
        y = y.reshape(8, y.shape[2], y.shape[3])

        return x, y
