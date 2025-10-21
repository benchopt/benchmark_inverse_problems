import torch


class HuggingFaceTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, key, physics, device, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.key = key
        self.device = device
        self.physics = physics

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        x = sample[self.key]  # Image PIL

        if self.transform:
            x = self.transform(x)

        x = x.to(self.device)

        y = self.physics(x.unsqueeze(0))
        y = y.squeeze(0)

        return x, y
