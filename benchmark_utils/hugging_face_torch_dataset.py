import torch


class HuggingFaceTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, key, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.key = key

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample[self.key]  # Image PIL

        if self.transform:
            image = self.transform(image)

        return image
