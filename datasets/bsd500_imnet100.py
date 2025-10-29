from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from torchvision import transforms
    from benchmark_utils.hugging_face_torch_dataset import (
        HuggingFaceTorchDataset
    )
    from benchmark_utils.helper import get_task_physic
    from datasets import load_dataset


class Dataset(BaseDataset):

    name = "BSD500_imnet100"

    parameters = {
        'task': [
            'denoising',
            'gaussian-debluring',
            'motion-debluring',
            'SRx4',
            'inpainting',
            'demosaicing'
        ],
        'img_size': [256],
    }

    requirements = ["datasets"]

    def get_data(self):
        # TODO: Remove
        device = (
            dinv.utils.get_freer_gpu()) if torch.cuda.is_available() else "cpu"

        n_channels = 3
        img_size = (n_channels, self.img_size, self.img_size)

        physics = get_task_physic(self.task, img_size, device)

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        path = get_data_path("BSD500")
        bsd500_dataset = dinv.datasets.BSDS500(
            path, download=True, transform=transform
        )
        train_dataset = HuggingFaceTorchDataset(
            bsd500_dataset,
            key=...,
            physics=physics,
            device=device,
            transform=transforms.Resize((self.img_size, self.img_size))
        )

        dataset_miniImnet100 = load_dataset("mterris/miniImnet100")
        test_dataset = HuggingFaceTorchDataset(
            dataset_miniImnet100["validation"],
            key="image",
            physics=physics,
            device=device,
            transform=transform
        )

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            dataset_name="BSD68",
            task_name=self.task,
            image_size=img_size
        )
