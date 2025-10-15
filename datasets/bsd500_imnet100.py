from benchopt import BaseDataset, safe_import_context, config
from pathlib import Path
import os

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from torchvision import transforms
    from benchmark_utils.image_dataset import ImageDataset
    from benchmark_utils.hugging_face_torch_dataset import (
        HuggingFaceTorchDataset
    )
    from deepinv.physics import Downsampling, Denoising, GaussianNoise
    from deepinv.physics.generator import MotionBlurGenerator
    from datasets import load_dataset
    import kagglehub


class Dataset(BaseDataset):

    name = "BSD500_imnet100"

    parameters = {
        'task': ['denoising',
                 'gaussian-debluring',
                 'motion-debluring',
                 'SRx4'],
        'img_size': [256],
    }

    requirements = ["datasets", "kagglehub"]

    def get_data(self):
        # TODO: Remove
        device = (
            dinv.utils.get_freer_gpu()) if torch.cuda.is_available() else "cpu"

        if self.task == "denoising":
            noise_level_img = 0.03
            physics = Denoising(GaussianNoise(sigma=noise_level_img))
        elif self.task == "gaussian-debluring":
            filter_torch = dinv.physics.blur.gaussian_blur(sigma=(3, 3))
            noise_level_img = 0.03
            n_channels = 3

            physics = dinv.physics.BlurFFT(
                img_size=(n_channels, self.img_size, self.img_size),
                filter=filter_torch,
                noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
                device=device
            )
        elif self.task == "motion-debluring":
            psf_size = 31
            n_channels = 3
            motion_generator = MotionBlurGenerator(
                (psf_size, psf_size),
                device=device
            )

            filters = motion_generator.step(batch_size=1)

            physics = dinv.physics.BlurFFT(
                img_size=(n_channels, self.img_size, self.img_size),
                filter=filters["filter"],
                device=device
            )
        elif self.task == "SRx4":
            n_channels = 3
            physics = Downsampling(img_size=(n_channels,
                                             self.img_size,
                                             self.img_size),
                                   filter="bicubic",
                                   factor=4,
                                   device=device)
        else:
            raise Exception("Unknown task")

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        path = Path(kagglehub.dataset_download(
            "balraj98/berkeley-segmentation-dataset-500-bsds500"))
        path = path / "images"

        train_dataset = ImageDataset(
            path / "train",
            transform=transform
        )

        dataset_miniImnet100 = load_dataset("mterris/miniImnet100")
        test_dataset = HuggingFaceTorchDataset(
            dataset_miniImnet100["validation"],
            key="image",
            transform=transform
        )

        data_path = Path(os.path.dirname(os.path.abspath(__file__)))
        data_path = data_path.parent / "data"

        dinv_dataset_path = dinv.datasets.generate_dataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            save_dir=data_path / "bsd500_imnet100",
            dataset_filename=self.task,
            device=device
        )

        train_dataset = dinv.datasets.HDF5Dataset(
            path=dinv_dataset_path, train=True
        )
        test_dataset = dinv.datasets.HDF5Dataset(
            path=dinv_dataset_path, train=False
        )

        x, y = train_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        x, y = test_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            dataset_name="BSD68",
            task_name=self.task
        )
