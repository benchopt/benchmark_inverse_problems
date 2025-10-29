from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from torchvision import transforms
    from datasets import load_dataset
    from benchmark_utils.hugging_face_torch_dataset import (
        HuggingFaceTorchDataset
    )
    from deepinv.physics import (
        Denoising,
        GaussianNoise,
        Downsampling,
        Demosaicing
    )
    from deepinv.physics.generator import MotionBlurGenerator


class Dataset(BaseDataset):

    name = "BSD500_CBSD68"

    parameters = {
        'task': ['denoising',
                 'gaussian-debluring',
                 'motion-debluring',
                 'SRx4',
                 'inpainting',
                 'demosaicing'],
        'img_size': [256],
    }

    requirements = ["datasets"]

    def get_data(self):
        # TODO: Remove
        device = (
            dinv.utils.get_freer_gpu()) if torch.cuda.is_available() else "cpu"

        n_channels = 3
        img_size = (n_channels, self.img_size, self.img_size)

        if self.task == "denoising":
            noise_level_img = 0.1
            physics = Denoising(GaussianNoise(sigma=noise_level_img))
        elif self.task == "gaussian-debluring":
            filter_torch = dinv.physics.blur.gaussian_blur(sigma=(3, 3))
            noise_level_img = 0.03

            physics = dinv.physics.BlurFFT(
                img_size=img_size,
                filter=filter_torch,
                noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
                device=device
            )
        elif self.task == "motion-debluring":
            psf_size = 31
            motion_generator = MotionBlurGenerator(
                (psf_size, psf_size),
                device=device
            )

            filters = motion_generator.step(batch_size=1)

            physics = dinv.physics.BlurFFT(
                img_size=img_size,
                filter=filters["filter"],
                device=device
            )
        elif self.task == "SRx4":
            physics = Downsampling(img_size=img_size,
                                   filter="bicubic",
                                   factor=4,
                                   device=device)
        elif self.task == "inpainting":
            physics = dinv.physics.Inpainting(img_size,
                                              mask=0.7,
                                              device=device)
        elif self.task == "demosaicing":
            physics = Demosaicing(img_size=img_size,
                                  device=device)
        else:
            raise Exception("Unknown task")

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

        dataset_cbsd68 = load_dataset("deepinv/CBSD68")
        test_dataset = HuggingFaceTorchDataset(
            dataset_cbsd68["train"],
            key="png",
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
