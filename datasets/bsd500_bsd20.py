from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from torchvision import transforms
    from benchmark_utils.image_dataset import ImageDataset
    from deepinv.physics import (
        Downsampling,
        Denoising,
        GaussianNoise,
        Demosaicing
    )
    from deepinv.physics.generator import MotionBlurGenerator


class Dataset(BaseDataset):

    name = "BSD500_BSD20"

    parameters = {
        'task': ['denoising',
                 'gaussian-debluring',
                 'motion-debluring',
                 'SRx4',
                 'demosaicing'],
        'img_size': [256],
    }

    requirements = ["datasets"]

    def get_data(self):
        # TODO: Remove
        device = (
            dinv.utils.get_freer_gpu()) if torch.cuda.is_available() else "cpu"

        n_channels = 3

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
            physics = Downsampling(img_size=(n_channels,
                                             self.img_size,
                                             self.img_size),
                                   filter="bicubic",
                                   factor=4,
                                   device=device)
        elif self.task == "demosaicing":
            physics = Demosaicing(img_size=(n_channels,
                                            self.img_size,
                                            self.img_size),
                                  device=device)
        else:
            raise Exception("Unknown task")

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        train_dataset = ImageDataset(
            config.get_data_path("BSD500") / "train",
            transform=transform
        )

        test_dataset = ImageDataset(
            config.get_data_path("BSD500") / "val",
            transform=transform,
            num_images=20
        )

        dinv_dataset_path = dinv.datasets.generate_dataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            save_dir=config.get_data_path(
                key="generated_datasets"
            ) / "bsd500_bsd20",
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
