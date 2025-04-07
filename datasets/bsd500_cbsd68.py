from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import deepinv as dinv
    from torchvision import transforms
    from datasets import load_dataset
    from benchmark_utils.image_dataset import ImageDataset
    from benchmark_utils.hugging_face_torch_dataset import HuggingFaceTorchDataset
    from deepinv.physics import Denoising, GaussianNoise


class Dataset(BaseDataset):

    name = "BSD500_CBSD68"

    parameters = {
        'task': ['denoising', 'debluring'],
        'img_size': [256],
    }

    requirements = ["datasets"]

    def get_data(self):
        if self.task == "denoising":
            noise_level_img = 0.03
            physics = Denoising(GaussianNoise(sigma=noise_level_img))
        elif self.task == "debluring":
            filter_torch = dinv.physics.blur.gaussian_blur(sigma=(3, 3))
            noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
            n_channels = 3  # 3 for color images, 1 for gray-scale images

            physics = dinv.physics.BlurFFT(
                img_size=(n_channels, self.img_size, self.img_size),
                filter=filter_torch,
                noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
                device="cuda"
            )
            #physics = dinv.physics.Inpainting(mask=0.5, tensor_size=(3, 256, 256), device=device)
        else:
            raise Exception("Unknown task")

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            #transforms.CenterCrop((32, 32)),
            transforms.ToTensor()
        ])

        train_dataset = ImageDataset(config.get_data_path("BSD500") / "train", transform=transform)

        dataset_cbsd68 = load_dataset("deepinv/CBSD68")
        test_dataset = HuggingFaceTorchDataset(dataset_cbsd68["train"], key="png", transform=transform)

        dinv_dataset_path = dinv.datasets.generate_dataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            save_dir=config.get_data_path(key="generated_datasets") / "bsd500_cbsd68",
            dataset_filename=self.task,
            device="cuda"
        )

        train_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)
        test_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=False)

        x, y = train_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        x, y = test_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        return dict(train_dataset=train_dataset, test_dataset=test_dataset, physics=physics, dataset_name="BSD68", task_name=self.task)
