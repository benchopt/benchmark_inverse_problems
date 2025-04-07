from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from torchvision import transforms
    from datasets import load_dataset
    from benchmark_utils.hugging_face_torch_dataset import HuggingFaceTorchDataset
    from deepinv.physics import Denoising, GaussianNoise


class Dataset(BaseDataset):

    name = "CBSD68_Set3c"

    parameters = {
        'task': ['denoising', 'debluring'],
        'img_size': [256],
    }

    requirements = ["datasets"]

    def get_data(self):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"  # TODO: Remove

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
                device=device
            )
        else:
            raise Exception("Unknown task")

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        dataset_CBSD68 = load_dataset("deepinv/CBSD68")
        train_dataset = HuggingFaceTorchDataset(dataset_CBSD68["train"], key="png", transform=transform)

        dataset_Set3c = load_dataset("deepinv/set3c")
        test_dataset = HuggingFaceTorchDataset(dataset_Set3c["train"], key="image", transform=transform)

        dinv_dataset_path = dinv.datasets.generate_dataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            save_dir=config.get_data_path(key="generated_datasets") / "sbsd68_set3c",
            dataset_filename=self.task,
            device=device
        )

        train_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)
        test_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=False)

        x, y = train_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        x, y = test_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        return dict(train_dataset=train_dataset, test_dataset=test_dataset, physics=physics, dataset_name="Set3c", task_name=self.task)
