from benchopt import BaseDataset, safe_import_context, config


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import deepinv as dinv
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets import load_dataset
    from benchmark_utils import HuggingFaceTorchDataset, ImageDataset
    from deepinv.physics import Denoising, GaussianNoise


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BSD500_CBSD68"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'task': ['denoising', 'debluring'],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = ["torch", "deepinv", "datasets"]

    def get_data(self):
        img_size = 256  # 64 x 64
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        num_workers = 4 if torch.cuda.is_available() else 0

        if self.task == "denoising":
            noise_level_img = 0.03
            physics = Denoising(GaussianNoise(sigma=noise_level_img))
        elif self.task == "debluring":
            filter_torch = dinv.physics.blur.gaussian_blur(sigma=(3, 3))
            noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
            n_channels = 3  # 3 for color images, 1 for gray-scale images

            physics = dinv.physics.BlurFFT(
                img_size=(n_channels, img_size, img_size),
                filter=filter_torch,
                device=device,
                noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
            )
            #physics = dinv.physics.Inpainting(mask=0.5, tensor_size=(3, 256, 256), device=device)
        else:
            raise Exception("Unknown task")

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # /!\ WARNING : Est-ce qu'il faut redimensionner les images ?
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
            device=device,
            save_dir=config.get_data_path(key="generated_datasets") / "bsd500_cbsd68",
            dataset_filename=self.task,
            num_workers=num_workers,
        )

        train_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)
        test_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=False)

        x, y = train_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        x, y = test_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        batch_size = 2
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        return dict(train_dataloader=train_dataloader, test_dataloader=test_dataloader, physics=physics, dataset_name="BSD68", task_name=self.task)
