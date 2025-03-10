from benchopt import BaseDataset, safe_import_context, config


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import deepinv as dinv
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets import load_dataset
    from benchmark_utils import HuggingFaceTorchDataset, constants
    from deepinv.physics import Denoising, GaussianNoise


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BSD500_CBSD68"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'random_state': [27],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = ["torch", "deepinv", "datasets"]

    def get_data(self):
        noise_level_img = constants()["noise_level_img"]
        img_size = constants()["img_size"]
        physics = Denoising(GaussianNoise(sigma=noise_level_img))
        device = constants()["device"]
        num_workers = constants()["num_workers"]

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # /!\ WARNING : Est-ce qu'il faut redimensionner les images ?
            transforms.ToTensor()
        ])

        dataset_CBSD68 = load_dataset("delta-prox/BSD500")
        train_dataset = HuggingFaceTorchDataset(dataset_CBSD68["train"], key="image", transform=transform)

        dataset_Set3c = load_dataset("deepinv/CBSD68")
        test_dataset = HuggingFaceTorchDataset(dataset_Set3c["train"], key="png", transform=transform)

        dinv_dataset_path = dinv.datasets.generate_dataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            device=device,
            save_dir=config.get_data_path(key="CBSD68_BSD500"),
            num_workers=num_workers,
        )

        train_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)
        test_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=False)

        #x, y = train_dataset[0]
        #dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        #x, y = test_dataset[0]
        #dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0)])

        batch_size = 2
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        test_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        return dict(train_dataloader=train_dataloader, test_dataloader=test_dataloader, physics=physics)
