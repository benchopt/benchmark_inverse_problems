from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import TensorDataset
    from deepinv.physics import Denoising, GaussianNoise


class Dataset(BaseDataset):
    name = "simulated"

    parameters = {}

    requirements = ["datasets"]

    def get_data(self):
        original_data = torch.ones(32, 3, 32, 32)
        noisy_data = original_data + torch.rand(32, 3, 32, 32)

        train_dataset = TensorDataset(noisy_data, original_data)
        test_dataset = TensorDataset(noisy_data, original_data)

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=Denoising(GaussianNoise(sigma=0.03)),
            dataset_name="simulated",
            task_name="test"
        )
