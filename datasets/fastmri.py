from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from benchmark_utils.fastmri_dataset import FastMRIDataset

MAX_COILS = 32  # Maximum number of coils to pad to
KSPACE_PADDED_SIZE = (700, 400)  # K-space size for FastMRI dataset


class Dataset(BaseDataset):
    name = "FastMRI"

    parameters = {}

    def get_data(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        rng = torch.Generator(device=device).manual_seed(0)

        physics_generator = dinv.physics.generator.GaussianMaskGenerator(
            img_size=KSPACE_PADDED_SIZE, acceleration=4, rng=rng, device=device
        )
        mask = physics_generator.step(
            batch_size=1, img_size=KSPACE_PADDED_SIZE
        )["mask"]

        train_dataset = FastMRIDataset(dinv.datasets.FastMRISliceDataset(
            config.get_data_path(key="fastmri_train"), slice_index="middle"
        ), mask, MAX_COILS)

        test_dataset = FastMRIDataset(dinv.datasets.FastMRISliceDataset(
            config.get_data_path(key="fastmri_test"), slice_index="middle"
        ), mask, MAX_COILS)

        x, y = train_dataset[0]

        img_size, kspace_shape = x.shape[-2:], KSPACE_PADDED_SIZE

        physics = dinv.physics.MultiCoilMRI(
            img_size=img_size,
            mask=mask,
            coil_maps=torch.ones(
                (MAX_COILS,) + kspace_shape,
                dtype=torch.complex64
            ),
            device=device,
        )

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            dataset_name="FastMRI",
            task_name="MRI",
            image_size=y.shape
        )
