from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch, torchvision
    from benchmark_utils.fastmri_dataset import FastMRIDataset


class Dataset(BaseDataset):
    name = "FastMRI"

    parameters = {
        'img_size': [128],
    }

    def get_data(self):
        device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        rng = torch.Generator(device=device).manual_seed(0)

        """transform = torchvision.transforms.Resize(self.img_size)
        knee_dataset = dinv.datasets.SimpleFastMRISliceDataset(
            dinv.utils.get_data_home(),
            anatomy="knee",
            transform=transform,
            train=True,
            download=True,
        )
        brain_dataset = dinv.datasets.SimpleFastMRISliceDataset(
            dinv.utils.get_data_home(),
            anatomy="brain",
            transform=transform,
            train=True,
            download=True,
        )
        
        physics_generator = dinv.physics.generator.GaussianMaskGenerator(
            img_size=(self.img_size, self.img_size), # img_size,
            acceleration=4,
            rng=rng,
            device=device
        )
        mask = physics_generator.step()["mask"]

        physics = dinv.physics.MRI(mask=mask,
                                   img_size=(self.img_size, self.img_size),
                                   device=device)
        
        dataset_path = dinv.datasets.generate_dataset(
            train_dataset=knee_dataset,
            test_dataset=brain_dataset,
            val_dataset=None,
            physics=physics,
            physics_generator=physics_generator,
            save_physics_generator_params=True,
            overwrite_existing=True,
            device=device,
            save_dir=config.get_data_path(
                key="fastmri",
            ) / "generated_dataset",
            batch_size=1,
        )

        train_dataset = dinv.datasets.HDF5Dataset(
            dataset_path, split="train"
        )
        test_dataset = dinv.datasets.HDF5Dataset(
            dataset_path, split="test"
        )

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            dataset_name="FastMRI",
            task_name="MRI",
            image_size=(2, 128, 128)
        )"""

        img_size = (320, 320)
        kspace_size = (640, 400)

        physics_generator = dinv.physics.generator.GaussianMaskGenerator(
            img_size=kspace_size,
            acceleration=4,
            rng=rng,
            device=device
        )
        mask = physics_generator.step()["mask"]
        
        train_dataset = FastMRIDataset(dinv.datasets.FastMRISliceDataset(
            config.get_data_path(
                key="fastmri",
            ) / "singlecoil_train",
            slice_index="middle",
        ), mask)

        test_dataset = FastMRIDataset(dinv.datasets.FastMRISliceDataset(
            config.get_data_path(
                key="fastmri",
            ) / "singlecoil_test",
            slice_index="middle",
        ), mask)
        
        x, y = train_dataset[0]
        n_coils = y.shape[2]

        physics = dinv.physics.MultiCoilMRI(
            mask=mask,
            img_size=img_size,
            coil_maps=torch.ones((n_coils,) + kspace_size, dtype=torch.complex64),
            device=device,
        )

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            dataset_name="FastMRI",
            task_name="MRI",
            image_size=(2, 128, 128)
        )
