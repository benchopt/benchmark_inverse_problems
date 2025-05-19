from benchopt import BaseDataset, safe_import_context, config

with (safe_import_context() as import_ctx):
    import deepinv as dinv
    import torch, torchvision


class Dataset(BaseDataset):
    name = "FastMRI"

    parameters = {
        'img_size': [128],
    }

    def get_data(self):
        device = dinv.utils.get_freer_gpu if torch.cuda.is_available() else "cpu"
        rng = torch.Generator(device=device).manual_seed(0)

        transform = torchvision.transforms.Resize(self.img_size)
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

        dinv.utils.plot({"knee": knee_dataset[0], "brain": brain_dataset[0]})

        physics_generator = dinv.physics.generator.GaussianMaskGenerator(
            img_size=(self.img_size, self.img_size),
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
            overwrite_existing=False,
            device=device,
            save_dir=config.get_data_path(
                key="generated_datasets"
            ) / "fastmri",
            batch_size=1,
        )

        train_dataset = dinv.datasets.HDF5Dataset(
            dataset_path, split="train", load_physics_generator_params=True
        )
        test_dataset = dinv.datasets.HDF5Dataset(
            dataset_path, split="test", load_physics_generator_params=True
        )

        dinv.utils.plot(
            {
                "x0": train_dataset[0][0],
                "mask0": train_dataset[0][2]["mask"],
                "x1": train_dataset[1][0],
                "mask1": train_dataset[1][2]["mask"],
            }
        )

        x, y, z = train_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0), z["mask"].unsqueeze(0)])

        x, y, z = test_dataset[0]
        dinv.utils.plot([x.unsqueeze(0), y.unsqueeze(0), z["mask"].unsqueeze(0)])

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            dataset_name="FastMRI",
            task_name="MRI",
            image_size=(2, 128, 128)
        )
