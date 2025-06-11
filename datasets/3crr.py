from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import torch
    import numpy as np
    import torchkbnufft as tbn

    import deepinv as dinv
    from deepinv.utils.plotting import plot, plot_curves, scatter_plot, plot_inset
    from deepinv.utils.demo import load_np_url, get_image_url, get_degradation_url
    from deepinv.utils.tensorlist import dirac_like
    from deepinv.physics import RadioInterferometry


class Dataset(BaseDataset):

    name = "3CRR"

    parameters = {}

    requirements = ["datasets"]

    def get_data(self):
        # TODO: Remove
        device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )

        image_gdth = load_np_url(get_image_url("3c353_gdth.npy"))
        image_gdth = torch.from_numpy(image_gdth).unsqueeze(0).unsqueeze(0).to(device)

        def to_logimage(im, rescale=False, dr=5000):
            r"""
            A function plotting the image in logarithmic scale with specified dynamic range
            """
            if rescale:
                im = im - im.min()
                im = im / im.max()
            else:
                im = torch.clamp(im, 0, 1)
            return torch.log10(dr * im + 1.0) / np.log10(dr)

        imgs = [image_gdth, to_logimage(image_gdth)]
        plot(
            imgs,
            titles=[f"Groundtruth", f"Groundtruth in logarithmic scale"],
            cmap="inferno",
            cbar=True,
        )

        uv = load_np_url(get_degradation_url("uv_coordinates.npy"))
        uv = torch.from_numpy(uv).to(device)

        scatter_plot([uv], titles=["uv coverage"], s=0.2, linewidths=0.0)

        # build sensing operator
        physics = RadioInterferometry(
            img_size=image_gdth.shape[-2:],
            samples_loc=uv.permute((1, 0)),
            real=True,
            device=device,
        )

        # Generate the physics
        torch.manual_seed(0)
        y = physics.A(image_gdth)
        noise = (torch.randn_like(y) + 1j * torch.randn_like(y)) / np.sqrt(2)
        y = y + tau * noise

        



        exit()

        return dict(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            physics=physics,
            dataset_name="Set3c",
            task_name=self.task,
            image_size=image_size
        )
