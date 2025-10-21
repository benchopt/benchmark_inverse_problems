from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv
    import numpy as np
    import torchvision
    from deepinv.optim import BaseOptim
    from deepinv.optim.prior import PnP
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim.optimizers import create_iterator
    from deepinv.optim.dpir import get_DPIR_params
    from benchmark_utils.denoiser_2c import Denoiser_2c
    from benchmark_utils.metrics import CustomPSNR
    from tqdm import tqdm


class Solver(BaseSolver):
    name = 'DPIR'

    parameters = {}

    sampling_strategy = 'run_once'

    requirements = []

    def set_objective(self, train_dataset, physics, image_size, dataset_name):
        batch_size = 1
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        self.device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )
        self.physics = physics
        self.image_size = image_size
        self.dataset_name = dataset_name

    def run(self, n_iter):
        best_sigma = 0
        best_psnr = 0

        # If the number of channels is 2 we use a custom DPIR solver
        if self.image_size[0] == 2:
            model_class = DPIR_2C
        else:
            model_class = dinv.optim.DPIR

        # If the number of channels is different from 1 or 3
        # then we can't use pretrained DRUNet
        for sigma in np.linspace(0.01, 0.1, 10):
            model = model_class(sigma=sigma, device=self.device)

            psnr = []

            for x, y in tqdm(self.train_dataloader, desc=f"DPIR : Looking for the best sigma"):
                x, y = x.to(self.device), y.to(self.device)

                x_hat = model(y, self.physics)

                if (self.dataset_name == 'FastMRI'):
                    transform = torchvision.transforms.Compose(
                        [
                            torchvision.transforms.CenterCrop(x.shape[-2:]),
                            dinv.metric.functional.complex_abs,
                        ]
                    )

                    CustomPSNR.transform = transform

                    psnr.append(CustomPSNR()(x_hat, x))
                else:
                    psnr.append(dinv.metric.PSNR()(x_hat, x))

            psnr = torch.mean(torch.cat(psnr)).item()

            if psnr > best_psnr:
                best_sigma = sigma
                best_psnr = psnr

            self.model = model_class(sigma=best_sigma, device=self.device)
        self.model.eval()

    def get_result(self):
        return dict(model=self.model, model_name="DPIR", device=self.device)


# Custom DPIR solver with 2 channels
class DPIR_2C(BaseOptim):
    def __init__(self, sigma=0.1, device="cuda"):
        prior = PnP(denoiser=Denoiser_2c(device=device))
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma)
        params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        super(DPIR_2C, self).__init__(
            create_iterator("HQS", prior=prior, F_fn=None, g_first=False),
            max_iter=max_iter,
            data_fidelity=L2(),
            prior=prior,
            early_stop=False,
            params_algo=params_algo,
        )
