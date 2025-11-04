from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv
    import numpy as np
    from tqdm import tqdm
    from benchmark_utils.helper import get_device


class Solver(BaseSolver):
    name = 'DPIR'

    parameters = {}

    sampling_strategy = 'run_once'

    requirements = []

    def set_objective(self, train_dataset, physics, image_size, batch_size):
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        self.device = get_device()
        self.physics = physics
        self.image_size = image_size

    def run(self, n_iter):
        best_sigma = 0
        best_psnr = 0

        # If the number of channels is different from 1 or 3
        # then we can't use pretrained DRUNet
        for sigma in np.linspace(0.01, 0.1, 10):
            model = dinv.optim.DPIR(sigma=sigma, device=self.device)

            psnr = []

            bar = tqdm(
                self.train_dataloader,
                desc="DPIR : Looking for the best sigma"
            )
            for x, y in bar:
                x, y = x.to(self.device), y.to(self.device)

                x_hat = model(y, self.physics)

                psnr.append(dinv.metric.PSNR()(x_hat, x))

            psnr = torch.mean(torch.cat(psnr)).item()

            if psnr > best_psnr:
                best_sigma = sigma
                best_psnr = psnr

            self.model = dinv.optim.DPIR(sigma=best_sigma, device=self.device)
        self.model.eval()

    def get_result(self):
        return dict(model=self.model, model_name="DPIR", device=self.device)
