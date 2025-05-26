from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv
    import numpy as np


class Solver(BaseSolver):
    name = 'DPIR'

    parameters = {}

    sampling_strategy = 'run_once'

    requirements = []

    def set_objective(self, train_dataset, physics, image_size):
        batch_size = 2
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        self.device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )
        self.physics = physics
        self.image_size = image_size

    def run(self, n_iter):
        best_sigma = 0
        best_psnr = 0
        lr = 0.001

        # If the number of channels is different from 1 or 3
        # then we can't use pretrained DRUNet
        if self.image_size[0] not in [1, 3]:
            noise_level_img = 0.03
            denoiser = dinv.models.DRUNet(in_channels=self.image_size[0],
                                          out_channels=self.image_size[0],
                                          pretrained=None,
                                          device=self.device)


        else:
            for sigma in np.linspace(0.01, 0.1, 10):
                model = dinv.optim.DPIR(sigma=sigma, device=self.device)

                results = dinv.test(
                    model,
                    self.train_dataloader,
                    self.physics,
                    metrics=[dinv.metric.PSNR(), dinv.metric.SSIM()],
                    device=self.device
                )

                if results["PSNR"] > best_psnr:
                    best_sigma = sigma
                    best_psnr = results["PSNR"]

            self.model = dinv.optim.DPIR(sigma=best_sigma, device=self.device)
        self.model.eval()

    def get_result(self):
        return dict(model=self.model, model_name="DPIR", device=self.device)
