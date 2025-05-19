from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv
    import numpy as np


class Solver(BaseSolver):
    name = 'DDRM'

    parameters = {}

    sampling_strategy = 'run_once'

    requirements = []

    def set_objective(self, train_dataset, physics):
        batch_size = 2
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        self.device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )
        self.physics = physics

    def run(self, n_iter):
        denoiser = dinv.models.DRUNet(pretrained="download").to(self.device)

        sigmas = (np.linspace(1,0, 100)
                  if torch.cuda.is_available()
                  else np.linspace(1, 0, 10))

        self.model = dinv.sampling.DDRM(
            denoiser=denoiser,
            etab=1.0,
            sigmas=sigmas,
            verbose=True
        )
        self.model.eval()

    def get_result(self):
        return dict(model=self.model, model_name="DiffPIR", device=self.device)
