from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv
    import numpy as np


class Solver(BaseSolver):
    name = 'IFFT2'

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
        def model(y):
            return self.physics.A_adjoint(y)

        self.model = model

    def get_result(self):
        return dict(model=self.model, model_name="IFFT2", device=self.device)
