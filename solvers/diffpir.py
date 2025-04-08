from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv


class Solver(BaseSolver):
    name = 'DiffPIR'

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

        self.model = dinv.sampling.DiffPIR(
            model=denoiser,
            data_fidelity=dinv.optim.data_fidelity.L2()
        )
        self.model.eval()

    def get_result(self):
        return dict(model=self.model, model_name="DiffPIR", device=self.device)
