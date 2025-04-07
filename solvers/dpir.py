from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv


class Solver(BaseSolver):
    name = 'DPIR'

    parameters = {}

    sampling_strategy = 'run_once'

    requirements = []

    def set_objective(self, train_dataset, physics):
        batch_size = 2
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        self.device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    def run(self, n_iter):
        self.model = dinv.optim.DPIR(sigma=0.03, device=self.device)

        self.model.eval()

    def get_result(self):
        return dict(model=self.model, model_name="DPIR", device=self.device)
