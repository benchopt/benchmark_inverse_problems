from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from torch.utils.data import DataLoader
    import deepinv as dinv
    from benchmark_utils.denoiser_2c import Denoiser_2c
    from benchmark_utils.helper import get_device


class Solver(BaseSolver):
    name = 'DiffPIR'

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
        if self.image_size[0] == 2:
            denoiser = Denoiser_2c(device=self.device)
        else:
            denoiser = dinv.models.DRUNet(
                pretrained="download",
                device=self.device
            )

        self.model = dinv.sampling.DiffPIR(
            model=denoiser,
            data_fidelity=dinv.optim.data_fidelity.L2(),
            device=self.device
        )

        self.model.eval()

    def get_result(self):
        return dict(model=self.model, model_name="DiffPIR", device=self.device)
