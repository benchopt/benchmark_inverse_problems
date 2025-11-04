import numpy as np
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv
    from benchmark_utils.helper import get_device


class Solver(BaseSolver):

    name = 'UNet'

    parameters = {
        'lr': list(np.logspace(-5, -2, 4))
    }

    sampling_strategy = 'run_once'

    requirements = []

    def set_objective(self, train_dataset, physics, image_size, batch_size):
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        self.device = get_device()
        self.physics = physics.to(self.device)
        self.image_size = image_size

    def run(self, n_iter):
        epochs = 4

        model = dinv.models.UNet(
            in_channels=3, out_channels=3, scales=4,
            batch_norm=False
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-8
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(epochs * 0.7)
        )

        criterion = dinv.loss.SupLoss(metric=dinv.metric.MSE())

        trainer = dinv.Trainer(
            model,
            device=self.device,
            verbose=True,
            wandb_vis=False,
            physics=self.physics,
            epochs=epochs,
            scheduler=scheduler,
            losses=criterion,
            optimizer=optimizer,
            show_progress_bar=True,
            train_dataloader=self.train_dataloader,
        )

        self.model = trainer.train()
        self.model.eval()

    def get_result(self):
        return dict(
            model=self.model,
            model_name=f"U-Net_{self.lr}",
            device=self.device
        )
