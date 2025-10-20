import numpy as np
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv


class UnetSR(torch.nn.Module):
    def __init__(self, scale_factor):
        super(UnetSR, self).__init__()
        self.scale_factor = scale_factor
        self.unet = dinv.models.UNet(
            in_channels=3, out_channels=3*scale_factor**2,
            scales=3, batch_norm=False
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(scale_factor)
        self.conv = torch.nn.Conv2d(
            3, 3, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x_input, physics, **kwargs):
        x = self.unet(x_input, physics, **kwargs)
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        return x


class Solver(BaseSolver):

    name = 'UNet'

    parameters = {
        'lr': list(np.logspace(-5, -2, 4))
    }

    sampling_strategy = 'run_once'

    requirements = []

    def set_objective(self, train_dataset, physics):
        batch_size = 2
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )

        # Determine scale factor for super-resolution tasks
        hr, lr = next(iter(self.train_dataloader))
        _, _, h_lr, _ = lr.shape
        _, _, h_hr, _ = hr.shape
        if h_lr != h_hr:
            self.scale_factor = h_hr // h_lr
        else:
            self.scale_factor = None

        self.device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )
        self.physics = physics.to(self.device)

    def run(self, n_iter):
        epochs = 4

        if self.scale_factor is not None:
            model = UnetSR(self.scale_factor).to(self.device)
        else:
            model = dinv.models.UNet(
                in_channels=3, out_channels=3, scales=3, batch_norm=False
            ).to(self.device)

        verbose = True  # print training information
        wandb_vis = False  # plot curves and images in Weight&Bias

        # choose training losses
        losses = dinv.loss.SupLoss(metric=dinv.metric.MSE())

        # choose optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-8
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(epochs * 0.8)
        )
        trainer = dinv.Trainer(
            model,
            device=self.device,
            verbose=verbose,
            wandb_vis=wandb_vis,
            physics=self.physics,
            epochs=epochs,
            scheduler=scheduler,
            losses=losses,
            optimizer=optimizer,
            show_progress_bar=True,
            train_dataloader=self.train_dataloader,
        )

        self.model = trainer.train()
        self.model.eval()

    def get_result(self):
        return dict(model=self.model, model_name="U-Net", device=self.device)
