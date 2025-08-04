import numpy as np
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv
    import torchvision
    from benchmark_utils.metrics import CustomMSE, CustomPSNR
    from benchmark_utils.custom_models import MRIUNet


class Solver(BaseSolver):

    name = 'UNet'

    parameters = {
        'lr': list(np.logspace(-5, -2, 4))
    }

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
        self.physics = physics.to(self.device)
        self.image_size = image_size
        self.dataset_name = dataset_name

    def run(self, n_iter):
        epochs = 4
        
        x, y = next(iter(self.train_dataloader))

        if self.dataset_name == 'FastMRI':
            model = MRIUNet(
                in_channels=y.shape[1] * y.shape[2], out_channels=x.shape[1], scales=3,
                batch_norm=False
            ).to(self.device)
        else:
            model = dinv.models.UNet(
                in_channels=y.shape[1], out_channels=x.shape[1], scales=3,
                batch_norm=False
            ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-8
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(epochs * 0.8)
        )

        # choose training losses
        if self.dataset_name == 'FastMRI':
            criterion = dinv.loss.SupLoss(metric=CustomMSE())
        else:
            criterion = dinv.loss.SupLoss(metric=dinv.metric.MSE())
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                x_hat = model(y, self.physics)
                
                if self.dataset_name == 'FastMRI':
                    transform = torchvision.transforms.Compose(
                        [
                            torchvision.transforms.CenterCrop(x.shape[-2:]),
                            dinv.metric.functional.complex_abs,
                        ]
                    )
                    criterion.metric.transform = transform
                loss = criterion(x_hat, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        
        self.model = model

    def get_result(self):
        return dict(model=self.model, model_name="U-Net", device=self.device)
