
import torch
from deepinv.models import DRUNet
from deepinv.models import Denoiser

class Denoiser_2c(Denoiser):
    def __init__(self, device):
        super(Denoiser_2c, self).__init__()
        self.model_c1 = DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)
        self.model_c2 = DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)

    def forward(self, y, sigma):
        y1, y2 = torch.split(y, 1, dim=1)

        x_hat_1 = self.model_c1(y1, sigma=sigma)
        x_hat_2 = self.model_c2(y2, sigma=sigma)

        x_hat = torch.cat([x_hat_1, x_hat_2], dim=1)
        
        return x_hat
