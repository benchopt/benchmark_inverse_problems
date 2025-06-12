import torch
from deepinv.optim import BaseOptim
from deepinv.models import Denoiser
from deepinv.optim.prior import PnP
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import create_iterator
from deepinv.optim.dpir import get_DPIR_params
from deepinv.models import DRUNet


class DPIR_2C(BaseOptim):
    def __init__(self, sigma=0.1, device="cuda"):
        prior = PnP(denoiser=DPIR_2C_Denoiser(in_channels=1, out_channels=1, pretrained="download", device=device))
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma)
        params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        super(DPIR_2C, self).__init__(
            create_iterator("HQS", prior=prior, F_fn=None, g_first=False),
            max_iter=max_iter,
            data_fidelity=L2(),
            prior=prior,
            early_stop=False,
            params_algo=params_algo,
        )

class DPIR_2C_Denoiser(Denoiser):
    def __init__(self, *DRUNet_args, **DRUNet_kwargs):
        super(DPIR_2C_Denoiser, self).__init__()
        self.model_c1 = DRUNet(*DRUNet_args, **DRUNet_kwargs)
        self.model_c2 = DRUNet(*DRUNet_args, **DRUNet_kwargs)
    
    def forward(self, y, sigma):
        y1, y2 = torch.split(y, 1, dim=1)

        x_hat_1 = self.model_c1(y1, sigma=sigma)
        x_hat_2 = self.model_c2(y2, sigma=sigma)

        x_hat = torch.cat([x_hat_1, x_hat_2], dim=1)
        
        return x_hat
