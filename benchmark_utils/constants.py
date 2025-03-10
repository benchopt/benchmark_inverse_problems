import torch
import deepinv as dinv


def constants():
    return dict(
        device=dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu",
        num_workers=4 if torch.cuda.is_available() else 0,
        noise_level_img=0.03,
        img_size=224,
    )
