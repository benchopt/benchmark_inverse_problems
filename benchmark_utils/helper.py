from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    from deepinv.physics import (
        Denoising,
        GaussianNoise,
        Downsampling,
        Demosaicing
    )
    from deepinv.physics.generator import MotionBlurGenerator


DEVICE = None


def get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE
    if dinv.torch.cuda.is_available():
        DEVICE = dinv.utils.get_freer_gpu()
    else:
        DEVICE = "cpu"
    return DEVICE


def get_task_physic(task, img_size, device):
    if task == "denoising":
        noise_level_img = 0.1
        physics = Denoising(GaussianNoise(sigma=noise_level_img))
    elif task == "gaussian-debluring":
        filter_torch = dinv.physics.blur.gaussian_blur(sigma=(3, 3))
        noise_level_img = 0.03

        physics = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=filter_torch,
            noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
            device=device
        )
    elif task == "motion-debluring":
        psf_size = 31
        motion_generator = MotionBlurGenerator(
            (psf_size, psf_size),
            device=device
        )

        filters = motion_generator.step(batch_size=1)

        physics = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=filters["filter"],
            device=device
        )
    elif task == "SRx4":
        physics = Downsampling(
            img_size=img_size,
            filter="bicubic",
            factor=4,
            device=device
        )
    elif task == "inpainting":
        physics = dinv.physics.Inpainting(
            img_size,
            mask=0.7,
            device=device
        )
    elif task == "demosaicing":
        physics = Demosaicing(
            img_size=img_size,
            device=device
        )
    else:
        raise Exception("Unknown task")
    return physics
