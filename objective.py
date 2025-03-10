import torch
from benchopt import BaseObjective, safe_import_context, config

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import deepinv as dinv
    from deepinv.training import test
    from benchmark_utils import constants


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Inverse Problems"

    # URL of the main repo for this benchmark.
    url = "https://github.com/benchopt/benchmark_inverse_problems"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'task': ['denoising']
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, train_dataloader, test_dataloader, physics):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.physics = physics

    def evaluate_result(self, model):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        x, y = next(iter(self.test_dataloader))

        if isinstance(model, dinv.optim.DPIR):  # /!\ To remove
            x_hat = model(y, self.physics)
        else:
            x_hat = model(y, 0.03 * 2)

        #dinv.utils.plot([x[0], y[0], x_hat[0]], ["Ground Truth", "Measurement", "Reconstruction"], rescale_mode="clip")

        m = dinv.loss.metric.PSNR()
        psnr = m(x_hat, x)

        m = dinv.loss.metric.SSIM()
        ssim = m(x_hat, x)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=torch.mean(psnr).item(),
            ssim=torch.mean(ssim).item()
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(beta=np.zeros(self.X.shape[1]))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(train_dataloader=self.train_dataloader)
