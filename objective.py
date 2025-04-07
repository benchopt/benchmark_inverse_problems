from benchopt import BaseObjective, safe_import_context, config

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import numpy as np
    import deepinv as dinv


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
    parameters = {}

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = ["torch", "numpy", "deepinv"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, train_dataset, test_dataset, physics, dataset_name, task_name):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.physics = physics
        self.dataset_name = dataset_name
        self.task_name = task_name

    def evaluate_result(self, model, model_name, device):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        #x, y = next(iter(self.test_dataloader))
        #x, y = x.to(device), y.to(device)

        #if isinstance(model, dinv.models.DeepImagePrior):
        #    y = y[:1, :, :, :]  # We keep the first image of the batch to make a batch of size 1

        #x_hat = model(y, self.physics)

        #dinv.utils.plot([x[0], y[0], x_hat[0]], ["Ground Truth", "Measurement", "Reconstruction"], suptitle=f"{self.task_name} with {model_name} on {self.dataset_name}", rescale_mode="clip", save_dir=f"./figs/{self.task_name}_with_{model_name}_on_{self.dataset_name}")

        batch_size = 2
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

        if isinstance(model, dinv.models.DeepImagePrior):
            psnr = []
            ssim = []

            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                x_hat = torch.cat([
                    model(y_i[None], self.physics) for y_i in y
                ])
                psnr.append(dinv.metric.PSNR()(x_hat, x))
                ssim.append(dinv.metric.SSIM()(x_hat, x))

            psnr = torch.mean(torch.cat(psnr)).item()
            ssim = torch.mean(torch.cat(ssim)).item()

            results = dict(PSNR=psnr, SSIM=ssim)
        else:
            results = dinv.test(
                model,
                test_dataloader,
                self.physics,
                metrics=[dinv.metric.PSNR(), dinv.metric.SSIM()],
                device=device
            )

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=results["PSNR"],
            ssim=results["SSIM"],
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
        return dict(train_dataset=self.train_dataset, physics=self.physics)
