from benchopt import BaseObjective, safe_import_context, config

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
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
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, train_dataloader, test_dataloader, physics, dataset_name, task_name):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
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

        if isinstance(model, dinv.models.DeepImagePrior):
            results = dict(PSNR=torch.empty(0).to(device), SSIM=torch.empty(0).to(device))

            for x, y in self.test_dataloader:
                x, y = x.to(device), y.to(device)
                x_hat = torch.empty((0, 3, 256, 256)).to(device)

                for y_alone in y:
                    x_hat_alone = model(y_alone.unsqueeze(0), self.physics)
                    x_hat = torch.cat([x_hat, x_hat_alone])

            # x_hat = torch.tensor([model(y_alone.unsqueeze(0), self.physics) for x, y in self.test_dataloader for y_alone in y])

                results["PSNR"] = torch.cat([results["PSNR"], dinv.metric.PSNR()(x_hat, x)])
                results["SSIM"] = torch.cat([results["SSIM"], dinv.metric.SSIM()(x_hat, x)])

            results["PSNR"] = results["PSNR"].mean().item()
            results["SSIM"] = results["SSIM"].mean().item()
        else:
            results = dinv.test(
                model,
                self.test_dataloader,
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
        return dict(train_dataloader=self.train_dataloader, physics=self.physics)
