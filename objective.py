from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
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
    requirements = ["pytorch:pytorch", "numpy", "deepinv", "pyiqa"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self,
                 train_dataset,
                 test_dataset,
                 physics,
                 dataset_name,
                 task_name):
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

        batch_size = 2
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

        if isinstance(model, dinv.models.DeepImagePrior):
            psnr = []
            ssim = []
            lpips = []

            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                x_hat = torch.cat([
                    model(y_i[None], self.physics) for y_i in y
                ])
                psnr.append(dinv.metric.PSNR()(x_hat, x))
                ssim.append(dinv.metric.SSIM()(x_hat, x))
                lpips.append(dinv.metric.LPIPS(device=device)(x_hat, x))

            psnr = torch.mean(torch.cat(psnr)).item()
            ssim = torch.mean(torch.cat(ssim)).item()
            lpips = torch.mean(torch.cat(lpips)).item()

            results = dict(PSNR=psnr, SSIM=ssim, LPIPS=lpips)
        else:
            results = dinv.test(
                model,
                test_dataloader,
                self.physics,
                metrics=[dinv.metric.PSNR(),
                         dinv.metric.SSIM(),
                         dinv.metric.LPIPS(device=device)],
                device=device
            )

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=results["PSNR"],
            ssim=results["SSIM"],
            lpips=results["LPIPS"]
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        model = dinv.optim.DPIR(sigma=0.03, device="cpu")
        return dict(model=model, model_name="TestSolver", device="cpu")

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(train_dataset=self.train_dataset, physics=self.physics)
