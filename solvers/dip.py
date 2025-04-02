from benchopt import BaseSolver, safe_import_context, config

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import deepinv as dinv
    import optuna


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'DIP'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {}

    sampling_strategy = 'run_once'

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ["torch", "optuna", "deepinv"]

    def set_objective(self, train_dataloader, physics):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.train_dataloader = train_dataloader
        self.physics = physics
        self.device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        def objective(trial):
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            iterations = trial.suggest_int('iterations', 50, 500, log=True)
            channels = trial.suggest_int('channels', 8, 128, log=True)

            model = self.get_model(lr, iterations, channels)
            
            psnr = torch.empty(0).to(self.device) 

            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                x_hat = torch.empty((0, 3, 256, 256)).to(self.device)

                for y_alone in y:
                    x_hat_alone = model(y_alone.unsqueeze(0), self.physics)
                    x_hat = torch.cat([x_hat, x_hat_alone])

                psnr = torch.cat([psnr, dinv.metric.PSNR()(x_hat, x)])

            psnr = psnr.mean().item()

            return psnr
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        best_trial = study.best_trial
        best_params = best_trial.params

        model = self.get_model(best_params['lr'], best_params['iterations'], best_params['channels'])

        return model

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.model, model_name="DIP", device=self.device)

    def get_model(self, lr, iterations, channels):
        in_size = [2, 2]  # size of the input to the decoder.
        backbone = dinv.models.ConvDecoder(
            img_shape=torch.Size([3, 256, 256]), in_size=in_size, channels=channels
        ).to(self.device)

        model = dinv.models.DeepImagePrior(
            backbone,
            learning_rate=lr,
            iterations=iterations,
            # verbose=True,
            input_size=[channels] + in_size,
        ).to(self.device)

        return model
