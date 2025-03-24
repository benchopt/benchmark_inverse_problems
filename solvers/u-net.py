from benchopt import BaseSolver, safe_import_context, config

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import deepinv as dinv
    from sklearn.model_selection import GridSearchCV


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'UNet'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {}

    sampling_strategy = 'run_once'

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ["scikit-learn"]

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

        #param_grid = {
        #    'lr': [1e-5, 1e-2],
        #    'epochs': [4, 10]
        #}

        #param_combinations = list(product(*param_grid.values()))

        #best_loss = float('inf')
        #best_params = None

        model = dinv.models.UNet(
            in_channels=3, out_channels=3, scales=3, batch_norm=False
        ).to(self.device)

        verbose = True  # print training information
        wandb_vis = False  # plot curves and images in Weight&Bias

        #for params in param_combinations:
        #    lr, epochs = params

        epochs = 4  # choose training epochs
        learning_rate = 5e-4

        # choose training losses
        losses = dinv.loss.SupLoss(metric=dinv.metric.MSE())

        # choose optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))
        trainer = dinv.Trainer(
            model,
            device=self.device,
            verbose=verbose,
            save_path=config.get_data_path(""),
            wandb_vis=wandb_vis,
            physics=self.physics,
            epochs=epochs,
            scheduler=scheduler,
            losses=losses,
            optimizer=optimizer,
            show_progress_bar=True,  # disable progress bar for better vis in sphinx gallery.
            train_dataloader=self.train_dataloader,
        )

        self.model = trainer.train()

        self.model.eval()

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.model, model_name="U-Net", device=self.device)
