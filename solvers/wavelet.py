from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from deepinv.optim.dpir import get_DPIR_params
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim.prior import PnP
    from deepinv.models import WaveletDenoiser
    from benchmark_utils import constants
    from deepinv.optim.optimizers import optim_builder


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Wavelet'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {}

    sampling_strategy = 'run_once'

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ["ptwt"]

    def set_objective(self, train_dataloader):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.train_dataloader = train_dataloader

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        self.model = WaveletDenoiser(device=constants()["device"])

        #sigma_denoiser, stepsize, max_iter = get_DPIR_params(constants()["noise_level_img"])
        #params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        #early_stop = False

        #data_fidelity = L2()

        #prior = PnP(denoiser=WaveletDenoiser(device=constants()["device"]))

        #self.model = optim_builder(
        #    iteration="HQS",
        #    prior=prior,
        #    data_fidelity=data_fidelity,
        #    early_stop=early_stop,
        #    max_iter=max_iter,
        #    verbose=True,
        #    params_algo=params_algo,
        #)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.model)
