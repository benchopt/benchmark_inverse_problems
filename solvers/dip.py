from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import deepinv as dinv
    import optuna


class Solver(BaseSolver):
    name = 'DIP'

    parameters = {}

    sampling_strategy = 'run_once'

    requirements = ["optuna"]

    def set_objective(self, train_dataset, physics):
        self.train_dataset = train_dataset
        batch_size = 32
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        self.device = (
            dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
        )
        self.physics = physics.to(self.device)

    def run(self, n_iter):
        def objective(trial):
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            iterations = trial.suggest_int('iterations', 50, 500, log=True)

            # TODO: Remove
            iterations = 5

            model = self.get_model(lr, iterations)

            psnr = []

            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                x_hat = torch.cat([
                    model(y_i[None], self.physics) for y_i in y
                ])
                psnr.append(dinv.metric.PSNR()(x_hat, x))

            psnr = torch.mean(torch.cat(psnr)).item()

            return psnr

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=1)

        best_trial = study.best_trial
        best_params = best_trial.params

        # TODO : replace 5 by best_params['iterations'])
        self.model = self.get_model(best_params['lr'], 5)

    def get_result(self):
        return dict(model=self.model, model_name="DIP", device=self.device)

    def get_model(self, lr, iterations):
        in_size = [4, 4]  # size of the input to the decoder.
        channels = 64
        backbone = dinv.models.ConvDecoder(
            img_shape=self.train_dataset[0][0].shape,
            in_size=in_size,
            channels=channels
        ).to(self.device)

        model = dinv.models.DeepImagePrior(
            backbone,
            learning_rate=lr,
            iterations=iterations,
            verbose=True,
            input_size=[channels] + in_size,
        ).to(self.device)

        return model
