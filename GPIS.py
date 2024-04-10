import torch
import gpytorch
import numpy as np


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, constant_mean_range, lengthscale_range, nu_i, outputscale_range):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            constant_constraint=gpytorch.constraints.Interval(constant_mean_range[0], constant_mean_range[1])
        )

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
            lengthscale_constraint=gpytorch.constraints.Interval(lengthscale_range[0], lengthscale_range[1]),
            nu=nu_i

            ),
        outputscale_constraint = gpytorch.constraints.Interval(outputscale_range[0], outputscale_range[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPISModel:
    def __init__(self, train_x, train_y, params):

        self.device = params.get('device', 'cpu')

        self.kernel_type = params.get('kernel_type', 'Matern')
        self.constant_mean_range = params.get('constant_mean_range', (0.01, 0.0101))
        self.lengthscale_range = params.get('lengthscale_range', (0.02, 0.0201))
        self.nu = params.get('nu', 1.5)
        self.outputscale_range = params.get('outputscale_range', (0.99, 1.0))
        self.likelihood_type = params.get('likelihood', 'GaussianLikelihood')
        self.optimizer_steps = params.get('optimizer_steps', 100)

        self.pre_sub = params.get('pre_sub', 1.0)

        self.load_data(train_x, train_y)

        self._setup_model(self.train_x, self.train_y)


    def _setup_model(self, train_x, train_y):

        if(self.likelihood_type == 'GaussianLikelihood'):
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model = ExactGPModel(train_x, train_y, self.likelihood, self.constant_mean_range, self.lengthscale_range, self.nu, self.outputscale_range)

        self.likelihood = self.likelihood.to(self.device)
        self.model = self.model.to(self.device)

    def load_data(self, train_x, train_y):

        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        self.train_y = torch.tensor(train_y, dtype=torch.float32).squeeze().to(self.device)

    def train(self):

        if self.model is None or self.likelihood is None:
            raise ValueError("Model and likelihood must be initialized before training.")

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.optimizer_steps):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            if(i % 10 == 0):
                print(f'Iter {i + 1}/{self.optimizer_steps} - Loss: {loss.item()}')

    def infer(self, test_points):

        if self.model is None or self.likelihood is None:
            raise ValueError("Model and likelihood must be initialized before inference.")

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor(test_points, dtype=torch.float32)
            test_x = test_x.to(self.device)
            observed_pred = self.likelihood(self.model(test_x))
        
        return observed_pred.mean+self.pre_sub, observed_pred.variance
    
    def get_device(self):
        return self.device
