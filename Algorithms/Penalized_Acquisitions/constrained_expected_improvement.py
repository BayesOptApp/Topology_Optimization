import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement

class PenalizedExpectedImprovement(AcquisitionFunction):
    def __init__(self, model: Model, constraint_model: Model, best_f: float, maximize: bool = True):
        super().__init__(model)
        #self.ei = ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        self.lei = LogExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        self.constraint_model = constraint_model

    @t_batch_mode_transform()
    def forward(self, X):
        # EI at X
        #ei = self.ei(X)
        lei = self.lei(X)
        # Probability of feasibility at X (assume constraint_model outputs mean, variance)
        posterior = self.constraint_model.posterior(X)
        mean = posterior.mean
        # For a <= 0 constraint, probability of feasibility is P(g(X) <= 0)
        # Assume single constraint for simplicity
        std = posterior.variance.sqrt().clamp_min(1e-9)
        prob_feas = torch.distributions.Normal(0, 1).cdf(-mean / std)
        # Penalized EI
        penalized_ei = torch.mul(lei, prob_feas.flatten())
        return penalized_ei