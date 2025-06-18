import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective

class PenalizedExpectedImprovement(AcquisitionFunction):
    def __init__(self, model: Model, constraint_model: Model, best_f: float, 
                 maximize: bool = True,
):
        super().__init__(model)
        #self.ei = ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        #if analytic:
        self.lei = LogExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        #else:
        #    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]), seed=seed)
        #    if maximize:
        #        self.lei = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
        #    else:
        #        minimization_objective = GenericMCObjective(lambda samples, X=None: -samples[..., 0])
        #        self.lei = qLogExpectedImprovement(model=model, best_f=-best_f, sampler=sampler,
        #                                           objective=minimization_objective)
        self.constraint_model = constraint_model

    @t_batch_mode_transform()
    def forward(self, X):
        # EI at X
        #ei = self.ei(X)
        #lei:Tensor = self.lei(X)
        # Probability of feasibility at X (assume constraint_model outputs mean, variance)
        posterior = self.constraint_model.posterior(X)
        mean = posterior.mean
        # For a <= 0 constraint, probability of feasibility is P(g(X) <= 0)
        # Assume single constraint for simplicity
        std = posterior.variance.sqrt().clamp_min(1e-9)
        prob_feas = torch.distributions.Normal(0, 1).cdf(-mean / std)  # [N, q, 1]
        batch_prob_feas = prob_feas.squeeze(-1)  # [N]

        batch_log_prob_feas = torch.log(batch_prob_feas.clamp_min(1e-9))  # [N]

        lei = self.lei(X)  # [N]
        penalized_ei = lei.view(-1) + batch_log_prob_feas.view(-1)  # [N]
        return penalized_ei
    

class qPenalizedExpectedImprovement(MCAcquisitionFunction):
    def __init__(self, model: Model, constraint_model: Model, best_f: float,
                 seed:int = 0, sampler=None,
                 maximize: bool = True):
        
        
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]), seed=seed)
        
        super().__init__(model=model, sampler=sampler)
        
        #self.sampler = sampler
        
        if not maximize:
            objective = GenericMCObjective(lambda samples, X=None: -samples[..., 0])
            self.lei = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler,
                                               objective=objective)
        else:
            self.lei = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)

        self.constraint_model = constraint_model
        
            
            

        

        

    @t_batch_mode_transform()
    def forward(self, X):
        # EI at X
        lei = self.lei(X)
        # Probability of feasibility at X (assume constraint_model outputs mean, variance)
        posterior = self.constraint_model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.sqrt().clamp_min(1e-9)
        prob_feas = torch.distributions.Normal(0, 1).cdf(-mean / std)  # [N, q, 1]
        batch_prob_feas = prob_feas.min(dim=1).values.squeeze(-1)  # [N]

        penalized_ei = lei * batch_prob_feas
        return penalized_ei