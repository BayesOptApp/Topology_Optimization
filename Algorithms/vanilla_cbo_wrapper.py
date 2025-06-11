from typing import Union

from IOH_Wrapper import Design_IOH_Wrapper
from IOH_Wrapper_LP import Design_LP_IOH_Wrapper
import ioh
from typing import Union, Optional, Dict, Tuple

import torch
from torch.quasirandom import SobolEngine
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.analytic import ConstrainedExpectedImprovement, LogConstrainedExpectedImprovement
from botorch.optim import optimize_acqf

from botorch.fit import fit_gpytorch_mll


import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from Algorithms.Penalized_Acquisitions.constrained_expected_improvement import PenalizedExpectedImprovement
#from   Penalized_Acquisitions import PenalizedExpectedImprovement

import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

class VanillaCBO:
    def __init__(self, ioh_prob:Union[Design_IOH_Wrapper,
                                      Design_LP_IOH_Wrapper,
                                      ioh.iohcpp.problem.RealSingleObjective], 
                                      batch_size:int=1,
                                      max_cholesky_size:int=1000,
                                      num_restarts:int=5):
        self.ioh_prob = ioh_prob
        #self.batch_size = batch_size
        self.batch_size = 1
        self.max_cholesky_size = max_cholesky_size
        self.num_restarts = num_restarts
    
    @property
    def ioh_prob(self):
        """
        Returns the IOH problem instance.
        """
        return self._ioh_prob
    
    @ioh_prob.setter
    def ioh_prob(self, ioh_prob:Union[Design_IOH_Wrapper,
                                        Design_LP_IOH_Wrapper,
                                        ioh.iohcpp.problem.RealSingleObjective]):
        """
        Sets the IOH problem instance.
        """
        if not isinstance(ioh_prob, (Design_IOH_Wrapper, Design_LP_IOH_Wrapper, ioh.iohcpp.problem.RealSingleObjective)):
            raise ValueError("ioh_prob must be an instance of Design_IOH_Wrapper," +  
                             + " Design_LP_IOH_Wrapper or RealSingleObjective.")
        self._ioh_prob = ioh_prob
    
    @property
    def batch_size(self):
        """
        Returns the batch size.
        """
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size:int):
        """
        Sets the batch size.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        self._batch_size = batch_size

    @property
    def dim(self):
        """
        Returns the dimension of the problem.
        """
        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper)):
            return self.ioh_prob.meta_data.n_variables
        elif isinstance(self.ioh_prob, ioh.iohcpp.problem.RealSingleObjective):
            return self.ioh_prob.meta_data.n_variables
        else:
            raise ValueError("Unsupported problem type.")
    
    @property
    def bounds(self):
        """
        Returns the bounds of the problem.
        """
        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper)):
            return [self.ioh_prob.bounds.lb[0], self.ioh_prob.bounds.ub[0]]
        elif isinstance(self.ioh_prob, ioh.iohcpp.problem.RealSingleObjective):
            return (-5, 5)
        else:
            raise ValueError("Unsupported problem type.")
    
    def bounds_for_torch(self)->Tensor:
        """
        Returns the bounds of the problem as a tensor.
        """
        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper)):
            tens = torch.tensor([self.ioh_prob.bounds.lb[0], self.ioh_prob.bounds.ub[0]], **tkwargs)
        elif isinstance(self.ioh_prob, ioh.iohcpp.problem.RealSingleObjective):
            tens = torch.tensor([-5, 5], **tkwargs)
        else:
            raise ValueError("Unsupported problem type.")
        
        # Convert to 2D tensor such that the first row is the lb and the second row is the ub and multiplied by the number of dimensions
        tens = tens.reshape(2, 1)
        tens = tens.repeat(1, self.dim)
        return tens.reshape(2, self.dim)
    
    @property
    def is_maximization(self)->bool:
        """
        Returns True if the problem is a maximization problem, False otherwise.
        """
        if self.ioh_prob.meta_data.optimization_type == ioh.OptimizationType.MAX:
            return True
        elif self.ioh_prob.meta_data.optimization_type == ioh.OptimizationType.MIN:
            return False
        else:
            raise ValueError("Unsupported problem type.")
    
    def eval_objective(self, x:Tensor)->float:
        """
        Evaluates the objective function at a given point x.
        """

        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, 
                                      Design_IOH_Wrapper, 
                                      ioh.iohcpp.problem.RealSingleObjective)):
            return self.ioh_prob(x.detach().cpu().numpy())
        else:
            raise ValueError("Unsupported problem type.")
        
    def _get_initial_points(self,
                            n_pts:int, 
                            seed:int=0):
        
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
        return X_init


    def _generate_batch(
        self,
        model,  # GP model
        Y:Tensor,  # Function values
        C:Tensor,  # Constraint values
        #batch_size:int,
        constraint_model, #GP Model for constraints
        #sobol: SobolEngine,
    ):
        
        # Get the best feasible point
        best_index = self.get_best_index_for_batch(Y, C)


        # Acquisition function: Expected Improvement
        #EI = qExpectedImprovement(model, best_f=y_feas.min(), maximize=False)
        cEI = PenalizedExpectedImprovement(
            model=model,
            constraint_model=constraint_model,
            best_f=Y[best_index].item(),
            maximize=self.is_maximization,
        )

        # Optimize acquisition function
        candidate, _ = optimize_acqf(
            cEI,
            bounds=self.bounds_for_torch(),
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        x = candidate.detach().squeeze(0)
        # Check constraint, if not feasible, fallback to random
        #if not self.constraint_func(x.cpu().numpy()):
        #    x = (self.bounds[:, 0] + (self.bounds[:, 1] - self.bounds[:, 0]) * torch.rand(self.bounds.shape[0], dtype=self.dtype, device=self.device))
        return x.reshape((self.batch_size,-1))


    
    def _get_fitted_model(self, X:Tensor, Y:Tensor):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior
            MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0))
        )

        # Create the model
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # Fit the model
        fit_gpytorch_mll(mll)

        return model
    
    def get_best_index_for_batch(self,Y: Tensor, C: Tensor)-> int:
        """Return the index for the best point."""
        is_feas = (C <= 0).all(dim=-1)
        if is_feas.any():  # Choose best feasible candidate
            score = Y.clone()
            score[~is_feas] = -float("inf")
            if self.is_maximization:
                return score.argmax()
            else:
                return score.argmin()
            return score.argmin()
        return C.clamp(min=0).sum(dim=-1).argmin()
    
    def _restart(self):
        # Empty the training data
        self.train_X = torch.empty((0, self.dim), **tkwargs)
        self.train_Y = torch.empty((0, 1), **tkwargs)
        self.train_C1 = torch.empty((0, 1), **tkwargs)
    
    def __call__(self, 
                 total_budget:int=1000,
                 random_seed:int=0, 
                 n_DoE:Optional[int]=10,
                 **kwargs):
        r"""
        Performs the optimization process.
        """

        if n_DoE is None:
            n_DoE = self.dim * 3
        
        # Initialize the storage for all the data   
        self.C1_store = torch.empty((0, 1), **tkwargs)
        self.X_store = torch.empty((0, self.dim), **tkwargs)
        self.Y_store = torch.empty((0, 1), **tkwargs)

        # Count the number of evaluations
        n_evals = 0

        # Count number of loops
        n_loops = 0
            
        # Restart (if needed)
        self._restart()

        # Generate initial points
        self.train_X = self._get_initial_points(n_DoE, random_seed + n_loops)
        self.train_Y = torch.tensor([self.eval_objective(x) for x in self.train_X], **tkwargs).unsqueeze(-1)

        n_loops += 1

        C1_holder = []
        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper)):
            # Evaluate the indexed 3 constraint (volume)
            for x in self.train_X:
                # Convert the tensor to a numpy array
                x_np = x.detach().cpu().numpy()
                # Evaluate the constraint function
                C1_holder.append(self.ioh_prob.compute_actual_volume_excess(x_np))

            self.train_C1 = torch.tensor(C1_holder, **tkwargs).unsqueeze(-1)
            self.C1_store = torch.cat((self.C1_store, self.train_C1), dim=0)

        self.X_store = torch.cat((self.X_store, self.train_X), dim=0)
        self.Y_store = torch.cat((self.Y_store, self.train_Y), dim=0)
        

        # Initialize a counter for the number of function evaluations
        n_evals += self.train_X.shape[0]


        # Note: We use 2000 candidates here to make the tutorial run faster.
        # SCBO actually uses min(5000, max(2000, 200 * dim)) candidate points by default.
        #N_CANDIDATES = 2000 if not SMOKE_TEST else 4


        # Run until TuRBO converges
        while n_evals<=total_budget:

            # Fit GP models for objective and constraints
            model = self._get_fitted_model(self.train_X, self.train_Y)
            c1_model = self._get_fitted_model(self.train_X, self.train_C1)
            

            # Generate a batch of candidates
            #with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_next:Tensor = self._generate_batch(
                model=model,
                Y=self.train_Y,
                C=self.train_C1,
                #batch_size=self.batch_size,
                constraint_model=ModelListGP(c1_model),
            )
            
            # Update the function evaluations counter
            n_evals += X_next.shape[0]

            # Evaluate both the objective and constraints for the selected candidates
            Y_next = torch.tensor([self.eval_objective(x) for x in X_next], 
                                dtype=dtype, 
                                device=device).unsqueeze(-1)
            
            C1_holder = []
            for x in X_next:
                # Convert the tensor to a numpy array
                x_np = x.detach().cpu().numpy()
                # Evaluate the constraint function
                C1_holder.append(self.ioh_prob.compute_actual_volume_excess(x_np))

            C1_next = torch.tensor(C1_holder, 
                                    dtype=dtype, 
                                    device=device).unsqueeze(-1)



            # Append data. Note that we append all data, even points that violate
            # the constraints. This is so our constraint models can learn more
            # about the constraint functions and gain confidence in where violations occur.
            self.train_X = torch.cat((self.train_X, X_next), dim=0)
            self.train_Y = torch.cat((self.train_Y, Y_next), dim=0)
            self.train_C1 = torch.cat((self.train_C1, C1_next), dim=0)
            #C2 = torch.cat((C2, C2_next), dim=0)

            # Concatenate with the store
            self.X_store = torch.cat((self.X_store, X_next), dim=0)
            self.Y_store = torch.cat((self.Y_store, Y_next), dim=0)
            self.C1_store = torch.cat((self.C1_store, C1_next), dim=0)
    

            # Print current status. Note that state.best_value is always the best
            # objective value found so far which meets the constraints, or in the case
            # that no points have been found yet which meet the constraints, it is the
            # objective value of the point with the minimum constraint violation.
            # if (state.best_constraint_values <= 0).all():
            #     print(f"{len(self.train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
            # else:
            #     violation = state.best_constraint_values.clamp(min=0).sum()
            #     print(
            #         f"{len(self.train_X)}) No feasible point yet! Smallest total violation: "
            #         f"{violation:.2e}, TR length: {state.length:.2e}"
            #     )


        
