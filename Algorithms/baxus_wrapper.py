import math
import os
import warnings

import time

from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper
from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper
import ioh
from typing import Union, Optional, Dict, Tuple

from copy import deepcopy

from dataclasses import dataclass

from botorch.optim import optimize_acqf

import gpytorch
import torch
import botorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.exceptions import ModelFittingError
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize
from .BAxUS_utils.BAxUS_utils import (
    increase_embedding_and_observations,
    update_state,
    embedding_matrix,
    BaxusState)

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")



# Define a class as a way of wrapping all the functions
class BAxUS_Wrapper:
    def __init__(self, ioh_prob:Union[Design_IOH_Wrapper,
                                      Design_LP_IOH_Wrapper,
                                      ioh.iohcpp.problem.RealSingleObjective], 
                                      batch_size:int=4,
                                      max_cholesky_size:Optional[float]=float("inf")):
        self.ioh_prob = ioh_prob
        self.batch_size = batch_size
        self.max_cholesky_size = max_cholesky_size
        self._starting_time = 0.0
    

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
    
    @property
    def starting_time(self)-> float:
        """
        Get the starting time of the optimization.

        Returns:
            float: Starting time in seconds.
        """
        return self._starting_time
    
    @starting_time.setter
    def starting_time(self, value:float):
        """
        Set the starting time of the optimization.

        Args:
            value (float): Starting time in seconds.
        """
        self._starting_time = value
    
    @property
    def running_time(self)-> float:
        """
        Get the running time of the optimization.

        Returns:
            float: Running time in seconds.
        """
        return time.time() - self.starting_time
    
    def eval_objective(self, x:Tensor)->float:
        """
        Evaluates the objective function at a given point x.
        """

        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, 
                                      Design_IOH_Wrapper, 
                                      ioh.iohcpp.problem.RealSingleObjective)):
            
            # Map the input to the problem's domain (0, 1)^d
            bounds = self.bounds
            x = (x + 1) / 2 * (bounds[1] - bounds[0]) + bounds[0]
            # Evaluate the objective function
            return self.ioh_prob(x.cpu().numpy())
        else:
            raise ValueError("Unsupported problem type.")

    
    def get_initial_points(self, dim: int, n_pts: int, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = (
            2 * sobol.draw(n=n_pts).to(dtype=dtype, device=device) - 1
        )  # points have to be in [-1, 1]^d
        return X_init
    
    def _set_all_seeds(self, seed: int = 0):
        r"""
        Set all random seeds for reproducibility.
        
        Args
        ----------
        - seed : `Optional[int]`: The seed to set for all random number generators. Default is 0.
        
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_candidate(
            self,
            seed,
            state,
            model,  # GP model
            X:Tensor,  # Evaluated points on the domain [-1, 1]^d
            Y:Tensor,  # Function values
            n_candidates=None,  # Number of candidates for Thompson sampling
            num_restarts:int=10,
            raw_samples:int=512,
            acqf:str="ts",  # "ei" or "ts"
    ):
        assert acqf in ("ts", "ei")
        assert X.min() >= -1.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        #weights = model.covar_module.lengthscale.detach().view(-1)
        weights = model.covar_module.base_kernel.lengthscale.detach().view(-1)
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length, -1.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length, -1.0, 1.0)

        if acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True,seed=seed)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=self.batch_size)

        elif acqf == "ei":
            ei = LogExpectedImprovement(model, Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next
    
    def _get_fitted_model(self, X:Tensor, Y:Tensor):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        #covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        #    MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0))
        #)
        model = SingleTaskGP(
            X,
            Y,
            #covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            fit_gpytorch_mll(mll)

        return model
    
    def _restart(self):
        # Empty the training data
        self.train_X = torch.empty((0, self.dim), **tkwargs)
        self.train_Y = torch.empty((0, 1), **tkwargs)
        self.train_C1 = torch.empty((0, 1), **tkwargs)
    
    def __call__(self,
                 total_budget:int=1000,
                 random_seed:int=0, 
                 n_DoE:Optional[int]=10,
                 num_restarts:int=10,
                 raw_samples:int=4,
                 acquisition_function:str="ts",):
        
        # Set the random seed for reproducibility
        self._set_all_seeds(random_seed)
        
        # Generate initial data
        if n_DoE is None:
            n_DoE = self.dim * 3
        
        max_cholesky_size = float("inf")
        N_CANDIDATES = min(5000, max(2000, 200 * self.dim)) if not SMOKE_TEST else 4

        # Start the timer
        self.starting_time = time.time()

        state = BaxusState(dim=self.dim, eval_budget=total_budget-n_DoE)
        S = embedding_matrix(input_dim=state.dim, target_dim=state.d_init)

        # Count the number of evaluations
        n_evals = 0
        
        # Initialize the storage for all the data   
        self.C1_store = torch.empty((0, 1), **tkwargs)
        self.Y_store = torch.empty((0, 1), **tkwargs)

        # Initialize the training data
        self.X_baxus_target = self.get_initial_points(state.d_init, n_DoE, random_seed)
        self.X_baxus_input = self.X_baxus_target @ S

        n_evals += n_DoE

        self.X_store_input =self.X_baxus_input.clone()
        self.Y_baxus = torch.tensor([-self.eval_objective(x.detach()) for x in self.X_baxus_input], dtype=dtype, device=device).unsqueeze(-1)
        self.Y_store = torch.cat((self.Y_store, self.Y_baxus), dim=0)
        self.C1_store = torch.tensor(
            [self.ioh_prob.compute_actual_volume_excess(x.detach().cpu().numpy()) for x in self.X_baxus_input], dtype=dtype, device=device).unsqueeze(-1)



        # Disable input scaling checks as we normalize to [-1, 1]
        with botorch.settings.validate_input_scaling(False):
            while n_evals <= total_budget:  # Run until evaluation budget depleted
                # Fit a GP model
                std_Y = self.Y_baxus.std()
                if std_Y < 1e-12:
                    train_Y = self.Y_baxus - self.Y_baxus.mean()
                else:
                    train_Y = (self.Y_baxus - self.Y_baxus.mean()) / std_Y
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                
                # Covariance module with constrained signal variance and lengthscales
                covar_module = ScaleKernel(
                    # base_kernel=MaternKernel(
                    #     nu=2.5,
                    #     #ard_num_dims=self.dim,  # your input dimension here
                    #     lengthscale_constraint=Interval(0.005,10.0)
                    # ),
                    base_kernel=RBFKernel(
                        #ard_num_dims=state.target_dim, 
                        lengthscale_constraint=Interval(0.005, 10.0)
                    ),
                    #outputscale_constraint=Interval(0.05, 20.0)
                )

                model = SingleTaskGP(
                    self.X_baxus_target, 
                    train_Y, 
                    likelihood=likelihood,
                    covar_module=covar_module,
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Do the fitting and acquisition function optimization inside the Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    # Fit the model
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError as e:

                        print(f"Model fitting failed: {e.args}")
                        print("Reverting to Adam-based optimization.")                        
                        # Right after increasing the target dimensionality, the covariance matrix becomes indefinite
                        # In this case, the Cholesky decomposition might fail due to numerical instabilities
                        # In this case, we revert to Adam-based optimization
                        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

                        for _ in range(300):
                            optimizer.zero_grad()
                            output = model(self.X_baxus_target)
                            loss = -mll(output, train_Y.flatten())
                            loss.backward()
                            optimizer.step()

                    # Create a batch
                    X_next_target = self._create_candidate(
                        seed=random_seed,
                        state=state,
                        model=model,
                        X=self.X_baxus_target,
                        Y=train_Y,
                        n_candidates=N_CANDIDATES,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        acqf=acquisition_function,
                    )

                X_next_input = X_next_target @ S

                # Sum the number of evaluations
                n_evals += X_next_input.shape[0]
                Y_next = torch.tensor(
                    [-self.eval_objective(x.detach()) for x in X_next_input], dtype=dtype, device=device
                ).unsqueeze(-1)

                C1_next = torch.tensor(
                    [self.ioh_prob.compute_actual_volume_excess(x.detach().cpu().numpy()) for x in X_next_input], dtype=dtype, device=device
                ).unsqueeze(-1)

                # Update state
                state = update_state(state=state, Y_next=Y_next)

                # Append data
                self.X_baxus_input = torch.cat((self.X_baxus_input, X_next_input), dim=0)
                self.X_baxus_target = torch.cat((self.X_baxus_target, X_next_target), dim=0)

                self.Y_baxus = torch.cat((self.Y_baxus, Y_next), dim=0)
                
                
                self.C1_store = torch.cat((self.C1_store, C1_next), dim=0)

                self.X_store_input = torch.cat((self.X_store_input, X_next_input.reshape((-1,self.dim))), dim=0)
                self.Y_store = torch.cat((self.Y_store, Y_next), dim=0)


                # Print current status
                print(
                    f"iteration {len(self.X_baxus_input)}, d={len(self.X_baxus_target.T)})  Best value: {state.best_value:.3}, TR length: {state.length:.3}"
                )

                if n_evals > total_budget:
                    print(f"Total budget of {total_budget} evaluations reached.")
                    break

                if state.restart_triggered:
                    state.restart_triggered = False
                    print("increasing target space")
                    S, self.X_baxus_target = increase_embedding_and_observations(
                        S, self.X_baxus_target, state.new_bins_on_split
                    )
                    print(f"new dimensionality: {len(S)}")
                    state.target_dim = len(S)
                    state.length = state.length_init
                    state.failure_counter = 0
                    state.success_counter = 0
