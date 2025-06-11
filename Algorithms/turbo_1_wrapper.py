import math
import os
import warnings

from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper
from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper
import ioh
from typing import Union, Optional, Dict, Tuple

from copy import deepcopy

from dataclasses import dataclass

from botorch.optim import optimize_acqf

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling, MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from .Turbo_utils.turbo_1_utils import TuRBOState, update_tr_length, get_best_index_for_batch, update_state

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")



# Define a class as a way of wrapping all the functions
class Turbo_1_Wrapper:
    def __init__(self, ioh_prob:Union[Design_IOH_Wrapper,
                                      Design_LP_IOH_Wrapper,
                                      ioh.iohcpp.problem.RealSingleObjective], 
                                      batch_size:int=4,
                                      max_cholesky_size:Optional[float]=float("inf")):
        self.ioh_prob = ioh_prob
        self.batch_size = batch_size
        self.max_cholesky_size = max_cholesky_size
    

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
        state:TuRBOState,
        model,  # GP model
        X:Tensor,  # Evaluated points on the domain [0, 1]^d
        Y:Tensor,  # Function values
        batch_size:int,
        n_candidates:int,  # Number of candidates for Thompson sampling
        sobol: SobolEngine,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        # Create the TR bounds
        best_ind = get_best_index_for_batch(Y=Y)
        x_center = X[best_ind, :].clone()
        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

        # Thompson Sampling w/ Constraints (SCBO)
        dim = X.shape[-1]
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points using  Max Posterior Sampling

        thompson_sampling = MaxPosteriorSampling(
            model=model,
            replacement=False,
        )
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

        return X_next
    
    def _get_fitted_model(self, X:Tensor, Y:Tensor):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
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
                 min_length:Optional[float]=0.5**7):
        # Generate initial data
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

        while n_evals<=total_budget:
            
            # Restart TuRBO if the trust region is too small
            self._restart()

            # Generate initial points
            self.train_X = self._get_initial_points(n_DoE, random_seed + n_loops)
            self.train_Y = torch.tensor([-1*self.eval_objective(x) for x in self.train_X], **tkwargs).unsqueeze(-1)

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

            # Initialize TuRBO state
            state = TuRBOState(self.dim, batch_size=self.batch_size,
                            length_min=min_length)

            # Note: We use 2000 candidates here to make the tutorial run faster.
            # SCBO actually uses min(5000, max(2000, 200 * dim)) candidate points by default.
            N_CANDIDATES = 2000 if not SMOKE_TEST else 4
            sobol = SobolEngine(self.dim, scramble=True, seed=random_seed)


            # Run until TuRBO converges
            while not state.restart_triggered and state.length > state.length_min:

                # Fit GP models for objective and constraints
                model = self._get_fitted_model(self.train_X, self.train_Y)
                #c1_model = self._get_fitted_model(self.train_X, self.train_C1)
                

                # Generate a batch of candidates
                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    X_next:Tensor = self._generate_batch(
                        state=state,
                        model=model,
                        X=self.train_X,
                        Y=self.train_Y,
                        batch_size=self.batch_size,
                        n_candidates=N_CANDIDATES,
                        sobol=sobol,
                    )
                
                # Update the function evaluations counter
                n_evals += X_next.shape[0]

                # Evaluate both the objective and constraints for the selected candidates
                Y_next = torch.tensor([-1*self.eval_objective(x) for x in X_next], 
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
                
                # C2_next = torch.tensor([eval_c2(x) for x in X_next], 
                #                        dtype=dtype, 
                #                        device=device).unsqueeze(-1)
                
                #C_next = torch.cat([C1_next, C2_next], dim=-1)
                C_next = C1_next

                # Update TuRBO state
                state = update_state(state=state, Y_next=Y_next)

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
    
                print(f"{len(self.train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
