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
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.exceptions.errors import ModelFittingError
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
class Turbo_M_Wrapper:
    def __init__(self, ioh_prob:Union[Design_IOH_Wrapper,
                                      Design_LP_IOH_Wrapper,
                                      ioh.iohcpp.problem.RealSingleObjective],
                                      n_trust_regions:int,
                                      batch_size:int=4,
                                      max_cholesky_size:Optional[float]=float("inf")):
        self.ioh_prob = ioh_prob
        self.batch_size = batch_size
        self.max_cholesky_size = max_cholesky_size
        self.n_trust_regions = n_trust_regions

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
            return self.ioh_prob(x.detach().cpu().numpy())
        else:
            raise ValueError("Unsupported problem type.")
    
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
            MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.001, 10.0))
        )
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        try:
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                fit_gpytorch_mll(mll)
        except ModelFittingError as e:
            print("Fitting failed, retrying with perturbed init...")
            model.covar_module.base_kernel.lengthscale = model.covar_module.base_kernel.lengthscale + 0.1 * torch.rand_like(model.covar_module.base_kernel.lengthscale)
            # Jitter the data
            Y_jittered = Y + 1e-6 * torch.randn_like(Y)
            model.set_train_data(X, Y_jittered.flatten(), strict=False)
            # Retry fitting
            fit_gpytorch_mll(mll)  # Retry


        return model
    
    def _restart(self):
        # Empty the training data
        self.train_X = torch.empty((0, self.dim), **tkwargs)
        self.train_Y = torch.empty((0, 1), **tkwargs)
        self.train_C1 = torch.empty((0, 1), **tkwargs)
        self.idx = torch.empty((0, 1), device="cpu")
    
    def __call__(self,
                 total_budget:int=1000,
                 random_seed:int=0, 
                 n_DoE:Optional[int]=10,
                 min_length:Optional[float]=0.5**7):
        
        
        # Set the random seed for reproducibility
        self._set_all_seeds(seed=random_seed)


        if n_DoE is None:
            n_DoE = self.dim * 3

        self.C1_store = torch.empty((0, 1), **tkwargs)
        self.X_store = torch.empty((0, self.dim), **tkwargs)
        self.Y_store = torch.empty((0, 1), **tkwargs)
        self.idx_store = torch.empty((0, 1), device="cpu")

        n_evals = 0
        n_loops = 0

        # Setup state and data for each trust region
        states = [TuRBOState(self.dim, batch_size=self.batch_size, length_min=min_length) for _ in range(self.n_trust_regions)]
        train_X = [self._get_initial_points(n_DoE, random_seed + i) for i in range(self.n_trust_regions)]
        train_Y = [torch.tensor([-1*self.eval_objective(x) for x in X], **tkwargs).unsqueeze(-1) for X in train_X]
        train_C1 = [torch.empty((0, 1), **tkwargs) for _ in range(self.n_trust_regions)]
        sobols = [SobolEngine(self.dim, scramble=True, seed=random_seed + i) for i in range(self.n_trust_regions)]

        # Set the timer
        self.starting_time = time.time()

        # Initialize the training data with the initial points
        for i in range(self.n_trust_regions):
            self.X_store = torch.cat((self.X_store, train_X[i]), dim=0)
            self.Y_store = torch.cat((self.Y_store, train_Y[i]), dim=0)
            self.idx_store = torch.cat((self.idx_store, i * torch.ones((n_DoE, 1, ), dtype=int)), dim=0)

            # Optional constraint collection (not active)
            # if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper)):
            #     C1_holder = []
            #     for x in train_X[i]:
            #         x_np = x.detach().cpu().numpy()
            #         C1_holder.append(self.ioh_prob.compute_actual_volume_excess(x_np))
            #     train_C1[i] = torch.tensor(C1_holder, **tkwargs).unsqueeze(-1)
            #     self.C1_store = torch.cat((self.C1_store, train_C1[i]), dim=0)

        n_evals += self.n_trust_regions * n_DoE

        N_CANDIDATES = 2000 if not SMOKE_TEST else 4

        while n_evals < total_budget:
            for i in range(self.n_trust_regions):
                if states[i].restart_triggered or states[i].length <= states[i].length_min:
                    # Restart the trust region
                    states[i] = TuRBOState(self.dim, batch_size=self.batch_size, length_min=min_length)

                    train_X[i] = self._get_initial_points(n_DoE, random_seed + i + n_loops * self.n_trust_regions)
                    train_Y[i] = torch.tensor([-1*self.eval_objective(x) for x in train_X[i]], **tkwargs).unsqueeze(-1)

                    n_evals += train_X[i].shape[0]
                    self.X_store = torch.cat((self.X_store, train_X[i]), dim=0)
                    self.Y_store = torch.cat((self.Y_store, train_Y[i]), dim=0)
                    self.idx_store = torch.cat((self.idx_store, i * torch.ones((train_X[i].shape[0], 1), dtype=int)))
                    continue

                model = self._get_fitted_model(train_X[i], train_Y[i])

                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    X_next = self._generate_batch(
                        state=states[i],
                        model=model,
                        X=train_X[i],
                        Y=train_Y[i],
                        batch_size=self.batch_size,
                        n_candidates=N_CANDIDATES,
                        sobol=sobols[i],
                    )

                Y_next = torch.tensor([-1*self.eval_objective(x) for x in X_next], **tkwargs).unsqueeze(-1)

                # Optional constraint evaluation (not active)
                # C1_holder = []
                # for x in X_next:
                #     x_np = x.detach().cpu().numpy()
                #     C1_holder.append(self.ioh_prob.compute_actual_volume_excess(x_np))
                # C1_next = torch.tensor(C1_holder, **tkwargs).unsqueeze(-1)

                train_X[i] = torch.cat((train_X[i], X_next), dim=0)
                train_Y[i] = torch.cat((train_Y[i], Y_next), dim=0)
                # train_C1[i] = torch.cat((train_C1[i], C1_next), dim=0)

                self.X_store = torch.cat((self.X_store, X_next), dim=0)
                self.Y_store = torch.cat((self.Y_store, Y_next), dim=0)
                self.idx_store = torch.cat((self.idx_store, i * torch.ones((X_next.shape[0], 1), dtype=int)), dim=0)
                # self.C1_store = torch.cat((self.C1_store, C1_next), dim=0)

                states[i] = update_state(state=states[i], Y_next=Y_next)

                print(f"TR-{i} | {train_X[i].shape[0]}) Best value: {states[i].best_value:.2e}, TR length: {states[i].length:.2e}")

                n_evals += X_next.shape[0]

                if n_evals > total_budget:
                    print(f"Total budget of {total_budget} evaluations reached.")
                    break
