import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from functools import partial
import time

from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper
from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper
from Design_Examples.IOH_Wrappers.IOH_Wrapper_Instanced import Design_IOH_Wrapper_Instanced
import ioh

# def the_callback(*args, **kwargs):
#     """
#     Callback function to monitor the optimization process.
    
#     Args:
#     ------------------
#         - intermediate_result (OptimizeResult): Intermediate result of the optimization process.
#         - budget (int): Maximum number of function evaluations allowed.
#     """

#     budget = kwargs.get('budget', 1000)  # Default budget if not provided
#     intermediate_result = OptimizeResult(args[0][0],  # The first argument is the intermediate result

#     if intermediate_result['nfev'] > budget:
#         raise StopIteration(
#             "Maximum number of function evaluations exceeded.")

class MaxIterClassWrapper:
    def __init__(self, problem:ioh.iohcpp.problem.RealSingleObjective, max_iter):
        """
        Wrapper for random search algorithm with a maximum iteration limit.

        Args:
            problem: An IOH problem instance with .dimension and .evaluate(x) methods.
            max_iter: Maximum number of iterations allowed.
        """
        self.ioh_prob = problem
        self.max_iter = max_iter
    
    def __call__(self, x):
        """
        Evaluate the solution x.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            float: Fitness value of the solution.
        """

        if self.ioh_prob.state.evaluations >= self.max_iter:
            raise StopIteration("Maximum number of iterations reached.")
        
        return self.ioh_prob(x)
    

class DifferentialEvolutionWrapper:
    def __init__(self, problem):
        """
        Wrapper for random search algorithm.

        Args:
            problem: An IOH problem instance with .dimension and .evaluate(x) methods.
        """
        self.ioh_prob = problem
        self._starting_time = 0.0
    
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
        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper, Design_IOH_Wrapper_Instanced)):
            return [self.ioh_prob.bounds.lb[0], self.ioh_prob.bounds.ub[0]]
        elif isinstance(self.ioh_prob, ioh.iohcpp.problem.RealSingleObjective):
            return (-5, 5)
        else:
            raise ValueError("Unsupported problem type.")
    
    def complete_bounds(self):
        """
        Returns the complete bounds of the problem.
        """
        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper, Design_IOH_Wrapper_Instanced)):
            return [(self.bounds[0],self.bounds[1]) for _ in range(self.dim)]
        elif isinstance(self.ioh_prob, ioh.iohcpp.problem.RealSingleObjective):
            return [(-5,5) for _ in range(self.dim)]
        else:
            raise ValueError("Unsupported problem type.")
    
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
    
    def map_to_search_space(self, x):
        """
        Maps a solution to the search space of the problem.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            np.ndarray: Mapped solution.
        """
        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper, Design_IOH_Wrapper_Instanced)):
            return np.clip(x, self.ioh_prob.bounds.lb[0], self.ioh_prob.bounds.ub[0])
        elif isinstance(self.ioh_prob, ioh.iohcpp.problem.RealSingleObjective):
            return np.clip(x, -5, 5)
        else:
            raise ValueError("Unsupported problem type.")


    def __call__(self, 
                 budget:int,
                 popsize:int=10,
                 random_seed:int=43,
                 tol=0.01, 
                 mutation=(0.5, 1), 
                 recombination=0.7, 
                 disp=False, 
                 polish=False,
                 callback=None,
                 init='latinhypercube'):
        
        """
        Perform random search for a given evaluation budget.

        Args:
            budget (int): Number of evaluations allowed.

        Returns:
            dict: Best solution found and its fitness.
        """

        # Compute the population size based on the budget and dimension
    
        if budget < popsize:
            raise ValueError("Budget must be greater than population size.")
        
        maxiter = budget //  (popsize*self.dim) 

        # Start the timer
        self.starting_time = time.time()


        result:OptimizeResult = differential_evolution(
            func=MaxIterClassWrapper(self.ioh_prob, max_iter=budget),
            popsize=popsize,
            bounds=self.complete_bounds(),
            strategy='best1bin',
            maxiter=maxiter,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            seed=random_seed,
            disp=disp,
            polish=polish,
            init=init,
            callback=callback,
             )
        
        return result['x'], result['fun']