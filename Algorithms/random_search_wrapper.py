import numpy as np

from IOH_Wrapper_LP import Design_LP_IOH_Wrapper
from IOH_Wrapper import Design_IOH_Wrapper
import ioh

class RandomSearchWrapper:
    def __init__(self, problem):
        """
        Wrapper for random search algorithm.

        Args:
            problem: An IOH problem instance with .dimension and .evaluate(x) methods.
        """
        self.ioh_prob = problem
    
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
    
    def map_to_search_space(self, x):
        """
        Maps a solution to the search space of the problem.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            np.ndarray: Mapped solution.
        """
        if isinstance(self.ioh_prob, (Design_LP_IOH_Wrapper, Design_IOH_Wrapper)):
            return np.clip(x, self.ioh_prob.bounds.lb[0], self.ioh_prob.bounds.ub[0])
        elif isinstance(self.ioh_prob, ioh.iohcpp.problem.RealSingleObjective):
            return np.clip(x, -5, 5)
        else:
            raise ValueError("Unsupported problem type.")


    def __call__(self, 
                 budget:int, 
                 random_seed:int=43):
        """
        Perform random search for a given evaluation budget.

        Args:
            budget (int): Number of evaluations allowed.

        Returns:
            dict: Best solution found and its fitness.
        """
        best_x = None
        best_f = None

        # Set a random generator from the given seed
        rng = np.random.default_rng(random_seed)

        for _ in range(budget):
            x = rng.uniform(self.bounds[0], self.bounds[1], size=self.dim)
            f = self.ioh_prob(self.map_to_search_space(x))

            if best_f is None:
                best_x = x.copy()
                best_f = f
            else:
                if self.is_maximization:
                    if f > best_f:
                        best_x = x.copy()
                        best_f = f
                else:
                    if f < best_f:
                        best_x = x.copy()
                        best_f = f

        return {"best_solution": best_x, "best_fitness": best_f}