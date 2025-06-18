import numpy as np
from typing import Optional, Dict, Tuple, Union
import time
try:
    from cma import fmin2
    from cma.evolution_strategy import CMAEvolutionStrategy
    from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper
    from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper
    import ioh

except:
    print("For this to run, install the cma library from Niko Hansen as `pip install cma`")

class CMA_ES_Optimizer_Wrapper:
    """
    Class wrapper for CMA-ES optimization of an IOH problem.
    """

    def __init__(self, 
                 ioh_problem:Union[ioh.iohcpp.problem.RealSingleObjective, 
                                   Design_LP_IOH_Wrapper, Design_IOH_Wrapper], 
                 x0=None, 
                 sigma0=0.25,
                 random_seed:int=0):
        """
        Initialize the optimizer.

        Args:
            ioh_problem: An IOH problem instance with .evaluate(x) and .dimension attributes.
            x0: Initial solution (list or np.ndarray). If None, uses zeros.
            sigma0: Initial standard deviation.
            options: Dictionary of CMA-ES options.
        """
        self.ioh_problem = ioh_problem
        self.random_seed = random_seed
        self.starting_time = 0.0

        # Extract the bounds from the problem
        lb = np.asanyarray(ioh_problem.bounds.lb).ravel()
        ub = np.asanyarray(ioh_problem.bounds.ub).ravel()

        if isinstance(ioh_problem, (Design_LP_IOH_Wrapper,Design_IOH_Wrapper)):
            # For Design_LP_IOH_Wrapper and Design_IOH_Wrapper, use the bounds from the problem
            self.bounds = [lb[0], ub[0]]
        elif isinstance(ioh_problem, ioh.iohcpp.problem.RealSingleObjective):
            # For RealSingleObjective, use the bounds from the problem
            self.bounds = (-5, 5)

        

        self.x0 = self.gen_xo(dimension=ioh_problem.meta_data.n_variables) if x0 is None else x0
        self.sigma0 = sigma0
        

        
        a =1 
        #self.es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, self.options)
    
    def gen_xo(self, dimension:int)->np.ndarray:
        """
        Generate a new solution using the CMA-ES strategy.

        Returns:
            np.ndarray: New solution.
        """

        # Set a random seed for reproducibility

        rng = np.random.default_rng(self.random_seed)

        # Generate a random solution within the bounds
        xo = rng.uniform(self.bounds[0], self.bounds[1], size=(dimension,))

        return xo.tolist()
    
    
    @property
    def running_time(self)-> float:
        """
        Get the running time of the optimization.

        Returns:
            float: Running time in seconds.
        """
        #return time.perf_counter() - self._starting_time
    
        if hasattr(self, "starting_time"):
            self.rt = time.perf_counter() - self.starting_time
            return self.rt
        return 0

        

    def __call__(self,
                 restarts:int=0,
                 bipop:bool=False,
                 tolfun:float=1e-6,
                 cma_active:bool=True,
                 max_f_evals:int=10000,
                 additional_options:Optional[Dict]=None,
                 verb_filenameprefix:str=None,
                 **kwargs):
        """
        Run CMA-ES optimization using cma.fmin2.

        Returns:
            (best_solution, best_fitness, es): Tuple with best solution, fitness, and CMAEvolutionStrategy instance.
        """

        self.options = {
            'bounds': self.bounds,
            'tolfun': tolfun,
            'seed': self.random_seed,
            'maxfevals': max_f_evals,
            'CMA_active': cma_active,
            'verb_filenameprefix':verb_filenameprefix
        }

        # Append the additional options if provided
        if additional_options is not None:
            for key, value in additional_options.items():
                if key not in self.options:
                    self.options[key] = value
                else:
                    if additional_options[key] != "bounds":
                        # If the key already exists and the value is not "bounds", overwrite it
                        # This is a warning to the user that they are overwriting an existing option
                        # with a new value.
                        print(f"Warning: Overwriting option {key} with value {value}")
                        self.options[key] = value
                    
                    else:
                        print(f"Warning: Option {key} is set to 'bounds', not overwriting.")

        # Set the starting time
        self.starting_time = time.perf_counter()

        best_solution, best_fitness = fmin2(objective_function=self.ioh_problem, 
                                                x0=self.x0, 
                                                sigma0=self.sigma0, 
                                                options=self.options,
                                                restarts=restarts,
                                                bipop=bipop
                                                )
        return best_solution, best_fitness


    @property
    def dimension(self):
        """
        Get the dimension of the problem.

        Returns:
            int: Dimension of the problem.
        """
        return self.ioh_problem.meta_data.n_variables
    
