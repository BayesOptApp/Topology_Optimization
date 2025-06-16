import numpy as np
from typing import Optional, Dict, Tuple, Union
try:
    import pandas as pd
    from hebo.optimizers.hebo import HEBO
    from hebo.optimizers.general import GeneralBO
    from hebo.optimizers.hebo_embedding import HEBO_Embedding

    from hebo.acquisitions.acq import GeneralAcq
    from hebo.design_space.design_space import DesignSpace
    from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper
    from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper
    import ioh

except:
    print("For this to run, install the cma library from Niko Hansen as `pip install cma`")

class HEBO_Wrapper:
    """
    Class wrapper for CMA-ES optimization of an IOH problem.
    """

    def __init__(self, 
                 ioh_problem:Union[ioh.iohcpp.problem.RealSingleObjective, 
                                   Design_LP_IOH_Wrapper, Design_IOH_Wrapper],
                 batch_size:int = 1
                 ):
        """
        Initialize the optimizer.

        Args:
            ioh_problem: An IOH problem instance with .evaluate(x) and .dimension attributes.
            x0: Initial solution (list or np.ndarray). If None, uses zeros.
            sigma0: Initial standard deviation.
            options: Dictionary of CMA-ES options.
        """
        self.ioh_problem = ioh_problem
        self.batch_size = batch_size

        # Extract the bounds from the problem
        lb = np.asanyarray(ioh_problem.bounds.lb).ravel()
        ub = np.asanyarray(ioh_problem.bounds.ub).ravel()

        if isinstance(ioh_problem, (Design_LP_IOH_Wrapper,Design_IOH_Wrapper)):
            # For Design_LP_IOH_Wrapper and Design_IOH_Wrapper, use the bounds from the problem
            self.bounds = [lb[0], ub[0]]
        elif isinstance(ioh_problem, ioh.iohcpp.problem.RealSingleObjective):
            # For RealSingleObjective, use the bounds from the problem
            self.bounds = (-5, 5)

    

    def generate_search_space(self):
        """
        Generate a search space for the optimization problem.
        
        Parameters:
        - n_vars (int): Number of variables in the search space.
        
        Returns:
        - list: A list of dictionaries defining the search space.
        """
        space = []
        for i in range(self.dimension):
            space.append({
                'name': f'x{i+1}',
                'type': 'num',
                'lb': self.bounds[0],
                'ub': self.bounds[1]
            })
        
        return space
        
    
    # Define the objective function
    def objective(self,
                x:pd.DataFrame):
        r"""
        Objective function to be minimized.

        Parameters
        ------------------
        - x (`pd.Dataframe`): Input parameters for the optimization problem.

        Returns
        ------------------
        - `float`: Objective value.
        """

        # Convert the input DataFrame to a numpy array
        x_values = x.to_numpy()

        # Extract the values of the constraints
        #constraints_val = []

        #for ii in range(4):
        #    constraints_val.append(mmc_prob.constraints[ii](x_values))

        # Get the evaluation of the objective function
        #obj_val = mmc_prob.evaluate(x_values)

        obj_val = []
        for ii in range(x_values.shape[0]):
            # Set the parameters in the problem instance
            obj_val.append(self.ioh_problem(x_values[ii,:]))


        return np.asarray(obj_val)
    

    def __call__(self,
                 n_DOE:int=10,
                 random_seed:int=0,
                 budget:int = 1000,
                 **kwargs):
        """
        Run HEBO optimization.
        Args:
            **kwargs: Additional parameters for the optimization.

        Returns:
            Tuple: Best solution and its fitness.
        """

        self.random_seed = random_seed
        self.budget = budget

        # Define the search space
        space = self.generate_search_space()

        # Create a DesignSpace object
        design_space = DesignSpace().parse(space)
    
        # Initialize HEBO optimizer
        opt = HEBO(design_space,
                rand_sample=n_DOE,
                scramble_seed=self.random_seed,
        )

        for i in range(self.budget// self.batch_size):
            rec = opt.suggest(n_suggestions=self.batch_size)
            y = self.objective(rec)
            opt.observe(rec, y)

        # Get the best result
        best_idx = np.argmin(opt.y)
        best_x = opt.X.iloc[best_idx]
        best_y = opt.y[best_idx]

        print(f"Best parameters: {best_x}")
        print(f"Best objective: {best_y}")

        return best_x, best_y


    @property
    def dimension(self):
        """
        Get the dimension of the problem.

        Returns:
            int: Dimension of the problem.
        """
        return self.ioh_problem.meta_data.n_variables