'''
Reinterpretation of code for Structural Optimisation
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

This is an example on how to call the problem and run 
an algorithm; in this case the algorithm will be CMA-ES from
Nikolaus Hansen.
'''

# Import the setup class

## ++++++++++++++++++++++++++++++++++++++++++++++++++++
from problems import get_problem
import os
import ioh
import numpy as np
from Algorithms.cma_es_wrapper import CMA_ES_Optimizer_Wrapper
## ++++++++++++++++++++++++++++++++++++++++++++++++++++


## ++++++++++++++++++++++++++++++++++++++++++++++++++++
## Global Variables
RANDOM_SEED:int =51
RUN_E:int =  144
## ++++++++++++++++++++++++++++++++++++++++++++++++++++



r"""
The next excerpt of code is just setting the IOH Logger. You may check the IOH Experimenter Wiki to see other ways to Log the corresponding results.
"""

triggers = [
    ioh.logger.trigger.Each(1),
    ioh.logger.trigger.OnImprovement()
]

logger = ioh.logger.Analyzer(
    root=os.getcwd(),                  # Store data in the current working directory
    folder_name=f"./Figures_Python/Run_{RUN_E}",       # in a folder named: './Figures_Python/Run_{run_e}'
    algorithm_name="CMA-ES",    # meta-data for the algorithm used to generate these results
    store_positions=True,               # store x-variables in the logged files
    triggers= triggers,

    additional_properties=[
        ioh.logger.property.CURRENTY,   # The constrained y-value, by default only the untransformed & unconstraint y
                                        # value is logged. 
        ioh.logger.property.RAWYBEST, # Store the raw-best
        ioh.logger.property.CURRENTBESTY, # Store the current best given the constraints
        ioh.logger.property.VIOLATION,  # The violation value
        ioh.logger.property.PENALTY,     # The applied penalty
    ]

)


# Get the problem 
ioh_prob = get_problem(
    problem_id=4,  # Problem ID for the structural optimisation problem; set to 1 for the cantilever beam problem
    dimension=45,  # Dimension of the problem
    run_number=RUN_E,  # Run number for the problem instance
    plot_stresses=True,  # Set to True if you want to plot the stresses
    instance=3,
)

# Track the number of Finite Element Evaluations (n_evals)
logger.watch(ioh_prob,"n_evals")

# Attach the logger to the problem
ioh_prob.attach_logger(logger)

# Set an instance of the CMA-ES optimizer
cma_es_optimizer = CMA_ES_Optimizer_Wrapper(
    ioh_problem=ioh_prob,  # The problem instance
    random_seed=RANDOM_SEED,  # Random seed for reproducibility
    sigma0=0.10,  # Initial standard deviation for the CMA-ES algorithm
)


#Run the optimization process
cma_es_optimizer(restarts=0, # Number of restarts for the CMA-ES algorithm
                 tolfun=1e-7, # Tolerance for the function value
                 max_f_evals=500 # Maximum number of function evaluations
                 )

ioh_prob.reset()
# Close the logger
ioh_prob.detach_logger()