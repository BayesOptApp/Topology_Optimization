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
from Algorithms.scbo_wrapper import SCBO_Wrapper
## ++++++++++++++++++++++++++++++++++++++++++++++++++++


## ++++++++++++++++++++++++++++++++++++++++++++++++++++
## Global Variables
RANDOM_SEED:int =49
RUN_E:int =  37
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
    algorithm_name="TuRBO_1",    # meta-data for the algorithm used to generate these results
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
    dimension=20,  # Dimension of the problem
    run_number=RUN_E,  # Run number for the problem instance
    plot_stresses=True,  # Set to True if you want to plot the stresses
    penalty_function=False  # Set to True if you want to use the penalty function
)

# Track the number of Finite Element Evaluations (n_evals)
logger.watch(ioh_prob,"n_evals")

# Attach the logger to the problem
ioh_prob.attach_logger(logger)

# Set an instance of the CMA-ES optimizer
optimizer = SCBO_Wrapper(
    ioh_prob=ioh_prob,  # The IOH problem instance
    batch_size=1  # Batch size for the optimizer
)


#Run the optimization process
optimizer(
    n_DoE=3*ioh_prob.problem_dimension,  # Number of Design of Experiments
    random_seed=RANDOM_SEED,  # Random seed for reproducibility
    total_budget=1000  # Budget for the optimization process
)

ioh_prob.reset()

# Close the logger
ioh_prob.detach_logger()