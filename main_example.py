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
from Algorithms.random_search_wrapper import RandomSearchWrapper
from Algorithms.turbo_1_wrapper import Turbo_1_Wrapper
## ++++++++++++++++++++++++++++++++++++++++++++++++++++


## ++++++++++++++++++++++++++++++++++++++++++++++++++++
## Global Variables
RANDOM_SEED:int = 7375
RUN_E:int =  5135
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
    problem_id=3,  # Problem ID for the structural optimisation problem; set to 1 for the cantilever beam problem
    dimension=40,  # Dimension of the problem
    run_number=RUN_E,  # Run number for the problem instance
    plot_stresses=False,  # Set to True if you want to plot the stresses
    plot_topology=True, # Set to True if you want to plot the topology (black and white display)
    instance=0,
)

# Write the ranges to same directory as log
ioh_prob.write_ranges_to_file(os.path.join(logger.output_directory, "ranges.txt"))

# Track the number of Finite Element Evaluations (n_evals)
logger.watch(ioh_prob,"n_evals")

# Attach the logger to the problem
ioh_prob.attach_logger(logger)

# Set an instance of the CMA-ES optimizer
cma_es_optimizer = CMA_ES_Optimizer_Wrapper(
    ioh_problem=ioh_prob,  # The problem instance
    random_seed=RANDOM_SEED,  # Random seed for reproducibility
    sigma0=0.2,  # Initial standard deviation for the CMA-ES algorithm
)



# rs_optimizer = RandomSearchWrapper(
#     problem=ioh_prob  # The problem instance
# )

# tr_opt = Turbo_1_Wrapper(
#     ioh_prob=ioh_prob,  # The problem instance
#     batch_size=1,
# )

#Run the optimization process
cma_es_optimizer(budget=5000, random_seed=RANDOM_SEED,additional_options = {"CMA_elitist":False})
#rs_optimizer(budget=5000,random_seed=RANDOM_SEED)

#tr_opt(total_budget=5000, random_seed=RANDOM_SEED, n_DoE=3*10)

ioh_prob.reset()
# Close the logger
ioh_prob.detach_logger()