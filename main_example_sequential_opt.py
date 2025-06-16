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
from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper
from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper

import os
import ioh
import numpy as np
from cma import fmin2
## ++++++++++++++++++++++++++++++++++++++++++++++++++++


## ++++++++++++++++++++++++++++++++++++++++++++++++++++
## Global Variables
RANDOM_SEED:int =70
RUN_E:int = 57
## ++++++++++++++++++++++++++++++++++++++++++++++++++++



# Generate Obj
ioh_prob:Design_IOH_Wrapper = Design_IOH_Wrapper(nelx=100,
                                                nely=50,                         
                                                nmmcsx=3,
                                                nmmcsy=2,
                                                symmetry_condition=True,
                                                volfrac=0.5,
                                                use_sparse_matrices=True,
                                                plot_variables=True,
                                                E0= 1.00,
                                                Emin= 1e-9,
                                                run_= RUN_E,
                                                continuity_check_mode="discrete",
                                                scalation_mode="unitary")

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

r"""
This is a relevant part of the code as this will control how the constraints influence the function evaluation.
The idea follows up from the constraint definition from IOH. Namely, this framework typifies the constraints in 4 classes:
    1. 1;NOT-> Type 1 or 'NOT' is that the constraint will not be evaluated.
    2. 2;HIDDEN-> Type 2 or 'HIDDEN', which means the constraint will be evaluated, but the target will be not penalized if the constraint condition is not fulfilled.
    3. 3;SOFT-> Type 3 or 'SOFT', which means the constraint will be evaluated, and the evaluation will result in a penalized function evaluation.
    4. 4;HARD-> Type 3 or 'SOFT', which means the constraint will be evaluated, and if not fulfilled, then the target function will not be computed and the resulting function evaluation just 
                corresponds to the penalty value.


In this topology framework, the problem has a container of 4 constraint functions. The list of these functions is the following:
1. Dirichlet Boundary Condition-> This function evaluates if there is at least material next to the clamped condition of the structure. If there is no material, then it computes a Minkowski
                                  distance (or max min norm) in a sense to check the minimum distance of a material element which is closest to the leftwise part of the domain.
2. Neumann Boundary Condition-> Similar to the first constraint function, ensures there is at least a material element next to the point load application node. And if not, then computes the 
                                Minkowski distance finding the least distance to the closest material element in the mesh.
3. Connectivity Condition-> A function, which checks if the design is connected. 
4. Volume Constraint-> Computes the fractional volume occupation (max(0,volume of the design/total volume)-volfrac) excess from the constraint. 

To run unbounded and/or search algorithms, we recommend to set the constraints 1 (Dirichlet) and 2 (Neumann) as type 4 such that the original target is not computed in such case. This is because
the dynamic matrices of the system are ill-conditioned. On the other hand we invite you to play with constraints 3 and 4 as you wish. The following examples is suited for CMA-ES.
"""
# Convert the first two constraints to a not
ioh_prob.convert_defined_constraint_to_type(0,2) # Dirichlet
ioh_prob.convert_defined_constraint_to_type(1,2) # Neumann

# Convert connectivity to a Hard constraint
ioh_prob.convert_defined_constraint_to_type(2,2) # Connectivity

# Convert volume constraint soft
ioh_prob.convert_defined_constraint_to_type(3,3) # Volume




# Set an initial starting point for CMA-ES
x_init = np.ravel(np.random.rand(1,ioh_prob.problem_dimension))

# Set the options for cma package `fmin` 
opts = {'bounds':[0,1],
                       'tolfun':1e-6,
                       'seed':RANDOM_SEED,
                       'verb_filenameprefix':os.path.join(logger.output_directory,"outcmaes","non_LP/"),
                       'maxfevals':150
}

# Attach the logger to the problem
ioh_prob.attach_logger(logger)

# Run CMA-ES
fmin2(ioh_prob,x_init,0.25,restarts=0,bipop=True,options=opts)

best_non_LP_solution = ioh_prob.state.current_best

print(f"Best non LP solution found: {best_non_LP_solution}")
ioh_prob.reset()

ioh_prob.detach_logger()


# Generate Obj
ioh_prob_LP:Design_IOH_Wrapper = Design_LP_IOH_Wrapper(nelx=ioh_prob.nelx,
                                                nely=ioh_prob.nely,                         
                                                nmmcsx=ioh_prob.nmmcsx,
                                                nmmcsy=ioh_prob.nmmcsy,
                                                symmetry_condition=ioh_prob.symmetry_condition_imposed,
                                                volfrac=ioh_prob.volfrac,
                                                use_sparse_matrices=ioh_prob.use_sparse_matrices,
                                                plot_variables=ioh_prob.plot_variables,
                                                E0= 1.00,
                                                Emin= 1e-9,
                                                run_= RUN_E,
                                                continuity_check_mode="discrete",
                                                scalation_mode="unitary",
                                                mode="LP",
                                                VR=0.5,
                                                V3_list=[0.5,0.5])

# Convert the first two constraints to a not
ioh_prob_LP.convert_defined_constraint_to_type(0,2) # Dirichlet
ioh_prob_LP.convert_defined_constraint_to_type(1,2) # Neumann

# Convert connectivity to a Hard constraint
ioh_prob_LP.convert_defined_constraint_to_type(2,2) # Connectivity

# Convert volume constraint soft
ioh_prob_LP.convert_defined_constraint_to_type(3,3) # Volume

print(np.asarray(best_non_LP_solution.x))

ioh_prob_LP.change_values_of_MMCs_from_unscaled_array(np.asarray(best_non_LP_solution.x),
                                                        repair_level=0)

# Set an initial starting point for CMA-ES
#x_init = np.hstack((best_non_LP_solution.x,np.ravel(np.random.rand(3,))))

x_init = np.ravel(np.random.rand(1,ioh_prob_LP.problem_dimension))

# Set the options for cma package `fmin` 
opts = {'bounds':[0,1],
                       'tolfun':1e-6,
                       'seed':RANDOM_SEED,
                       'verb_filenameprefix':os.path.join(logger.output_directory,"outcmaes","LP/"),
                       'maxfevals':35
}

# Attach the logger to the problem
ioh_prob_LP.attach_logger(logger)

# Run CMA-ES
fmin2(ioh_prob_LP,x_init,0.25,restarts=0,bipop=True,options=opts)

best_non_LP_solution = ioh_prob_LP.state.current_best

print(f"Best LP solution found: {best_non_LP_solution}")
ioh_prob_LP.reset()

ioh_prob_LP.detach_logger()




