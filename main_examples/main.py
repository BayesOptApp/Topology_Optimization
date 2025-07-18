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
from boundary_conditions import BoundaryConditionList, LineDirichletBC, PointNeumannBC, PointDirichletBC
import os
import ioh
import numpy as np

try:
    import cma
    from cma import fmin2
except:
    print("For this to run, install the cma library from Niko Hansen as `pip install cma`")
## ++++++++++++++++++++++++++++++++++++++++++++++++++++


## ++++++++++++++++++++++++++++++++++++++++++++++++++++
## Global Variables
RANDOM_SEED:int =22
RUN_E:int =  1007590
## ++++++++++++++++++++++++++++++++++++++++++++++++++++

r"""
This section is to show how to call an instance of the problem. In this case
you should call an instance from the class Design_IOH_Wrapper, which extends the definition
of the normal IOH `RealSingleObjective` problem instance. The parameters this object should receive are:

- nelx: `int`: This is the number of elements in x direction. 
- nely: `int`: This is the number of elements in y direction.
                Be careful to set these numbers to be high as this might make the runs much slower. We recommend to use the same ratios used by Guo et al. [1]
- nmmcsx: `int`: Number of Moving morphable components in x-direction (for initialization purposes) -> is functional so far, but this parameter is intended to be deprecated in the future.
- nmmcsy: `int`: Number of Moving morphable components in y-direction (for initialization purposes)
- symmetry_condition: `bool`: When activated this symmetry condition, then the topology is mirrored along the x-axis. 
- volfrac: `float`: A floating point value between 0 to 1, which determines the constraint of the total amount of available volume (area technically) the structure should occupy.
- use_sparse_matrices: `bool`: A handle to switch the solver use either full matrices and sparse matrices. This is intended to be deprecated. For performance reasons set it to `True`.
- plot_variables: `bool`: A trigger to plot the Von Mises Stress Contours and deformed structure from good designs. 
                          The threshold is hard coded to plot every design which has a target value less than 4. For upcoming versions, the threshold will be set by the user from this point on.
- E0: `float`: Just a parameter to represent the material presence of an element. This parameter was set for numerical studies, but just fix it to 1.00
- Emin `float`: Just a parameter to represent the abscence of material or "Ersatzmaterial" formulation. This parameter was also set free for numerical studies, but you can omit it or set it to
                1e-09.
- run_: `int`: An integer value representing the current run of the algorithm. This is just important to pointing to which folder will the plot be downloaded.


"""

boundary_conds = BoundaryConditionList()
# Add Dirichlet Boundary Condition
dirichlet_bc_1 = LineDirichletBC(start_point=(0.0, 0.0), end_point=(0.0, 1.0),blocked_dof=(1,2))
#dirichlet_bc_1 = PointDirichletBC(location=(0.0,0.0),blocked_dof=(1,2))
#irichlet_bc_2 = PointDirichletBC(location=(0.0,1.0),blocked_dof=(1,2))
# Add Neumann Boundary Condition
neumann_bc = PointNeumannBC(location=(1.0, 0.0),force_vector=(-0.0, -0.25/30 ))

boundary_conds.append(dirichlet_bc_1)
#boundary_conds.append(dirichlet_bc_2)
boundary_conds.append(neumann_bc)


# Generate Obj
ioh_prob:Design_IOH_Wrapper = Design_IOH_Wrapper(nelx=120,
                                                nely=60,                         
                                                nmmcsx=4,
                                                nmmcsy=2,
                                                symmetry_condition=False,
                                                volfrac=0.5,
                                                use_sparse_matrices=True,
                                                plot_variables=True,
                                                E0= 1.00,
                                                Emin= 1e-9,
                                                run_= RUN_E,
                                                boundary_conditions_list=boundary_conds,
                                                continuity_check_mode="discrete")

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
opts:cma.CMAOptions = {'bounds':[0,1],
                       'tolfun':1e-6,
                       'seed':RANDOM_SEED,
                       'verb_filenameprefix':os.path.join(logger.output_directory,"outcmaes/")
}

# Attach the logger to the problem
ioh_prob.attach_logger(logger)

# Run CMA-ES
fmin2(ioh_prob,x_init,0.25,restarts=0,bipop=True,options=opts)

ioh_prob.reset()
logger.close()

