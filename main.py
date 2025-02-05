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
from IOH_Wrappers import Design_IOH_Wrapper
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
RANDOM_SEED:int =98894
RUN_E:int = 90
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
- mode: `str`: This is a parameter to choose between two modes. The first mode, namely `TO` just refers to optimize the topology without fiber steering. 
               Whereas the mode `TO+LP` optimizes both the topology and lamination parameters. By activating the latter mode, the number of variables of the problem scales as
               D=5*nmmcs + 3, where 'nmmcs' stands for total number of Moving Morphable components (MMC). You can compute the total number of MMC by just computing nmmcs = nmmcsx * nmmcsy.
- symmetry_condition: `bool`: When activated this symmetry condition, then the topology is mirrored along the x-axis. 
- volfrac: `float`: A floating point value between 0 to 1, which determines the constraint of the total amount of available volume (area technically) the structure should occupy.
- use_sparse_matrices: `bool`: A handle to switch the solver use either full matrices and sparse matrices. This is intended to be deprecated. For performance reasons set it to `True`.
- VR: `float`: A floating point value between 0 to 1 denoting the volume ratio of fiber to matrix of the composite material.
- V3_1_init: `float`: A floating point value between -1 to 1 denoting the first lamination parameter
- V3_2_init: `float`: A floating point value between -1 to 1 denoting the second lamination parameter.
- plot_variables: `bool`: A trigger to plot the Von Mises Stress Contours and deformed structure from good designs. 
                          The threshold is hard coded to plot every design which has a target value less than 4. For upcoming versions, the threshold will be set by the user from this point on.
- E0: `float`: Just a parameter to represent the material presence of an element. This parameter was set for numerical studies, but just fix it to 1.00
- Emin `float`: Just a parameter to represent the abscence of material or "Ersatzmaterial" formulation. This parameter was also set free for numerical studies, but you can omit it or set it to
                1e-09.
- run_: `int`: An integer value representing the current run of the algorithm. This is just important to pointing to which folder will the plot be downloaded.


"""
# Generate Obj
ioh_prob:Design_IOH_Wrapper = Design_IOH_Wrapper(nelx=100,
                                                nely=50,                         
                                                #nmmcsx=10,
                                                nmmcsx=3,
                                                nmmcsy=2,
                                                mode="TO",
                                                symmetry_condition=True,
                                                volfrac=0.5,
                                                use_sparse_matrices=True,
                                                VR=0.5,
                                                V3_1_init=0, #-0.1,
                                                V3_2_init=0, #-0.4,
                                                plot_variables=False,
                                                E0= 1.00,
                                                Emin= 1e-9,
                                                run_= RUN_E)

r"""
The next excerpt of code is just setting the IOH Logger. You may check the IOH Experimenter Wiki to see other ways to Log the corresponding results.
"""

triggers = [
    ioh.logger.trigger.Each(1),
    ioh.logger.trigger.OnImprovement()
]

logger = ioh.logger.Analyzer(
    root=os.getcwd(),                  # Store data in the current working directory
    folder_name=f"./Figures_Python/Run_{run_e}",       # in a folder named: './Figures_Python/Run_{run_e}'
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

# Convert the first two constraints to a not
ioh_prob.convert_defined_constraint_to_type(0,4)
ioh_prob.convert_defined_constraint_to_type(1,4)

# Convert connectivity to a hidden
ioh_prob.convert_defined_constraint_to_type(2,3)

# Convert volume constraint soft
ioh_prob.convert_defined_constraint_to_type(3,3)


x_init = np.ravel(np.random.rand(1,ioh_prob.problem_dimension))


opts:cma.CMAOptions = {'bounds':[0,1],'tolfun':1e-6,'seed':RANDOM_SEED,'verb_filenameprefix':os.path.join(logger.output_directory,"outcmaes/")
}



ioh_prob.attach_logger(logger)


#run_experiment(problem=ioh_prob,algorithm=es,n_runs=1)
ioh_prob.reset()
logger.close()

