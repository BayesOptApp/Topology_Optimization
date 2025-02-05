'''
Reinterpretation of code for Structural Optimisation
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

'''

RANDOM_SEED:int =98894

# Import the setup class
from IOH_Wrappers import Design_IOH_Wrapper
import os
import ioh
import numpy as np
import cma
from cma import fmin2

np.random.seed(RANDOM_SEED)
run_e:int = 90
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
                                                plot_variables=True,
                                                E0= 1.00,
                                                Emin= 1e-9,
                                                run_= run_e)

# setupp.nelx = 100
# setupp.nely = 50
# setupp.nmmcsx = 3
# setupp.nmmcsy = 2
# setupp.plotVariables = True
# setupp.symm = True
# setupp.sparse_matrices = True
# setupp.cost_function= "compliance"
# setupp.scalation_mode = "unitary" #"Bujny" #"unitary"

#ioh_prob.scalation_mode = "Bujny"

# iterr_ = 1

triggers = [
    ioh.logger.trigger.Each(1),
    ioh.logger.trigger.OnImprovement()
]

logger = ioh.logger.Analyzer(
    root=os.getcwd(),                  # Store data in the current working directory
    folder_name=f"./Figures_Python/Run_{run_e}",       # in a folder named: 'my-experiment'
    algorithm_name="random-search",    # meta-data for the algorithm used to generate these results
    store_positions=True,               # store x-variables in the logged files
    triggers= triggers,

    additional_properties=[
        ioh.logger.property.CURRENTY,   # The constrained y-value, by default only the untransformed & unconstraint y
                                        # value is logged. 
        ioh.logger.property.RAWYBEST, # Store the raw-best
        ioh.logger.property.CURRENTBESTY, # Store the current best given the constraints
        ioh.logger.property.VIOLATION,  # The violation value
        ioh.logger.property.PENALTY,     # The applied penalty
        #ioh.logger.property.Property(ioh_prob.multi_fidelity,"multi_fidelity")
    ]

)

# Convert the first two constraints to a not
ioh_prob.convert_defined_constraint_to_type(0,4)
ioh_prob.convert_defined_constraint_to_type(1,4)

# Convert connectivity to a hidden
ioh_prob.convert_defined_constraint_to_type(2,3)

# Convert volume constraint soft
ioh_prob.convert_defined_constraint_to_type(3,3)

#ioh_prob.attach_logger(logger)

# class RandomSearch:
#     'Simple random search algorithm'
#     def __init__(self, n: int, length: float = 0.0):
#         self.n: int = n
#         self.length: float = length
        
#     def __call__(self, problem: ioh.problem.RealSingleObjective) -> None:
#         'Evaluate the problem n times with a randomly generated solution'
        
#         for _ in range(self.n):
#             # We can use the problems bounds accessor to get information about the problem bounds
#             x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
#             self.length = np.linalg.norm(x)
            
#             problem(x)

#x_init =setupp.kickstart_initial_solution(1)
x_init = np.ravel(np.random.rand(1,ioh_prob.problem_dimension))
# x_init[0] = 0.5
# x_init[1] = 0.5
# x_init[2] = np.arctan2(20,80)/np.pi
# x_init[3] = 1.00
# x_init[4] = 0.25


opts:cma.CMAOptions = {'bounds':[0,1],'tolfun':1e-6,'seed':RANDOM_SEED,'verb_filenameprefix':os.path.join(logger.output_directory,"outcmaes/"),
                       #"CMA_elitist":True}
}
#obj:ioh.iohcpp.RealConstraint = ioh_prob.constraints[0]
#print(type(obj),obj.is_feasible(x_init),obj(x_init))


#es = cma.evolution_strategy.fmin2(objective_function=ioh_prob,x0=x_init.tolist(),sigma0=0.25,eval_initial_x=True,restarts=10,options=opts)
#es = cma.evolution_strategy.fmin_con2(objective_function=ioh_prob,x0=x_init.tolist(),sigma0=0.25,constraints=,eval_initial_x=True,restarts=10,options=opts)

print(ioh_prob(x_init))
print(ioh_prob.compute_volume_ratio())
print(ioh_prob.constraints[0].violation(),
      ioh_prob.constraints[1].violation(),
      ioh_prob.constraints[2].violation(),
      ioh_prob.constraints[3].violation())

print(ioh_prob.state.current,
      ioh_prob.state.current_best,
      ioh_prob.state.current_best_internal)
ioh_prob.reset()

ioh_prob.attach_logger(logger)
# print(ioh_prob.constraints[0].is_feasible(x_init),
#       ioh_prob.constraints[1].is_feasible(x_init),
#       ioh_prob.constraints[2].is_feasible(x_init),
#       ioh_prob.constraints[3].is_feasible(x_init))


#fmin2(ioh_prob,x0=x_init,sigma0=0.5, restarts=10,options=opts,eval_initial_x=True)

# ioh_prob(np.array([
#           0.10134948995428217,
#             0.16233463388049874,
#               0.07157868044650234,
#                 0.8587460280468939,
#                   0.5820400994445769,
#                     0.9710072293346924,
#                       0.988509522087844,
#                         0.9214939159270011,
#                           0.9984886450413559,
#                             0.690955278058433,
#                               0.11386072883169937,
#                                 0.004928841524785139,
#                                   0.8630482038765932,
#                                     0.8178264774929366,
#                                       0.23501815097402287,
#                                         0.9467068730504858,
#                                           0.7788067786758578,
#                                             0.4205246872979749]))

ioh_prob(np.asarray([0.315993,
                     0.992194,
                       0.223560,
                         0.989524,
                           0.295946,
                             0.102780,
                               0.060761,
                                 0.997351,
                                   0.999988,
                                     0.869155,
                                       0.997227,
                                         0.999841,
                                           0.865063,
                                             0.994711,
                                               0.468883
    
]))

# ioh_prob(np.asarray([0.690270,
#                       0.882899,
#                         0.979480,
#                           0.081549,
#                             0.469595,
#                               0.673344,
#                                 0.723723,
#                                   0.150956,
#                                     0.282197,
#                                       0.998440,
#                                         0.545777,
#                                           0.781643,
#                                             0.108919,
#                                               0.629890,
#                                                 0.373743,
#                                                   0.969662,
#                                                     0.932918,
#                                                       0.112846,
#                                                         0.593810,
#                                                           0.707488,
#                                                             0.228357,
#                                                               0.272544,
#                                                                 0.052227,
#                                                                   0.811336,
#                                                                     0.878094,
#                                                                       0.729351,
#                                                                         0.593808,
#                                                                           0.006526,
#                                                                             0.268517,
#                                                                               0.184798,
#                                                                                 0.491615,
#                                                                                   0.024685,
#                                                                                     0.556982,
#                                                                                       0.681184,
#                                                                                         0.489442,
#                                                                                           0.803533,
#                                                                                             0.740681,
#                                                                                               0.958416,
#                                                                                                 0.136290,
#                                                                                                   0.912501,
#                                                                                                     0.970544, 0.772983,
#                                                                                                       0.549028,
#                                                                                                         0.085079,
#                                                                                                           0.762410,
#                                                                                                             0.921956,
#                                                                                                               0.652434,
#                                                                                                                 0.039647,
#                                                                                                                   0.800840,
#                                                                                                                     0.280888,
#                                                                                                                       0.939942,
#                                                                                                                         0.800401,
#                                                                                                                           0.388491  
# ]
# ))

# ioh_prob(np.asarray(
#     [
#        0.047869, 
#        0.064347,
#          0.999137,
#            0.337069,
#              0.789257,
#                0.915548,
#                  0.923095,
#                    0.071728,
#                      0.961833,
#                        0.768791,
#                          0.297292,
#                            0.338242,
#                              0.079145,
#                                0.587935,
#                                  0.619396
#     ]
# ))

# If we want to perform multiple runs with the same objective function, after every run, the problem has to be reset, 
# such that the internal state reflects the current run.
# def run_experiment(problem:ioh.problem.RealSingleObjective, algorithm:cma.CMAEvolutionStrategy, n_runs=5):
#     for run in range(n_runs):
        
#         # Run the algorithm on the problem
#         algorithm.optimize(objective_fct=problem,iterations=12,verb_disp=None)

#         # print the best found for this run
#         print(f"run: {run+1} - best found:{problem.state.current_best.y: .3f}")

#         # Reset the problem
#         problem.reset()


#run_experiment(problem=ioh_prob,algorithm=es,n_runs=1)
ioh_prob.reset()
logger.close()

