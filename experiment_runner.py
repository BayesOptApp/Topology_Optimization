from typing import List, Optional
import argparse
import os
import ioh
from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper
from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper

def args_parser(args:Optional[list]=None)-> argparse.Namespace:
    parser = argparse.ArgumentParser(description=r"""Run IOH_Wrapper_LP 
                                     or IOH_Wrapper with selected algorithm.""")
    parser.add_argument(
        "--problem",
        choices=["IOH_Wrapper_LP", "IOH_Wrapper"],
        help="Choose which wrapper to run.",
        default="IOH_Wrapper_LP"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["turbo-m", "turbo-1", "CMA-ES","BAxUS", "HEBO","Vanilla-BO", "DE",'Random-Search'],  # Replace with actual algorithm names
        default="CMA-ES",
        help="Algorithm to use."
    )

    parser.add_argument(
        "--optimization_type",
        type=str,
        choices=["TO+LP", "TO", "LP"],
        help="Type of optimization problem to run.",
        default="TO+LP"
    )

    parser.add_argument(
        "--budget",
        type=int,
        help="Number of evaluations for the optimization.",
        default=1000
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--run_number",
        type=int,
        default=1,
        help="Run number"
    )

    parser.add_argument(
        "--symmetry_active",
        type=int,
        choices=[0, 1],
        default=1,
        help="Activate symmetry condition in the problem."
    )

    parser.add_argument(
        "--nmmcsx",
        type=int,
        default=3,
        help="Number of MCS in x-direction."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size of the algorithm, i.e. how many points to evaluate in parallel."
    )

    parser.add_argument(
        "--nmmcsy",
        type=int,
        default=2,
        help="Number of MCS in x-direction."
    )

    parser.add_argument(
        "--volfrac",
        type=float,
        default=0.5,
        choices=[float(i)/100 for i in range(1, 101)],
        help="Volume fraction for the problem."
    )

    parser.add_argument(
        "--n_doe_mult",
        type=int,
        default=3,
        help="Multiplier of the initial DoE"
    )

    parser.add_argument(
        "--continuity_check_mode",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
        help="Continuity check mode for the problem."
    )

    parser.add_argument(
        "--nelx",
        type=int,
        default=100,
        choices=[80, 100, 120, 140, 160,180, 200],
        help="Number of elements in the x-direction."
    )

    parser.add_argument(
        "--nely",
        type=int,
        default=50,
        choices=[40, 50, 60, 70, 80, 90, 100],
        help="Number of elements in the x-direction."
    )

    parser.add_argument(
        "--plot_variables",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to plot variables during the optimization."
    )

    parser.add_argument(
        "--sigma_0",
        type=float,
        default=0.25,
        help="Initial sigma for the CMA-ES algorithm."
    )

    argspace = parser.parse_args(args)

    return argspace

if __name__ == "__main__":
    import sys
    # Parse command line arguments
    # This allows the script to be run with command line arguments
    argspace:argparse.Namespace = args_parser(sys.argv[1:])  # Pass command line arguments to the parser

    # Set the algorithm based on the parsed arguments
    algorithm_name = argspace.algorithm
    problem = argspace.problem
    optimization_type = argspace.optimization_type
    random_seed = argspace.random_seed
    run_number = argspace.run_number
    symmetry_active = bool(argspace.symmetry_active)
    nmmcsx = argspace.nmmcsx
    nmmcsy = argspace.nmmcsy
    volfrac = argspace.volfrac
    n_doe_mult = argspace.n_doe_mult
    continuity_check_mode = argspace.continuity_check_mode
    plot_variables = bool(argspace.plot_variables)
    nelx = argspace.nelx
    nely = argspace.nely
    batch_size = argspace.batch_size
    sigma_0 = argspace.sigma_0
    budget = argspace.budget

    print(f"Running with parameters: {argspace}")


    # Create the problem instance based on the selected wrapper
    if problem == "IOH_Wrapper_LP":
        ioh_prob = Design_LP_IOH_Wrapper(
            mode=optimization_type,
            random_seed=random_seed,
            run_=run_number,
            symmetry_condition=symmetry_active,
            nmmcsx=nmmcsx,
            nmmcsy=nmmcsy,
            nelx=nelx,
            nely=nely,
            volfrac=volfrac,
            VR=0.5,
            V3_list=[0, 0],  # -0.1, -0.4
            continuity_check_mode=continuity_check_mode,
            plot_variables=plot_variables,
        )
    else:
        ioh_prob = Design_IOH_Wrapper(
            random_seed=random_seed,
            run_=run_number,
            symmetry_condition=symmetry_active,
            nmmcsx=nmmcsx,
            nmmcsy=nmmcsy,
            nelx=nelx,
            nely=nely,
            volfrac=volfrac,
            VR=0.5,
            V3_list=[0, 0],  # -0.1, -0.4
            continuity_check_mode=continuity_check_mode,
            plot_variables=plot_variables,
        )
    
    r"""
The next excerpt of code is just setting the IOH Logger. You may check the IOH Experimenter Wiki to see other ways to Log the corresponding results.
"""

    triggers = [
        ioh.logger.trigger.Each(1),
        ioh.logger.trigger.OnImprovement()
    ]

    logger = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name=f"./Figures_Python/Run_{run_number}",       # in a folder named: './Figures_Python/Run_{run_e}'
        algorithm_name=algorithm_name,    # meta-data for the algorithm used to generate these results
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
    ioh_prob.convert_defined_constraint_to_type(0,2) # Dirichlet
    ioh_prob.convert_defined_constraint_to_type(1,2) # Neumann

    # Convert connectivity to a Hard constraint
    ioh_prob.convert_defined_constraint_to_type(2,2) # Connectivity

    # Convert volume constraint soft
    ioh_prob.convert_defined_constraint_to_type(3,3) # Volume

    logger.watch(ioh_prob,"n_evals")

    ioh_prob.attach_logger(logger)  # Attach the logger to the problem instance

    # Set up the algorithm based on the selected algorithm
    if algorithm_name == "turbo-m":
        from Algorithms.turbo_m_wrapper import Turbo_M_Wrapper
        algorithm = Turbo_M_Wrapper(ioh_prob,
                                    n_trust_regions=3,                          
                                    batch_size=batch_size)
        
        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget,
                  random_seed=random_seed,
                  n_DoE= n_doe_mult*ioh_prob.problem_dimension)
        
    elif algorithm_name == "turbo-1":
        from Algorithms.turbo_1_wrapper import Turbo_1_Wrapper
        algorithm = Turbo_1_Wrapper(ioh_prob, 
                                    batch_size=batch_size)
        
        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget,
                  random_seed=random_seed,
                  n_DoE= n_doe_mult*ioh_prob.problem_dimension)
    
    elif algorithm_name == "BAxUS":
        from Algorithms.baxus_wrapper import BAxUS_Wrapper
        algorithm = BAxUS_Wrapper(ioh_prob, batch_size=batch_size)
        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget,
                  random_seed=random_seed,
                  n_DoE=n_doe_mult*ioh_prob.problem_dimension)
        
    elif algorithm_name == "CMA-ES":
        from Algorithms.cma_es_wrapper import CMA_ES_Optimizer_Wrapper
        algorithm = CMA_ES_Optimizer_Wrapper(ioh_prob, random_seed=random_seed,
                                             sigma0=sigma_0)
        # Run the algorithm with the specified parameters
        algorithm(max_f_evals=budget,
                  restarts=10,
                  cma_active=True,
                  random_seed=random_seed,
                  verb_filenameprefix=os.path.join(logger.output_directory,"outcmaes/"))
        
    elif algorithm_name == "HEBO":
        from Algorithms.hebo_wrapper import HEBO_Wrapper
        algorithm = HEBO_Wrapper(ioh_prob, batch_size=batch_size)
        # Run the algorithm with the specified parameters
        algorithm(budget=budget,
                  random_seed=random_seed,
                  n_DoE=n_doe_mult*ioh_prob.problem_dimension)
    elif algorithm_name == "Vanilla-BO":
        from Algorithms.vanilla_bo_wrapper import VanillaBO
        # Initialize the VanillaBO algorithm with the problem instance and random seed
        # Note: VanillaBO is a placeholder for the actual implementation
        algorithm = VanillaBO(ioh_prob,
                              num_restarts=5,
                              batch_size=batch_size)
        # Run the algorithm with the specified parameters
        algorithm(budget=budget,
                  random_seed=random_seed,
                  n_DoE=n_doe_mult*ioh_prob.problem_dimension)
        
    elif algorithm_name == "DE":
        from Algorithms.de_wrapper import DifferentialEvolutionWrapper
        algorithm = DifferentialEvolutionWrapper(ioh_prob)
        # Run the algorithm with the specified parameters
        algorithm(budget=budget,
                  random_seed=random_seed)
    elif algorithm_name == "Random-Search":
        from Algorithms.random_search_wrapper import RandomSearchWrapper
        algorithm = RandomSearchWrapper(ioh_prob)

        # Run the algorithm with the specified parameters
        algorithm(budget=budget,
                  random_seed=random_seed)



    ioh_prob.reset()
    logger.close()
