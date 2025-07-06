from typing import List, Optional
import argparse
import os
import ioh
from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper
from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper
from numpy import ndarray, asarray, zeros_like, zeros



def determine_dimension_a_priori_mmcs(nmmcsx:int, nmmcsy:int, symmetry_condition:bool) -> int:
    """
    Determine the dimension of the problem based on the number of master nodes in x and y directions.
    
    Args:
        nmmcsx (int): Number of master nodes in x-direction.
        nmmcsy (int): Number of master nodes in y-direction.
    
    Returns:
        int: The total dimension of the problem.
    """
    total_dimension = nmmcsx * nmmcsy * 5  # Each master node contributes 5 variables (e.g., x, y, z, vx, vy)
    if symmetry_condition:
        # If symmetry condition is active, reduce the dimension by half
        total_dimension //= 2
    return total_dimension

def determine_dimension_a_priori_lp(n_master_nodes:int) -> int:
    """
    Determine the dimension of the problem based on the number of master nodes.
    
    Args:
        n_master_nodes (int): Number of master nodes.
    
    Returns:
        int: The total dimension of the problem.
    """

    return n_master_nodes + 1

def args_parser(args:Optional[list]=None)-> argparse.Namespace:
    parser = argparse.ArgumentParser(description=r"""Run IOH_Wrapper_LP 
                                     or IOH_Wrapper with selected algorithm.""")
    # parser.add_argument(
    #     "--problem",
    #     choices=["IOH_Wrapper_LP", "IOH_Wrapper"],
    #     help="Choose which wrapper to run.",
    #     default="IOH_Wrapper_LP"
    # )

    parser.add_argument(
        "--material_definition",
        type=str,
        default="orthotropic",
        choices=["orthotropic", "isotropic", "quasi-isotropic"],
        help="Material definition to use in the problem.",
    )

    parser.add_argument(
        "--n_master_nodes",
        type=int,
        default=2,
        choices=[2,3],
        help="Number of master nodes for Lamination Parameter interpolation in the problem."
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["turbo-m", "turbo-1", "CMA-ES","BAxUS", "HEBO","Vanilla-BO","Vanilla-cBO", "DE",'Random-Search',"SCBO"],  # Replace with actual algorithm names
        default="CMA-ES",
        help="Algorithm to use."
    )

    # parser.add_argument(
    #     "--optimization_type",
    #     type=str,
    #     choices=["TO+LP", "TO", "LP"],
    #     help="Type of optimization problem to run.",
    #     default="TO+LP"
    # )

    parser.add_argument(
        "--budget",
        type=int,
        help="Number of total evaluations for the optimization.",
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
        help="Number of MCS in y-direction."
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
    #problem = argspace.problem
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
    material_definition = argspace.material_definition
    n_master_nodes = argspace.n_master_nodes


    dimension_LP = determine_dimension_a_priori_lp(n_master_nodes)
    dimension_mmcs = determine_dimension_a_priori_mmcs(nmmcsx, nmmcsy, symmetry_active)

    budget_to = int(budget*(dimension_mmcs/(dimension_mmcs + dimension_LP)))  # Budget for the TO problem
    budget_lp = budget - budget_to  # Budget for the LP problem

    print(f"Running with parameters: {argspace}")

    # Set the material definition dictionary based on the selected material definition
    if  material_definition == "orthotropic":
        material_definition_dict = {
            "E11": 25,  # Young's modulus in x-direction
            "E22": 1,   # Young's modulus in y-direction
            "nu12": 0.25,  # Poisson's ratio
            "G12": 0.5,  # Shear modulus
        }
    
    elif material_definition == "quasi-isotropic":
        material_definition_dict = {
            "E11": 13,  # Young's modulus in x-direction
            "E22": 13,  # Young's modulus in y-direction
            "nu12": 0.25,  # Poisson's ratio
            "G12": 13/((1+0.25)),   # Shear modulus
        }
    
    else:
        material_definition_dict = {
            "E11": 13,  # Young's modulus in x-direction
            "E22": 13,  # Young's modulus in y-direction
            "nu12": 0.25,  # Poisson's ratio
            "G12": 13/(2*(1+0.25)),   # Shear modulus
        }
    

    # Set the master nodes setup based on the selected number of master nodes
    if n_master_nodes == 2:
        interp_points = [(0.0,0.0),(1.0,0.5)]
    elif n_master_nodes == 3:
        interp_points = [(0.0,0.0),(1.0,0.5),(0.5,0.0)]
    

    ioh_prob = Design_IOH_Wrapper(
            random_seed=random_seed,
            run_=run_number,
            symmetry_condition=symmetry_active,
            nmmcsx=nmmcsx,
            nmmcsy=nmmcsy,
            nelx=nelx,
            nely=nely,
            volfrac=volfrac,
            continuity_check_mode=continuity_check_mode,
            plot_variables=plot_variables,
            material_properties_dict=material_definition_dict
    )
    
    

    # # Create the problem instance based on the selected wrapper
    # if problem == "IOH_Wrapper_LP":
    #     from numpy import ndarray, asarray, zeros_like, zeros
    #     ioh_prob = Design_LP_IOH_Wrapper(
    #         random_seed=random_seed,
    #         run_=run_number,
    #         symmetry_condition=symmetry_active,
    #         nmmcsx=nmmcsx,
    #         nmmcsy=nmmcsy,
    #         nelx=nelx,
    #         nely=nely,
    #         volfrac=volfrac,
    #         VR=0.5,
    #         V3_list=zeros((n_master_nodes,),dtype=float).tolist(),  # -0.1, -0.4
    #         continuity_check_mode=continuity_check_mode,
    #         plot_variables=plot_variables,
    #         interpolation_points=interp_points,
    #     )
    # else:
    #     ioh_prob = Design_IOH_Wrapper(
    #         random_seed=random_seed,
    #         run_=run_number,
    #         symmetry_condition=symmetry_active,
    #         nmmcsx=nmmcsx,
    #         nmmcsy=nmmcsy,
    #         nelx=nelx,
    #         nely=nely,
    #         volfrac=volfrac,
    #         continuity_check_mode=continuity_check_mode,
    #         plot_variables=plot_variables,
    #         material_properties_dict=material_definition_dict
    #     )
    
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

    
    # Convert the constraints to the hidden type
    ioh_prob.convert_defined_constraint_to_type(0,2) # Dirichlet
    ioh_prob.convert_defined_constraint_to_type(1,2) # Neumann
    ioh_prob.convert_defined_constraint_to_type(2,2) # Connectivity

    # Convert volume constraint to not
    if algorithm_name not in ["Vanilla-cBO", "SCBO"]:
        ioh_prob.convert_defined_constraint_to_type(3,3) # Volume
    else:
        ioh_prob.convert_defined_constraint_to_type(3,1)

    logger.watch(ioh_prob,"n_evals")
    logger.watch(ioh_prob,"evaluation_time")
    logger.watch(ioh_prob,"actual_volume_excess")  # Track the number of constraints


    ioh_prob.attach_logger(logger)  # Attach the logger to the problem instance

    # Set up the algorithm based on the selected algorithm
    if algorithm_name == "turbo-m":
        from Algorithms.turbo_m_wrapper import Turbo_M_Wrapper
        algorithm = Turbo_M_Wrapper(ioh_prob,
                                    n_trust_regions=3,                          
                                    batch_size=batch_size)
        
        logger.watch(algorithm,"running_time")
        try:
            algorithm(total_budget=budget_to,
                      random_seed=random_seed,
                      n_DoE= n_doe_mult*ioh_prob.problem_dimension)
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget_to,
                  random_seed=random_seed,
                  n_DoE= n_doe_mult*ioh_prob.problem_dimension)
        
    elif algorithm_name == "turbo-1":
        from Algorithms.turbo_1_wrapper import Turbo_1_Wrapper
        algorithm = Turbo_1_Wrapper(ioh_prob, 
                                    batch_size=batch_size)
        
        logger.watch(algorithm,"running_time")
        
        try:
            # Run the algorithm with the specified parameters
            algorithm(total_budget=budget_to,
                        random_seed=random_seed,
                        n_DoE= n_doe_mult*ioh_prob.problem_dimension)
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
    
    elif algorithm_name == "BAxUS":
        from Algorithms.baxus_wrapper import BAxUS_Wrapper
        algorithm = BAxUS_Wrapper(ioh_prob, batch_size=batch_size)

        logger.watch(algorithm,"running_time")

        try:
        # Run the algorithm with the specified parameters
            algorithm(total_budget=budget_to,
                    random_seed=random_seed,
                    n_DoE=n_doe_mult*ioh_prob.problem_dimension)
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
        
    elif algorithm_name == "CMA-ES":
        from Algorithms.cma_es_wrapper import CMA_ES_Optimizer_Wrapper
        algorithm = CMA_ES_Optimizer_Wrapper(ioh_prob, random_seed=random_seed,
                                             sigma0=sigma_0)
        
        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        try:
            algorithm(max_f_evals=budget_to,
                        restarts=10,
                        cma_active=True,
                        random_seed=random_seed,
                        verb_filenameprefix=os.path.join(logger.output_directory,"outcmaes","Non_LP/"))
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
        
    elif algorithm_name == "HEBO":
        from Algorithms.hebo_wrapper import HEBO_Wrapper
        algorithm = HEBO_Wrapper(ioh_prob, batch_size=batch_size)

        logger.watch(algorithm,"running_time")
        
        try:
            # Run the algorithm with the specified parameters
            algorithm(budget=budget_to,
                    random_seed=random_seed,
                    n_DOE=n_doe_mult*ioh_prob.problem_dimension)
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
        
    elif algorithm_name == "Vanilla-BO":
        from Algorithms.vanilla_bo_wrapper import VanillaBO
        # Initialize the VanillaBO algorithm with the problem instance and random seed
        # Note: VanillaBO is a placeholder for the actual implementation
        algorithm = VanillaBO(ioh_prob,
                              num_restarts=5,
                              batch_size=batch_size)
        
        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        try:
            algorithm(total_budget=budget_to,
                    random_seed=random_seed,
                    n_DoE=n_doe_mult*ioh_prob.problem_dimension)
        
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
        
    elif algorithm_name == "DE":
        from Algorithms.de_wrapper import DifferentialEvolutionWrapper
        algorithm = DifferentialEvolutionWrapper(ioh_prob)
        
        logger.watch(algorithm,"running_time")
        # Run the algorithm with the specified parameters

        try:
            algorithm(budget=budget_to,
                    random_seed=random_seed)
        
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
        
    elif algorithm_name == "Random-Search":
        from Algorithms.random_search_wrapper import RandomSearchWrapper
        algorithm = RandomSearchWrapper(ioh_prob)

        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        try:
            algorithm(budget=budget_to,
                    random_seed=random_seed)
        
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
        
    elif algorithm_name == "SCBO":
        from Algorithms.scbo_wrapper import SCBO_Wrapper
        algorithm = SCBO_Wrapper(ioh_prob, batch_size=batch_size)

        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        try:
            algorithm(total_budget=budget_to,
                    random_seed=random_seed,
                    n_DoE=n_doe_mult*ioh_prob.problem_dimension)
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")
    
    elif algorithm_name == "Vanilla-cBO":
        from Algorithms.vanilla_cbo_wrapper import VanillaCBO
        # Initialize the VanillaCBO algorithm with the problem instance and random seed
        # Note: VanillaCBO is a placeholder for the actual implementation
        algorithm = VanillaCBO(ioh_prob,
                               num_restarts=5,
                               batch_size=batch_size)
        
        logger.watch(algorithm,"running_time")
        
        # Run the algorithm with the specified parameters
        try:
            algorithm(total_budget=budget_to,
                    random_seed=random_seed,
                    n_DoE=n_doe_mult*ioh_prob.problem_dimension)
        except Exception as e:
            print(f"Exception occurred during algorithm execution: {e.args}")

    # Store the best solution found by the non-LP problem
    best_non_LP_solution = ioh_prob.state.current_best

    print(f"Best MMC Combination solution found: {best_non_LP_solution}")

    ioh_prob.reset()
    logger.reset()  # Reset the logger after the optimization process
    ioh_prob.detach_logger()

    logger = ioh.logger.Analyzer(
        root=os.getcwd(),                  # Store data in the current working directory
        folder_name=f"./Figures_Python/Run_{run_number}_LP",       # in a folder named: './Figures_Python/Run_{run_e}'
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


    # Set the problem instances for the problem
    ioh_prob_LP = Design_LP_IOH_Wrapper(
            random_seed=random_seed,
            run_=run_number,
            symmetry_condition=symmetry_active,
            nmmcsx=nmmcsx,
            nmmcsy=nmmcsy,
            nelx=nelx,
            nely=nely,
            volfrac=volfrac,
            VR=0.5,
            V3_list=zeros((n_master_nodes,),dtype=float).tolist(),  # -0.1, -0.4
            continuity_check_mode=continuity_check_mode,
            plot_variables=plot_variables,
            interpolation_points=interp_points,
            mode="LP",
            material_properties_dict=material_definition_dict
        )
    

    # Change the MMC configuration
    ioh_prob_LP.change_values_of_MMCs_from_unscaled_array(asarray(best_non_LP_solution.x),
                                                        repair_level=0)
    

    # Convert the constraints to the hidden type
    ioh_prob_LP.convert_defined_constraint_to_type(0,2) # Dirichlet
    ioh_prob_LP.convert_defined_constraint_to_type(1,2) # Neumann
    ioh_prob_LP.convert_defined_constraint_to_type(2,2) # Connectivity

    ioh_prob_LP.convert_defined_constraint_to_type(3,3)

    # Convert volume constraint to not
    #if algorithm_name not in ["Vanilla-cBO", "SCBO"]:
    #    ioh_prob.convert_defined_constraint_to_type(3,3) # Volume
    #else:
    #    ioh_prob.convert_defined_constraint_to_type(3,1)

    logger.watch(ioh_prob_LP,"n_evals")
    logger.watch(ioh_prob_LP,"evaluation_time")
    logger.watch(ioh_prob_LP,"actual_volume_excess")  # Track the number of constraints


    ioh_prob_LP.attach_logger(logger)  # Attach the logger to the problem instance

    # Set up the algorithm based on the selected algorithm
    if algorithm_name == "turbo-m":
        from Algorithms.turbo_m_wrapper import Turbo_M_Wrapper
        algorithm = Turbo_M_Wrapper(ioh_prob_LP,
                                    n_trust_regions=3,                          
                                    batch_size=batch_size)
        
        logger.watch(algorithm,"running_time")
        
        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget_lp,
                  random_seed=random_seed,
                  n_DoE= n_doe_mult*ioh_prob_LP.problem_dimension)
        
    elif algorithm_name == "turbo-1":
        from Algorithms.turbo_1_wrapper import Turbo_1_Wrapper
        algorithm = Turbo_1_Wrapper(ioh_prob_LP, 
                                    batch_size=batch_size)
        
        logger.watch(algorithm,"running_time")
        
        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget_lp,
                  random_seed=random_seed,
                  n_DoE= n_doe_mult*ioh_prob_LP.problem_dimension)
    
    elif algorithm_name == "BAxUS":
        from Algorithms.baxus_wrapper import BAxUS_Wrapper
        algorithm = BAxUS_Wrapper(ioh_prob_LP, batch_size=batch_size)

        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget_lp,
                  random_seed=random_seed,
                  n_DoE=n_doe_mult*ioh_prob_LP.problem_dimension)
        
    elif algorithm_name == "CMA-ES":
        from Algorithms.cma_es_wrapper import CMA_ES_Optimizer_Wrapper
        algorithm = CMA_ES_Optimizer_Wrapper(ioh_prob_LP, random_seed=random_seed,
                                             sigma0=sigma_0)
        
        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        algorithm(max_f_evals=budget_lp,
                  restarts=10,
                  cma_active=True,
                  random_seed=random_seed,
                  verb_filenameprefix=os.path.join(logger.output_directory,"outcmaes","LP/"))
        
    elif algorithm_name == "HEBO":
        from Algorithms.hebo_wrapper import HEBO_Wrapper
        algorithm = HEBO_Wrapper(ioh_prob_LP, batch_size=batch_size)

        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        algorithm(budget=budget_lp,
                  random_seed=random_seed,
                  n_DOE=n_doe_mult*ioh_prob_LP.problem_dimension)
        
    elif algorithm_name == "Vanilla-BO":
        from Algorithms.vanilla_bo_wrapper import VanillaBO
        # Initialize the VanillaBO algorithm with the problem instance and random seed
        # Note: VanillaBO is a placeholder for the actual implementation
        algorithm = VanillaBO(ioh_prob_LP,
                              num_restarts=5,
                              batch_size=batch_size)
        
        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget_lp,
                  random_seed=random_seed,
                  n_DoE=n_doe_mult*ioh_prob_LP.problem_dimension)
        
    elif algorithm_name == "DE":
        from Algorithms.de_wrapper import DifferentialEvolutionWrapper
        algorithm = DifferentialEvolutionWrapper(ioh_prob_LP)
        
        logger.watch(algorithm,"running_time")
        # Run the algorithm with the specified parameters
        algorithm(budget=budget_lp,
                  random_seed=random_seed)
        
    elif algorithm_name == "Random-Search":
        from Algorithms.random_search_wrapper import RandomSearchWrapper
        algorithm = RandomSearchWrapper(ioh_prob_LP)

        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        algorithm(budget=budget_lp,
                  random_seed=random_seed)
        
    elif algorithm_name == "SCBO":
        from Algorithms.turbo_1_wrapper import Turbo_1_Wrapper
        algorithm = Turbo_1_Wrapper(ioh_prob_LP, batch_size=batch_size)

        logger.watch(algorithm,"running_time")

        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget_lp,
                  random_seed=random_seed,
                  n_DoE=n_doe_mult*ioh_prob_LP.problem_dimension)
    
    elif algorithm_name == "Vanilla-cBO":
        from Algorithms.vanilla_bo_wrapper import VanillaBO
        # Initialize the VanillaCBO algorithm with the problem instance and random seed
        # Note: VanillaCBO is a placeholder for the actual implementation
        algorithm = VanillaBO(ioh_prob_LP,
                               num_restarts=5,
                               batch_size=batch_size)
        
        logger.watch(algorithm,"running_time")
        
        # Run the algorithm with the specified parameters
        algorithm(total_budget=budget_lp,
                  random_seed=random_seed,
                  n_DoE=n_doe_mult*ioh_prob_LP.problem_dimension)
        

    #logger.reset()  # Reset the logger after the optimization process
    ioh_prob_LP.reset()  # Reset the problem instance after the optimization process
    ioh_prob_LP.detach_logger()  # Detach the logger from the problem instance
    logger.reset()