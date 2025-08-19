import importlib
from typing import Union, Optional, List, Tuple, Dict
from Design_Examples.IOH_Wrappers.IOH_Wrapper_Instanced import Design_IOH_Wrapper_Instanced
from Design_Examples.Raw_Design.Design import DEFAULT_MATERIAL_PROPERTIES
from boundary_conditions import BoundaryConditionList, PointDirichletBC, LineDirichletBC, PointNeumannBC
import os


### -----------------------------------------------------------------------------------------------------
### CONSTANTS
### -----------------------------------------------------------------------------------------------------

DEFAULT_PROBLEM_NAMES = ("cantilever_beam", "short_beam", "mbb", "michell_truss")
ALLOWED_NUMBERS = [*range(1, len(DEFAULT_PROBLEM_NAMES) + 1)]
                         

def get_problem(problem_id:Union[str,int], 
                dimension:int,
                instance:Optional[int]=0, 
                plot_topology:Optional[bool]=False,
                plot_stresses:Optional[bool]=False,
                run_number:Optional[int] = 1,
                penalty_function:Optional[bool]=True)->Design_IOH_Wrapper_Instanced:
    """
    Dynamically imports and returns a problem class from the problems package.

    Args
    ------------
        - problem (`Union[str, int]`): The name or index of the problem to import.
        - dimension (`int`): The dimension of the problem.
        - instance (`int`): The instance number for the problem.
        - plot_stresses (`Optional[bool]`): Whether to plot stresses for the problem.
        - plot_topology (`Optional[bool]`): Whether to plot the topology for the problem.
        - run_number (`Optional[int]`): The run number for the problem instance.
        - penalty_function (`Optional[bool]`): Whether to apply a penalty function to the problem.

    Returns:
        - `Design_IOH_Wrapper`: An instance of the problem class.

    Raises:
        ImportError: If the module or class cannot be found.
    """

    # Determine the problem name based on input
    if isinstance(problem_id, int):
        if problem_id not in ALLOWED_NUMBERS:
            raise ValueError(f"Problem number must be one of {ALLOWED_NUMBERS}.")
        problem_name = DEFAULT_PROBLEM_NAMES[problem_id - 1]
    elif isinstance(problem_id, str):
        if problem_id.lower() not in DEFAULT_PROBLEM_NAMES:
            raise ValueError(f"Problem name must be one of {DEFAULT_PROBLEM_NAMES}.")
        problem_name = problem_id
    else:
        raise TypeError("Problem must be a string or an integer.")
    

    # Instantiate a variable to store the problem instance
    problem:Optional[Design_IOH_Wrapper_Instanced] = None
    
    # Now given the parameters, we can import the problem
    if problem_name == "cantilever_beam":
       problem = _set_cantilever_beam_problem(dimension, instance, plot_topology, plot_stresses, run_number)
    elif problem_name == "short_beam":
        problem = _set_short_beam_problem(dimension, instance, plot_topology, plot_stresses, run_number)
    elif problem_name == "mbb":
        problem = _set_mbb_problem(dimension, instance, plot_topology, plot_stresses, run_number)
    elif problem_name == "michell_truss":
        problem = _set_michell_truss_problem(dimension, instance, plot_topology, plot_stresses, run_number)
    else:
        raise ValueError(f"Unknown problem name: {problem_name}")
    
    # Convert the first two constraints to HIDDEN constraints
    problem.convert_defined_constraint_to_type(0,2) # Dirichlet
    problem.convert_defined_constraint_to_type(1,2) # Neumann

    # Convert connectivity to HIDDEN constraint
    problem.convert_defined_constraint_to_type(2,2) # Connectivity

    if penalty_function:
        problem.convert_defined_constraint_to_type(3,3) # Volume
    else:
        problem.convert_defined_constraint_to_type(3,1) # Volume without penalty


    return problem
    

    



def _set_cantilever_beam_problem(dimension:int, 
                                 instance:int,
                                 plot_topology:bool,
                                 plot_stresses:bool,
                                 run_number:int)->Design_IOH_Wrapper_Instanced:
    """
    Sets up the Cantilever Beam problem.

    Args
    ------------
        - dimension (`int`): The dimension of the problem.
        - instance (`int`): The instance number for the problem.
        - plot_stresses (`bool`): Whether to plot stresses for the problem.
        - run_number (`int`): The run number for the problem instance.

    Returns:
        - `Design_IOH_Wrapper`: An instance of the Cantilever Beam problem.
    """

    ### ------------------------------------------------------------------------------------------------------
    ### CONSTANTS OF THIS CASE
    ### ------------------------------------------------------------------------------------------------------

    NELX:int = 120
    NELY:int = 60

    # Check the dimension is a multiple of 5
    if dimension % 5 != 0:
        raise ValueError("Dimension must be a multiple of 5 for Cantilever Beam problem.")
    
    # Generate a random generation by using the instance
    import numpy as np
    #rng = np.random.default_rng(instance)

    
    bound_ = 0.05*0.25
    # Get a uniform random number between the bounds
    # if instance == 0:
    #     random_number = 1.0
    # else:
    #     random_number = 1 + rng.uniform(-0.05, 0.05)
    
    # Get the number of MMC given the dimension
    num_mmc:int = dimension // 5
    
    # Set the boundary conditions for this problem
    boundary_conditions = BoundaryConditionList()

    # Add Dirichlet boundary conditions at the left edge
    dirichlet_BC_left = LineDirichletBC(start_point=(0.0, 0.0),
                                            end_point=(0.0, 1.0),
                                            blocked_dof=(1,2))
    
    boundary_conditions.add(dirichlet_BC_left)

    # Add Neumann boundary condition at the right edge
    neumann_BC_right = PointNeumannBC(location=(1.0,0.5),
                                      force_vector=(0.0, -0.25))
    
    boundary_conditions.add(neumann_BC_right)

    # Create the problem instance
    problem = Design_IOH_Wrapper_Instanced(
        nmmcsx= num_mmc,
        nmmcsy= 2,
        instance= instance,
        nelx= NELX,
        nely= NELY,
        volfrac= 0.5,
        symmetry_condition= True,
        scalation_mode="unitary",
        plot_topology=plot_topology,
        plot_variables=plot_stresses,
        cost_function= "compliance",
        use_sparse_matrices= True,
        continuity_check_mode= "discrete",
        run_= run_number,
        boundary_conditions_list= boundary_conditions,
        problem_aux_name = "cantilever_beam"
    )

    return problem


def _set_short_beam_problem(dimension:int, 
                            instance:int,
                            plot_topology:bool,
                            plot_stresses:bool,
                            run_number:int)->Design_IOH_Wrapper_Instanced:
    """
    Sets up the Short Beam problem.

    Args
    ------------
        - dimension (`int`): The dimension of the problem.
        - instance (`int`): The instance number for the problem.
        - plot_stresses (`bool`): Whether to plot stresses for the problem.
        - run_number (`int`): The run number for the problem instance.

    Returns:
        - `Design_IOH_Wrapper`: An instance of the Short Beam problem.
    """

    ### ------------------------------------------------------------------------------------------------------
    ### CONSTANTS OF THIS CASE
    ### ------------------------------------------------------------------------------------------------------

    NELX:int = 120
    NELY:int = 60

    # Check the dimension is a multiple of 5
    if dimension % 5 != 0:
        raise ValueError("Dimension must be a multiple of 5 for Short Beam problem.")
    
    # Ensure the dimension is at least 10
    if dimension < 10:
        raise ValueError("Dimension must be at least 10 for Short Beam problem.")
    
    # Generate a random generation by using the instance
    import numpy as np
    #rng = np.random.default_rng(instance)

    
    bound_ = 0.05*0.25/30

    # Get a uniform random number between the bounds
    # if instance == 0:
    #     random_number = 1.0
    # else:
    #     random_number = 1 + rng.uniform(-0.05, 0.05)
    
    # Get the number of MMC given the dimension
    num_mmc:int = dimension // 5
    
    # Set the boundary conditions for this problem
    boundary_conditions = BoundaryConditionList()

    # Add Dirichlet boundary conditions at the left edge
    dirichlet_BC_left = LineDirichletBC(start_point=(0.0, 0.0),
                                            end_point=(0.0, 1.0),
                                            blocked_dof=(1, 2))
    
    boundary_conditions.add(dirichlet_BC_left)

    # Add Neumann boundary condition at the right edge
    neumann_BC_right = PointNeumannBC(location=(1.0,0.0),
                                      force_vector=(0.0, -0.25/5))
    
    boundary_conditions.add(neumann_BC_right)

    # Create the problem instance
    problem = Design_IOH_Wrapper_Instanced(
        nmmcsx= num_mmc,
        instance= instance,
        nmmcsy= 1,
        nelx= NELX,
        nely= NELY,
        volfrac= 0.5,
        symmetry_condition= False,
        scalation_mode="unitary",
        plot_topology=plot_topology,
        plot_variables=plot_stresses,
        cost_function= "compliance",
        use_sparse_matrices= True,
        continuity_check_mode= "discrete",
        run_= run_number,
        boundary_conditions_list= boundary_conditions,
        problem_aux_name = "short_beam",
        Emin=1e-7,
        standard_weight=5000.0,
    )

    # Modify the penalties of the problem


    return problem


def _set_mbb_problem(dimension:int,
                     instance:int,
                     plot_topology:bool,
                     plot_stresses:bool,
                     run_number:int)->Design_IOH_Wrapper_Instanced:
    """
    Sets up the MBB Beam problem.

    Args
    ------------
        - dimension (`int`): The dimension of the problem.
        - instance (`int`): The instance number for the problem.
        - plot_stresses (`bool`): Whether to plot stresses for the problem.
        - run_number (`int`): The run number for the problem instance.

    Returns:
        - `Design_IOH_Wrapper`: An instance of the MBB problem.
    """

    ### ------------------------------------------------------------------------------------------------------
    ### CONSTANTS OF THIS CASE
    ### ------------------------------------------------------------------------------------------------------

    NELX:int = 150
    NELY:int = 50

    # Check the dimension is a multiple of 5
    if dimension % 5 != 0:
        raise ValueError("Dimension must be a multiple of 5 for MBB problem.")
    
    # Ensure the dimension is at least 10
    if dimension < 10:
        raise ValueError("Dimension must be at least 10 for MBB problem.")
    
    # Generate a random generation by using the instance
    import numpy as np
    #rng = np.random.default_rng(instance)

    
    #bound_ = 0.05*0.25/10

    # Get a uniform random number between the bounds
    # if instance == 0:
    #     random_number = 1.0
    # else:
    #     random_number = 1 + rng.uniform(-0.05, 0.05)
    
    # Get the number of MMC given the dimension
    num_mmc:int = dimension // 5
    
    # Set the boundary conditions for this problem
    boundary_conditions = BoundaryConditionList()

    # Add Dirichlet boundary conditions at the left edge
    dirichlet_BC_left = LineDirichletBC(start_point=(0.0, 0.0),
                                            end_point=(0.0, 1.0),
                                            blocked_dof=1)
    
    boundary_conditions.add(dirichlet_BC_left)

    # Add Dirichlet boundary conditions at the right bottom node
                                            
    dirichlet_BC_right = PointDirichletBC(location=(1.0, 0.0),
                                            blocked_dof=(2))
    
    boundary_conditions.add(dirichlet_BC_right)

    # Add Neumann boundary condition at the right edge
    neumann_BC_top = PointNeumannBC(location=(1.0/NELX,1.0),
                                      force_vector=(0.0, -0.25/10),
                                      )
    
    boundary_conditions.add(neumann_BC_top)

    # Create the problem instance
    problem = Design_IOH_Wrapper_Instanced(
        nmmcsx= num_mmc,
        nmmcsy= 1,
        nelx= NELX,
        nely= NELY,
        instance= instance,
        volfrac= 0.5,
        symmetry_condition= False,
        scalation_mode="unitary",
        plot_topology=plot_topology,
        plot_variables=plot_stresses,
        cost_function= "compliance",
        use_sparse_matrices= True,
        continuity_check_mode= "discrete",
        run_= run_number,
        boundary_conditions_list= boundary_conditions,
        problem_aux_name = "mbb",
        Emin=5e-8,
        standard_weight=3000.0,
    )

    return problem


def _set_michell_truss_problem(dimension:int, 
                               instance:int,
                               plot_topology:bool,
                                plot_stresses:bool,
                                run_number:int)->Design_IOH_Wrapper_Instanced:
    """
    Sets up the Michell Truss problem.

    Args
    ------------
        - dimension (`int`): The dimension of the problem.
        - instance (`int`): The instance number for the problem.
        - plot_stresses (`bool`): Whether to plot stresses for the problem.
        - run_number (`int`): The run number for the problem instance.

    Returns:
        - `Design_IOH_Wrapper`: An instance of the Michell Truss problem.
    """

    ### ------------------------------------------------------------------------------------------------------
    ### CONSTANTS OF THIS CASE
    ### ------------------------------------------------------------------------------------------------------

    NELX:int = 50
    NELY:int = 150


    # Generate a random generation by using the instance
    import numpy as np
    #rng = np.random.default_rng(instance)

    
    bound_ = 0.05*0.25

    # Get a uniform random number between the bounds
    # if instance == 0:
    #     random_number = 1.0
    # else:
    #     random_number = 1 + rng.uniform(-0.05, 0.05)

    # Invert the material properties
    material_properties = DEFAULT_MATERIAL_PROPERTIES.copy()

    temp = material_properties["E11"]
    material_properties["E11"] = material_properties["E22"]
    material_properties["E22"] = temp


    # Check the dimension is a multiple of 5
    if dimension % 5 != 0:
        raise ValueError("Dimension must be a multiple of 5 for Michell Truss problem.")
    
    # Ensure the dimension is at least 5
    if dimension < 5:
        raise ValueError("Dimension must be at least 10 for Michell Truss problem.")
    
    # Get the number of MMC given the dimension
    num_mmc:int = dimension // 5
    
    # Set the boundary conditions for this problem
    boundary_conditions = BoundaryConditionList()

    # Add Dirichlet boundary conditions at the left edge
    dirichlet_BC_left = PointDirichletBC(location=(0.0, 0.0),
                                            blocked_dof=(1,2))
    
    boundary_conditions.add(dirichlet_BC_left)

    # Add Dirichlet boundary conditions at the right bottom node
                                            
    dirichlet_BC_right = PointDirichletBC(location=(0.0, 1.0),
                                            blocked_dof=(1,2))
    
    boundary_conditions.add(dirichlet_BC_right)

    # Add Neumann boundary condition at the right edge
    neumann_BC_top = PointNeumannBC(location=(0.0,0.5),
                                      force_vector=(-0.25, 0.0),
                                      )
    
    boundary_conditions.add(neumann_BC_top)

    # Create the problem instance
    problem = Design_IOH_Wrapper_Instanced(
        nmmcsx= num_mmc,
        nmmcsy= 2,
        nelx= NELX,
        nely= NELY,
        instance= instance,
        volfrac= 0.5,
        symmetry_condition= True,
        scalation_mode="unitary",
        plot_topology=plot_topology,
        plot_variables=plot_stresses,
        cost_function= "compliance",
        use_sparse_matrices= True,
        continuity_check_mode= "discrete",
        run_= run_number,
        boundary_conditions_list= boundary_conditions,
        problem_aux_name = "michell_truss",
        Emin= 1e-4,
        plot_modifier_dict = {
            "rotate": True,  # Rotate the plot for better visualization
            "rotate_angle": 90,  # Rotate by 90 degrees
        },
        standard_weight=1500.0,
    )

    return problem


    
    