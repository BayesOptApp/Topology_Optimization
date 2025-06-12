'''
Reinterpretation of code for Structural Optimisation
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

'''

# import libraries
import numpy as np
from typing import List, Tuple, Union
from finite_element_solvers.four_point_quadrature_plane_stress_composite import Mesh, CompositeMaterialMesh
#from finite_element_solvers import select_2D_quadrature_solver

from meshers.MeshGrid2D import MeshGrid2D
from boundary_conditions import BoundaryConditionList
from material_parameterizations.lpim import LaminationMasterNode
# Import plotting functions
from utils.Helper_Plots import plotNodalVariables, plotNodalVariables_pyvista
from utils.Helper_Plots import plot_LP_Parameters, plot_LP_Parameters_pyvista

from typing import Optional, Union
# Import the LP functions
#from material_parameterizations.lp import compute_elemental_lamination_parameters
from material_parameterizations.lpim import compute_elemental_lamination_parameters


''' Constants '''

# Meshgrid properties constants
ELEMENT_LENGTH_DEFAULT:float = 1.0 # Default length of Finite 2D Plate Element
ELEMENT_HEIGHT_DEFAULT:float = 1.0 # Default height of Finite 2D Plate Element

THICKNESS_DEFAULT:float = 1.0 # Default Material Thickness
RHO_DEFAULT:float = 1.0 # Default Material Density

PENALTY_FACTOR_DEFAULT:float = 1e20 # Default Penalty Factor

COST_FUNCTIONS:tuple = ("mean displacement","compliance")


'''
End of Constants
'''
 
'''
ADDITIONAL (HELPER) FUNCTIONS
'''

def evaluate_FEA(TO_mat:np.ndarray,iterr:int,
                 sample:int,volfrac:float,Emin:float,E0:float,run_:int,
                 boundary_conditions:BoundaryConditionList,
                 material_properties_dict:dict,
                 penalty_factor:float=PENALTY_FACTOR_DEFAULT,
                 plotVariables:bool=False,
                 sparse_matrices_solver:bool=False,pyvista_plot=True,
                 cost_function:str = COST_FUNCTIONS[0],
                 plot_modifier_dict:Optional[dict]=None,
                 **kwargs)->float:
    
    '''
    Method to evaluate the cost function of a design given by some parameter

    Inputs:
    - x: (1 x 3) Array with lamination parameters
    - TO_mat: Density Mapping of the design
    - iterr: Current iteration of optimisation loop
    - penalty_factor: factor determined to penalize bad or unfeasible designs
    - symmetry_cond: handle informing if the symmetry condition is active for the design
    - sparse_matrices_solver: handle used to determine if using the FE solver with sparse matrices. This is useful for very large systems
    - Emin:
    - E0: 
    - pyvista_plot: Call pyvista to draw the plots
    - cost_function: A string defining the cost function. Set "mean displacement" to compute the cost function based on the mean displacement, or "compliance" to define the cost function based on the average compliance of the design.
    '''

    # Check the entry on the cost function
    if cost_function not in COST_FUNCTIONS:
        raise ValueError("The cost function set is not allowed")
    
    
    # Get length and height of the elements based on density matrix
    l:float = TO_mat.shape[1]
    h:float = TO_mat.shape[0]
    
    # Generate the mesh object
    mesh:Mesh = Mesh(boundary_conditions_list=boundary_conditions,
                     length=l,
                     height=h,
                     element_length=ELEMENT_LENGTH_DEFAULT,
                     element_height=ELEMENT_HEIGHT_DEFAULT,
                     sparse_matrices=sparse_matrices_solver,
                     E11=material_properties_dict["E11"],
                     E22=material_properties_dict["E22"],
                     G12=material_properties_dict["G12"],
                     nu12=material_properties_dict["nu12"])
    
    # Reshape the density matrix into a vector
    #density_vec:np.ndarray = np.rot90(TO_mat).reshape((1,mesh.MeshGrid.nel_total),order='F')
    density_vec:np.ndarray = TO_mat.reshape((1,mesh.MeshGrid.nel_total),order='C')
    
     # Extract the number of elements
    nelx:int = mesh.MeshGrid.nelx
    nely:int = mesh.MeshGrid.nely
    
    mesh.set_matrices(density_vec,THICKNESS_DEFAULT,RHO_DEFAULT,E0,Emin)

   
    # Compute the displacements and other metrics
    u,u_mean,u_tip = mesh.compute_displacements()

    # Reshape the displacements
    u_r:np.ndarray = u.reshape((-1,2))

    # N_static - Calculate the deformed global node matrix
    N_static:np.ndarray = np.array([mesh.MeshGrid.coordinate_grid[:,0],
                                    mesh.MeshGrid.coordinate_grid[:,1]+u_r[:,0],
                                    mesh.MeshGrid.coordinate_grid[:,2]+u_r[:,1]])
    N_static = N_static.T 

    # Compute the cost function
    if cost_function == COST_FUNCTIONS[0]:
        part_sum = np.sum((TO_mat>Emin))
        cost:float = u_mean + penalty_factor*max(0.0, part_sum-nelx*nely*volfrac)
    elif cost_function == COST_FUNCTIONS[1]:
        comp_vec = mesh.mesh_compute_compliance(disp=u,density_vector=density_vec,
                                                thickness=THICKNESS_DEFAULT,
                                                E0=E0,Emin=Emin)

        #Manipulate compliance
        ce:np.ndarray = comp_vec.reshape((mesh.MeshGrid.nely,mesh.MeshGrid.nelx),order='F')
        # Compute compliance value
        compliance = np.sum(ce)

        cost:float = compliance + penalty_factor*max(0.0, np.sum((TO_mat>Emin))-
                                                    mesh.MeshGrid.nelx*mesh.MeshGrid.nely*volfrac)

    if (np.all((np.abs(u_r) < 50)) and plotVariables):
        # Retrieve stresses and strains from the displacements
        list_of_vars = mesh.mesh_retrieve_Strain_Stress(
                                                        density_vector=density_vec,
                                                        disp=u)  
        # Identify the corresponding stresses and strains
        #epsxxN: = list_of_vars[0]
        #epsyyN = list_of_vars[1]
        #epsxyN = list_of_vars[2]
        #epsxxE = list_of_vars[3]
        #epsyyE = list_of_vars[4]
        #epsxyE = list_of_vars[5]
        #sigxxN = list_of_vars[6]
        #sigyyN = list_of_vars[7]
        #sigxyN = list_of_vars[8]
        vonMisesN = list_of_vars[9]
        #sigxxE = list_of_vars[10]
        #sigyyE = list_of_vars[11]
        #sigxyE = list_of_vars[12]
        #vonMisesE = list_of_vars[13]

        # Stress contours
        mat_ind = (density_vec>Emin)

        if not pyvista_plot:
            plotNodalVariables(cost=cost,N_static=N_static,element_map=mesh.MeshGrid.E,
                            mat_ind = mat_ind, nodal_variable= vonMisesN,
                                iterr=iterr,sample=sample,run_=run_)
             
        else:
            plotNodalVariables_pyvista(cost=cost,N_static=N_static,element_map=mesh.MeshGrid.E,
                            mat_ind = mat_ind, nodal_variable= vonMisesN,
                                iterr=iterr,sample=sample,run_=run_,
                                plot_modifier_dict=plot_modifier_dict)



    

    return cost

def evaluate_FEA_LP(x:np.ndarray,TO_mat:np.ndarray,iterr:int,
                 sample:int,volfrac:float,Emin:float,E0:float,run_:int,
                 boundary_conditions:BoundaryConditionList,
                 material_properties_dict:dict,
                 interpolation_points:List[Tuple[Union[float,int], Union[float,int]]],
                 penalty_factor:float=PENALTY_FACTOR_DEFAULT,
                 plotVariables:bool=False,
                 symmetry_cond:bool=True,
                 sparse_matrices_solver:bool=False,
                 pyvista_plot=True,
                 cost_function:str = COST_FUNCTIONS[0],
                 mode:str="TO",
                 plot_modifier_dict:Optional[dict]=None,
                 **kwargs)->float:
    
    '''
    Method to evaluate the cost function of a design given by some parameter

    Inputs:
    - x: (1 x 3) Array with lamination parameters
    - TO_mat: Density Mapping of the design
    - iterr: Current iteration of optimisation loop
    - penalty_factor: factor determined to penalize bad or unfeasible designs
    - symmetry_cond: handle informing if the symmetry condition is active for the design
    - sparse_matrices_solver: handle used to determine if using the FE solver with sparse matrices. This is useful for very large systems
    - Emin:
    - E0: 
    - pyvista_plot: Call pyvista to draw the plots
    - cost_function: A string defining the cost function. Set "mean displacement" to compute the cost function based on the mean displacement, or "compliance" to define the cost function based on the average compliance of the design.
    '''

    # Check the entry on the cost function
    if cost_function not in COST_FUNCTIONS:
        raise ValueError("The cost function set is not allowed")
    
    x = x.flatten()
    
    VR:float = x[0]
    
    # Get length and height of the elements based on density matrix
    l:float = TO_mat.shape[1]
    h:float = TO_mat.shape[0]

    
    # Generate the mesh object
    mesh:Mesh = CompositeMaterialMesh(boundary_conditions_list=boundary_conditions,
                                      length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
                                      element_height=ELEMENT_HEIGHT_DEFAULT,
                                      sparse_matrices=sparse_matrices_solver,
                                      E11=material_properties_dict["E11"],
                                      E22=material_properties_dict["E22"],
                                      G12=material_properties_dict["G12"],
                                      nu12=material_properties_dict["nu12"])
    
    # Reshape the density matrix into a vector
    #density_vec:np.ndarray = np.rot90(TO_mat).reshape((1,mesh.MeshGrid.nel_total),order='F')
    density_vec:np.ndarray = TO_mat.reshape((1,mesh.MeshGrid.nel_total),order='C')
    
     # Extract the number of elements
    nelx:int = mesh.MeshGrid.nelx
    nely:int = mesh.MeshGrid.nely

    # Construct a list of lamination master nodes
    lamination_master_nodes:List[LaminationMasterNode] = list()
    for ii in range(len(interpolation_points)):
        # Create a new lamination master node
        lmn:LaminationMasterNode = LaminationMasterNode(x=interpolation_points[ii][0]*nelx,
                                                        y=interpolation_points[ii][1]*nely,
                                                        V3=x[ii+1])
        # Append the lamination master node to the list
        lamination_master_nodes.append(lmn)

    V1_e,V3_e = compute_elemental_lamination_parameters(mesh.MeshGrid,
                                                        master_nodes_list=lamination_master_nodes,
                                                        VR=VR,
                                                        symmetry_cond=symmetry_cond,
    )
    
    mesh.set_matrices(density_vec,V1_e,V3_e,THICKNESS_DEFAULT,RHO_DEFAULT,E0,Emin)

   
    # Compute the displacements and other metrics
    u,u_mean,u_tip = mesh.compute_displacements()

    # Reshape the displacements
    u_r:np.ndarray = u.reshape((-1,2))

    # N_static - Calculate the deformed global node matrix
    N_static:np.ndarray = np.array([mesh.MeshGrid.coordinate_grid[:,0],
                                    mesh.MeshGrid.coordinate_grid[:,1]+u_r[:,0],
                                    mesh.MeshGrid.coordinate_grid[:,2]+u_r[:,1]])
    N_static = N_static.T 

    # Compute the cost function
    if cost_function == COST_FUNCTIONS[0]:
        part_sum = np.sum((TO_mat>Emin))
        cost:float = u_mean + penalty_factor*max(0.0, part_sum-nelx*nely*volfrac)
    elif cost_function == COST_FUNCTIONS[1]:
        comp_vec = mesh.mesh_compute_compliance(disp=u,
                                                density_vector=density_vec,
                                                V1_e=V1_e,
                                                V3_e=V3_e,
                                                thickness=THICKNESS_DEFAULT,
                                                E0=E0,Emin=Emin)

        #Manipulate compliance
        ce:np.ndarray = comp_vec.reshape((mesh.MeshGrid.nely,mesh.MeshGrid.nelx),order='F')
        # Compute compliance value
        compliance = np.sum(ce)

        cost:float = compliance + penalty_factor*max(0.0, np.sum((TO_mat>Emin))-
                                                    mesh.MeshGrid.nelx*mesh.MeshGrid.nely*volfrac)

    if (np.all((np.abs(u_r) < 100)) and plotVariables):
        # Retrieve stresses and strains from the displacements
        list_of_vars = mesh.mesh_retrieve_Strain_Stress(V1_e=V1_e,
                                                        V3_e=V3_e,
                                                        density_vector=density_vec,
                                                        disp=u)  
        # Identify the corresponding stresses and strains
        #epsxxN: = list_of_vars[0]
        #epsyyN = list_of_vars[1]
        #epsxyN = list_of_vars[2]
        #epsxxE = list_of_vars[3]
        #epsyyE = list_of_vars[4]
        #epsxyE = list_of_vars[5]
        #sigxxN = list_of_vars[6]
        #sigyyN = list_of_vars[7]
        #sigxyN = list_of_vars[8]
        vonMisesN = list_of_vars[9]
        #sigxxE = list_of_vars[10]
        #sigyyE = list_of_vars[11]
        #sigxyE = list_of_vars[12]
        #vonMisesE = list_of_vars[13]

        # Stress contours
        mat_ind = (density_vec>Emin)

        if not pyvista_plot:
            plotNodalVariables(cost=cost,N_static=N_static,element_map=mesh.MeshGrid.E,
                            mat_ind = mat_ind, nodal_variable= vonMisesN,
                                iterr=iterr,sample=sample,run_=run_,
                                plot_modifier_dict=plot_modifier_dict)
            
            if mode.find("LP") != -1:
            
                plot_LP_Parameters(cost=cost,N_static=mesh.MeshGrid.coordinate_grid,
                            element_map=mesh.MeshGrid.E,
                            NN=mesh.MeshGrid.grid_point_number_total,
                            NN_l=mesh.MeshGrid.grid_point_number_X,
                            NN_h=mesh.MeshGrid.grid_point_number_Y,
                            mat_ind = mat_ind, V1_e = V1_e,V3_e=V3_e,
                                iterr=iterr,sample=sample,run_=run_)
            
        else:
            plotNodalVariables_pyvista(cost=cost,N_static=N_static,element_map=mesh.MeshGrid.E,
                            mat_ind = mat_ind, nodal_variable= vonMisesN,
                                iterr=iterr,sample=sample,run_=run_)
            
            if mode.find("LP") != -1:
                plot_LP_Parameters_pyvista(cost=cost,N_static=mesh.MeshGrid.coordinate_grid,
                                element_map=mesh.MeshGrid.E,
                                NN=mesh.MeshGrid.grid_point_number_total,
                                NN_l=mesh.MeshGrid.grid_point_number_X,
                                NN_h=mesh.MeshGrid.grid_point_number_Y,
                                mat_ind = mat_ind, V1_e = V1_e,V3_e=V3_e,
                                    iterr=iterr,sample=sample,run_=run_,
                                    plot_modifier_dict=plot_modifier_dict)

    

    return cost



def return_element_midpoint_positions(TO_mat:np.ndarray,Emin:float,E0:float):
    '''
    This function returns the positions of the midpoints of the devised mesh.

    ----------------
    Inputs:
    - TO_mat: topology indicating density/material distribution
    - Emin: Setting of the Ersatz Material; to be numerically close to 0
    - E0: Setting the Material interpolator (close to 1)
    '''

    # TODO: Generate a new mesh (apply some sparsity to not generate a large matrix space)
    # Get length and height of the elements based on density matrix
    l:float = TO_mat.shape[1]
    h:float = TO_mat.shape[0]
    
    # Generate the mesh object
    mesh:CompositeMaterialMesh = CompositeMaterialMesh(length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
                     element_height=ELEMENT_HEIGHT_DEFAULT,
                     sparse_matrices=True)
    
    # Call the meshgrid function with the array of midpoints
    midpoints:np.ndarray = mesh.MeshGrid.compute_element_midpoints()

    return midpoints


def compute_number_of_joined_bodies(TO_mat:np.ndarray,Emin:float,E0:float)-> int:
    '''
    This function returns the positions the number of joined bodies. This points out
    which solutions are unfeasible as the beams are not totally connected

    ----------------
    Inputs:
    - TO_mat: topology indicating density/material distribution
    - Emin: Setting of the Ersatz Material; to be numerically close to 0
    - E0: Setting the Material interpolator (close to 1)
    '''
        
    # Get length and height of the elements based on density matrix
    l:float = TO_mat.shape[1]
    h:float = TO_mat.shape[0]
    
    # Generate the mesh object
    mesh:CompositeMaterialMesh = CompositeMaterialMesh(length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
                     element_height=ELEMENT_HEIGHT_DEFAULT,
                     sparse_matrices=True)
    
    # Reshape the density matrix into a vector
    #density_vec:np.ndarray = np.rot90(TO_mat).reshape((1,mesh.MeshGrid.nel_total),order='F')
    density_vec:np.ndarray = TO_mat.reshape((1,mesh.MeshGrid.nel_total),order='C')
    
    # Get an ordered array with the element_numbering:
    ordered_arr:np.ndarray = np.arange(0, mesh.MeshGrid.nel_total)

    # Get the elements where there is material
    material_elems:np.ndarray = ordered_arr[np.where(np.abs(density_vec.ravel()-E0)<1e-12)]

    # Array to store the already picked material elements
    set_C:list = list()
    # Generate an array to store the material elements belonging to a set
    set_A:list = list()

    # Generate an array to store the sets of bodies
    set_bodies:list = list()

    # Kickstart the algorithm with some random index
    curIdx:int = np.random.randint(0,np.size(material_elems))

    # Start a while loop and stop until all the list if material elements is exhausted
    while True:
        
        # Call the recursive function
        append_mass_elements_recursive(curIdx,set_A,set_C,material_elems,mesh.MeshGrid.find_neighbouring_elements_quad)

        # Attach the nodes list to the body list
        set_bodies.append(set_A)

        # Delete the nodes stored in A
        set_A = list()

        # Check possible indices to lookup
        possible_choices = np.ravel(material_elems[np.logical_not(np.isin(material_elems,set_C))])

        if len(possible_choices) >= 1:
            # Get the new index
            curIdx = possible_choices[np.random.randint(0,possible_choices.size)]
        else:
            # Break the loop
            break
    
    # Return the number of joined bodies and the array
    return len(set_bodies),set_bodies


def append_mass_elements_recursive(idx:int,set_A_tot:list,set_C_tot:list,
                                    set_mat_elems:np.ndarray,find_neighbours_function)->None:
    r"""
    This is a recursive function to check the neighbours of a given element index and then attach to the respective
    sets.

    Args
    ----------------------------
    
    - idx: Integer with a pointer to an element of the mesh
    - set_A_tot: the list with all the current stored element indices of the n-th body
    - set_C_total: the list of all used previous material indices
    - set_mat_elems: the list of all elements which are not empty or not "Ersatz Material"
    - find_neighbours_function: a function which finds the neighbours of the given element (received as a parameter)
    """

    # Step 0: Append the index to the list
    set_A_tot.append(idx)
    set_C_tot.append(idx)

    # Step 1: Get all the neighbours of the given element
    neighs:np.ndarray = find_neighbours_function(idx)

    # Step 2: From these neighbours, get all the neighbours which are material neighbours
    material_neighs = neighs[np.isin(neighs,set_mat_elems)]

    # Step 3: Get the neighbors that are not in the taken/used list
    if len(set_C_tot) < 1:
        possible_choices = material_neighs.copy()
    else:
        possible_choices = material_neighs[np.logical_not(np.isin(material_neighs,set_C_tot))]

    # Step 4: Use the function recursively by looping all over the possible choices
    if possible_choices.size >=1:
        for idxs in possible_choices:
            append_mass_elements_recursive(idxs,set_A_tot,set_C_tot,set_mat_elems,find_neighbours_function)
    else:
        return
    

def compute_number_of_joined_bodies_2(TO_mat:np.ndarray,Emin:float,E0:float)-> int:
    '''
    This function returns the positions the number of joined bodies. This points out
    which solutions are unfeasible as the beams are not totally connected

    
    Args:
    ----------------
    - TO_mat: topology indicating density/material distribution
    - Emin: Setting of the Ersatz Material; to be numerically close to 0
    - E0: Setting the Material interpolator (close to 1)
    '''
        
    # Get length and height of the elements based on density matrix
    l:float = TO_mat.shape[1]
    h:float = TO_mat.shape[0]
    
    # Generate the mesh object
    mesh:CompositeMaterialMesh = CompositeMaterialMesh(length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
                     element_height=ELEMENT_HEIGHT_DEFAULT,
                     sparse_matrices=True)
    
    # Reshape the density matrix into a vector
    #density_vec:np.ndarray = np.rot90(TO_mat).reshape((1,mesh.MeshGrid.nel_total),order='F')
    density_vec:np.ndarray = TO_mat.reshape((1,mesh.MeshGrid.nel_total),order='C')
    

    # Get an ordered array with the element_numbering:
    ordered_arr:np.ndarray = np.ravel(np.arange(0, mesh.MeshGrid.nel_total))

    # Get the elements where there is material
    material_elems:np.ndarray = ordered_arr[np.where(np.abs(density_vec.ravel()-E0)<1e-12)]

    # Generate an array to store if the element has been visited
    visited:np.ndarray = np.zeros_like(material_elems,dtype=bool)

    # Generate an array to store the material elements belonging to a set
    set_A:list = list()

    # Generate an array to store the sets of bodies
    set_bodies:list = list()

    for idx,elem_idx in enumerate(material_elems):
        if not visited[idx]:
            append_mass_elements_iterative(idx=idx, elem_idx=elem_idx,visited_list=visited,
                                           set_A_tot=set_A,material_elem_list= material_elems,
                                           find_neighbours_function= mesh.MeshGrid.find_neighbouring_elements_quad
                                           )
            
            # Append the bodies
            set_bodies.append(set_A)

            #Clear the list of bodies
            set_A = list()

    # Return the number of joined bodies and the array
    return len(set_bodies),set_bodies
    



def append_mass_elements_iterative(idx:int, 
                                   elem_idx:int, 
                                   material_elem_list:np.ndarray,
                                   visited_list:np.ndarray, 
                                   set_A_tot:list,
                                   find_neighbours_function)->None:
    
    # Initialize the stack
    stack = [(idx,elem_idx)]

    while len(stack) > 0:
        curIdx, cur_elem_idx = stack.pop()

        if visited_list[curIdx]:
            continue

        # Mark the element as visited
        visited_list[curIdx] = True

        # Store the element in the list
        set_A_tot.append(cur_elem_idx)

        # Get all the neighbours idxs
        neighs:np.ndarray = find_neighbours_function(cur_elem_idx)

        # From these neighbours, get all the neighbours which are material neighbours
        material_neighs = np.sort(neighs[np.isin(neighs,material_elem_list)])

        # Get the indices referenced to the material element list
        idxs = np.ravel(np.argwhere(np.isin(material_elem_list,material_neighs)))

        # Add all the neighbours to the stack
        for ii in range(idxs.size):
            stack.append(( idxs[ii],material_neighs[ii]))





'''
TODO: This function should be modified to interact with other nodal
variables of interest

def plot_nodal_variables(nodX:np.ndarray,TO_mat:np.ndarray,Emin:float,E0:float,):
    """
    This function is added to plot variables on a mesh given a resulting variable of interest defined on each node of the
    mesh.

    Inputs:
    - nodX: np.ndarray -> (m*n) x 1 vector which stores the values of the nodal variable at each node of the mesh
    - TO_mat: np.ndarray -> TO_mat: Density Mapping of the design
    - Emin: float -> Maximum value of average elasticity
    - E0: float -> Minimum value of average elasticity (greater than 0 to define gaps)
    """

    a = 1


'''


'''
END OF ADDITIONAL (HELPER) FUNCTIONS
'''



    
    

