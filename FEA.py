'''
Reinterpretation of code for Structural Optimisation
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

'''

# import libraries
import numpy as np
import math
from FEM import Mesh
from MeshGrid2D import MeshGrid2D

# Import plotting functions
from Helper_Plots import plotNodalVariables, plotNodalVariables_pyvista
from Helper_Plots import plot_LP_Parameters, plot_LP_Parameters_pyvista

'''
DEFAULT LAMBDA FUNCTIONS
'''

# Locations of the master nodes 
xy_1_fun = lambda le,he: np.array([0.0*le,0.0*he])
xy_2_fun = lambda le,he: np.array([1.0*le,0.5*he])

# Functions computing V1_1 and V1_2
V1_1_fun = lambda VR,V3_1: (2*VR-1)*math.sqrt((V3_1+1)/2) 
V1_2_fun = lambda VR,V3_2: (2*VR-1)*math.sqrt((V3_2+1)/2) 


'''
END OF DEFAULT LAMBDA FUNCTIONS
'''


''' Constants '''
#------------------------------VOLUMETRIC RATIOS -----------------------------------------------------
VR_OPT:float = 0.0 # Volumetric ratio for the right sides of Miki's diagram
V3_1_OPT:float = 1.0 # V3 value of the first master node
V3_2_OPT:float = -1.0 # V3 value of the second master node

# Computation of V1's
V1_1_OPT:float = V1_1_fun(VR_OPT,V3_1_OPT) # V1 value of the first master node
V1_2_OPT:float = V1_2_fun(VR_OPT,V3_2_OPT) # V1 value of the second master node

DV_DEFAULT:float = 0.001

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

def CurveInterpolation(w1:float,V3_1:float,V3_2:float,VR:float)->list:
    '''

    Function which performs curve interpolation according to Miki's diagram
    
    Inputs:
    - w1:   Ratio of the dist. from the 1st point to the dist. sum
        0 gives exactly the 1st point
        1 gives exactly the 2nd point
    - V3_1: V3 of the 1st master node
    - V3_2: V3 of the 2nd master node
    - VR:   Volumetric ratio of the layers

    Outputs:
    V1 and V3 of the current point
    '''

    #Step size for the search
    dy:float = DV_DEFAULT

    c:float = 2*VR-1

    y1:float = V3_1
    y2:float = V3_2

    # Initialise output
    V1:float = 0.0
    V3:float = 0.0

    # w1=0 gives exactly the 1st point
    if abs(w1) < 1e-12:
        V3:float = y1
        V1:float = c*math.sqrt((V3+1)/2)
    # w1=1 gives exactly the 2nd point
    elif abs(w1-1)< 1e-12:
        V3:float = y2
        V1:float = c*math.sqrt((V3+1)/2)
    else:
        if y1 == -1:
            y1 = -0.999999

        if y2 == -1:
            y2 = -0.999999
        
        if y2 > y1:
            y1y2 = 1/(8*math.sqrt(c**2+8*y2+8))*math.sqrt((c**2+8*y2+8)/(y2+1))*(2*math.sqrt(2)*(y2+1)*math.sqrt(c**2+8*y2+8)+c**2*math.sqrt(y2+1)*math.log(math.sqrt(2)*math.sqrt(c**2+8*y2+8)+4*math.sqrt(y2+1)))- 1/(8*math.sqrt(c**2+8*y1+8))*math.sqrt((c**2+8*y1+8)/(y1+1))*(2*math.sqrt(2)*(y1+1)*math.sqrt(c**2+8*y1+8)+c**2*math.sqrt(y1+1)*math.log(math.sqrt(2)*math.sqrt(c**2+8*y1+8)+4*math.sqrt(y1+1)))
            diff_old = 1000
            #V3 = y1
            #V1 = c*math.sqrt((V3+1)/2)
            for y in np.arange(y1,y2,dy):
                yy1 = 1/(8*math.sqrt(c**2+8*y+8))*math.sqrt((c**2+8*y+8)/(y+1))*(2*math.sqrt(2)*(y+1)*math.sqrt(c**2+8*y+8)+c**2*math.sqrt(y+1)*math.log(math.sqrt(2)*math.sqrt(c**2+8*y+8)+4*math.sqrt(y+1)))-1/(8*math.sqrt(c**2+8*y1+8))*math.sqrt((c**2+8*y1+8)/(y1+1))*(2*math.sqrt(2)*(y1+1)*math.sqrt(c**2+8*y1+8)+c**2*math.sqrt(y1+1)*math.log(math.sqrt(2)*math.sqrt(c**2+8*y1+8)+4*math.sqrt(y1+1)))
                diff = abs((yy1/y1y2)-w1); # difference from the target
                # Check convergencce
                if diff > diff_old:
                    break
                diff_old = diff
                V3 = y
                V1 = c*math.sqrt((V3+1)/2)

        elif y2 < y1:
            diff_old = 1000
            #V3 = y1;
            #V1 = c*sqrt((V3+1)/2);
            y1y2 = 1/(8*math.sqrt(c**2+8*y1+8))*math.sqrt((c**2+8*y1+8)/(y1+1))*(2*math.sqrt(2)*(y1+1)*math.sqrt(c**2+8*y1+8)+c**2*math.sqrt(y1+1)*math.log(math.sqrt(2)*math.sqrt(c**2+8*y1+8)+4*math.sqrt(y1+1)))- 1/(8*math.sqrt(c**2+8*y2+8))*math.sqrt((c**2+8*y2+8)/(y2+1))*(2*math.sqrt(2)*(y2+1)*math.sqrt(c**2+8*y2+8)+c**2*math.sqrt(y2+1)*math.log(math.sqrt(2)*math.sqrt(c**2+8*y2+8)+4*math.sqrt(y2+1)))
            for y in np.arange(y1,y2,-dy):
                yy1 = 1/(8*math.sqrt(c**2+8*y1+8))*math.sqrt((c**2+8*y1+8)/(y1+1))*(2*math.sqrt(2)*(y1+1)*math.sqrt(c**2+8*y1+8)+c**2*math.sqrt(y1+1)*math.log(math.sqrt(2)*math.sqrt(c**2+8*y1+8)+4*math.sqrt(y1+1)))- 1/(8*math.sqrt(c**2+8*y+8))*math.sqrt((c**2+8*y+8)/(y+1))*(2*math.sqrt(2)*(y+1)*math.sqrt(c**2+8*y+8)+c**2*math.sqrt(y+1)*math.log(math.sqrt(2)*math.sqrt(c**2+8*y+8)+4*math.sqrt(y+1)))
                diff = abs((yy1/y1y2)-w1) # difference from the target

                # Check convergence
                if diff > diff_old:
                    break
                diff_old = diff
                V3 = y
                V1 = c*math.sqrt((V3+1)/2)
        elif y2 == y1:
            V3 = y1
            V1 = c*math.sqrt((V3+1)/2)

    return V1,V3



def setup_lamination_parameters(NE:int,nelx:int,
                                nely:int,
                                symmetry_cond:bool)->list:
    '''
    Setup Parameters for Drawing Lamination Parameters.

    Inputs:
    - NE: Total number of finite elements
    - nelx: total number of finite elements in x-direction
    - nely: total number of elements in y-direction
    - symmetry_cond: Application of symmetry condition
    '''


    NP:int = math.ceil(NE/2)
    # The elements sharing the same property (in each row)
    # Symmetric points wrt vertical and horizontal axes passing through the
    # center, as we divided the domain in 2
    # Size NP x 2
    E_P = []
    n_m = 0
    j = nelx-1
    k = nelx*(nely-1)

    for ii in range(NP):
        
        # Increase counter
        n_m = n_m+1 
            
        #Go to the upper row in the rectangle
        if math.remainder(ii+1,nelx) == 0:
            E_P.append([n_m-1,n_m+k-1])
            j = nelx-1
            k = k-2*nelx
            #n_m = n_m+NE_l/2;
            continue

        # Master node at bottom-left
        E_P.append([n_m-1,n_m+k-1])
        j = j-2
    
    E_P:np.ndarray = np.array(E_P)


    return NP,E_P

    


def calculate_points_on_arc_segment(V3_1:float = V3_1_OPT, 
                                    V3_2:float = V3_2_OPT,
                                    VR:float = VR_OPT,
                                    dV3:float=DV_DEFAULT)->list:
    '''
    Calculate the points on the arc segment - discretization of the arch non
    the Miki's diagram
    '''
    

    if V3_1 > V3_2:
        V3_arc:np.ndarray = np.arange(V3_1,V3_2,-dV3)
        V1_arc:np.ndarray = (2*VR-1)*np.sqrt((V3_arc+1)/2)
    elif V3_1 < V3_2:
        V3_arc:np.ndarray = np.arange(V3_1,V3_2,dV3)
        V1_arc:np.ndarray = (2*VR-1)*np.sqrt((V3_arc+1)/2)
    else:
        V3_arc:np.ndarray = np.array([V3_2])
        V1_arc:np.ndarray = (2*VR-1)*np.sqrt((V3_arc+1)/2)

    return V1_arc,V3_arc

def compute_elemental_lamination_parameters(NE:int,nelx:int,nely:int,
                                            E:np.ndarray,V3_1:float,
                                            V3_2:float,VR:float,N:np.ndarray,
                                            length:float,height:float,
                                            symmetry_cond:bool)->list:

    '''
    Function to compute the elemental lamination parameters.

    Inputs:
    - NE: Total number of elements of Finite Element Mesh
    - nelx: Total number of elements in x-direction
    - nely: Total number of elements in y-direction
    - V3_1: Lamination parameter V3_1
    - V3_2: Lamination parameter V3_2
    - VR: Lamination parameter VR
    - N: Array with position of the nodes of the finite element mesh
    - length: Length of the element
    - height: height of the element
    - symmetry_cond: boolean variable managing if the symmetry condition is "on"
    
    
    '''
    V1_e = np.zeros((NE,1))
    V3_e = np.zeros((NE,1))

    V1_arc,V3_arc = calculate_points_on_arc_segment(V3_1,V3_2,VR)

    NP,E_P = setup_lamination_parameters(NE,nelx,nely,symmetry_cond)

    xy_1:np.ndarray = xy_1_fun(length,height)
    xy_2:np.ndarray = xy_2_fun(length,height)

    # Initialize arrays to contain elemental lam. par.s
    c = np.zeros((len(V3_arc),3))
    c[:,0] = 1.0
    c[:,1] = np.transpose(np.linspace(0.0,1.0,len(V3_arc)))

    #Calculate elemental angles
    # Loop for each property
    for p in range(NP):
        
        # Element in the lower left rectangle
        # (The other 3 element will be mirrored from
        # this one using elemental property matrix)
        e = E_P[p,0]
        
        #Element center in global coordinates
        x_C = np.mean([N[E[e,1],1],N[E[e,2],1],N[E[e,3],1],N[E[e,4],1]])
        y_C = np.mean([N[E[e,1],2],N[E[e,2],2],N[E[e,3],2],N[E[e,4],2]])
                    
        #Distance from the 1st point
        d1 = math.sqrt((xy_1[0]-x_C)**2 + (xy_1[1]-y_C)**2)
        #Distance from the 2nd point
        d2 = math.sqrt((xy_2[0]-x_C)**2 + (xy_2[1]-y_C)**2)
        
        # Weights of the points on the current element
        # 0 gives exactly the 1st point
        # 1 gives exactly the 2nd point
        # Weight of 1st point
        w1:float = d1/(d1+d2)
        
        # Calculate elemental lamination parameters
        V1,V3 = CurveInterpolation(w1,V3_1,V3_2,VR)
        
        V1_e[E_P[p,0]] = V1
        V3_e[E_P[p,0]] = V3

        # Mirror the lam. par. values for the symmetric elements
        # and store them
        
        V1_e[E_P[p,1]] = V1   
        V3_e[E_P[p,1]] = V3


    return V1_e,V3_e

def evaluate_FEA(x:np.ndarray,TO_mat:np.ndarray,iterr:int,
                 sample:int,volfrac:float,Emin:float,E0:float,run_:int,
                 penalty_factor:float=PENALTY_FACTOR_DEFAULT,
                 plotVariables:bool=False,symmetry_cond:bool=True,
                 sparse_matrices_solver:bool=False,pyvista_plot=True,
                 cost_function:str = COST_FUNCTIONS[0],
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
    
    V3_1:float = x[1]
    V3_2:float = x[2]
    VR:float = x[0]
    
    # Get length and height of the elements based on density matrix
    l:float = TO_mat.shape[1]
    h:float = TO_mat.shape[0]
    
    # Generate the mesh object
    mesh:Mesh = Mesh(length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
                     element_height=ELEMENT_HEIGHT_DEFAULT,
                     sparse_matrices=sparse_matrices_solver)
    
    # Reshape the density matrix into a vector
    #density_vec:np.ndarray = np.rot90(TO_mat).reshape((1,mesh.MeshGrid.nel_total),order='F')
    density_vec:np.ndarray = TO_mat.reshape((1,mesh.MeshGrid.nel_total),order='C')
    
     # Extract the number of elements
    nelx:int = mesh.MeshGrid.nelx
    nely:int = mesh.MeshGrid.nely

    V1_e,V3_e = compute_elemental_lamination_parameters(mesh.MeshGrid.nel_total,
                                                        nelx,
                                                        nely,
                                                        mesh.MeshGrid.E,
                                                        V3_1,V3_2,VR,
                                                        mesh.MeshGrid.coordinate_grid,
                                                        l,h,
                                                        symmetry_cond)
    
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
        comp_vec = mesh.mesh_compute_compliance(disp=u,density_vector=density_vec,
                                                V1_e=V1_e,V3_e=V3_e,thickness=THICKNESS_DEFAULT,
                                                E0=E0,Emin=Emin)

        #Manipulate compliance
        ce:np.ndarray = comp_vec.reshape((mesh.MeshGrid.nely,mesh.MeshGrid.nelx),order='F')
        # Compute compliance value
        compliance = np.sum(ce)

        cost:float = compliance + penalty_factor*max(0.0, np.sum((TO_mat>Emin))-
                                                    mesh.MeshGrid.nelx*mesh.MeshGrid.nely*volfrac)

    if (np.all((np.abs(u_r) < 50)) and plotVariables):
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
                                iterr=iterr,sample=sample,run_=run_)
            
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

            plot_LP_Parameters_pyvista(cost=cost,N_static=mesh.MeshGrid.coordinate_grid,
                            element_map=mesh.MeshGrid.E,
                            NN=mesh.MeshGrid.grid_point_number_total,
                            NN_l=mesh.MeshGrid.grid_point_number_X,
                            NN_h=mesh.MeshGrid.grid_point_number_Y,
                            mat_ind = mat_ind, V1_e = V1_e,V3_e=V3_e,
                                iterr=iterr,sample=sample,run_=run_)

    

    return cost



def compute_objective_function(x:np.ndarray,TO_mat:np.ndarray,iterr:int,
                 sample:int,Emin:float,E0:float,run_:int,
                 plotVariables:bool=False,symmetry_cond:bool=True,
                 sparse_matrices_solver:bool=False, 
                 cost_function:str=COST_FUNCTIONS[0],
                 pyvista_plot=True)-> float:
    '''
    Method to evaluate the objective function of a design given by some parameter

    -----------
    Inputs:
    - x: (1 x 3) Array with lamination parameters
    - TO_mat: Density Mapping of the design
    - iterr: Current iteration of optimisation loop
    - symmetry_cond: handle informing if the symmetry condition is active for the design
    - sparse_matrices_solver: handle used to determine if using the FE solver with sparse matrices. This is useful for very large systems
    - Emin: Setting of the Ersatz Material; to be numerically close to 0
    - E0: Setting the Material interpolator (close to 1)
    - cost_function: A string defining the cost function. Set "mean displacement" to compute the cost function based on the mean displacement, or "compliance" to define the cost function based on the average compliance of the design.
    - pyvista_plot: Handler to allow plotting with PyVista instead of the default Matplotlib, which is usually faster.

    Returns
    -----------
    - cost: Some floating point value with the physical cost evaluated
    '''
    
    # Check the entry on the cost function
    if cost_function not in COST_FUNCTIONS:
        raise ValueError("The cost function set is not allowed")
    
    x = x.flatten()
    
    V3_1:float = x[1]
    V3_2:float = x[2]
    VR:float = x[0]
    
    # Get length and height of the elements based on density matrix
    l:float = TO_mat.shape[1]
    h:float = TO_mat.shape[0]
    
    # Generate the mesh object
    mesh:Mesh = Mesh(length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
                     element_height=ELEMENT_HEIGHT_DEFAULT,
                     sparse_matrices=sparse_matrices_solver)
    
    # Reshape the density matrix into a vector
    #density_vec:np.ndarray = np.rot90(TO_mat).reshape((1,mesh.MeshGrid.nel_total),order='F')
    density_vec:np.ndarray = TO_mat.reshape((1,mesh.MeshGrid.nel_total),order='C')
    
    
     # Extract the number of elements
    nelx:int = mesh.MeshGrid.nelx
    nely:int = mesh.MeshGrid.nely

    V1_e,V3_e = compute_elemental_lamination_parameters(mesh.MeshGrid.nel_total,
                                                        nelx,
                                                        nely,
                                                        mesh.MeshGrid.E,
                                                        V3_1,V3_2,VR,
                                                        mesh.MeshGrid.coordinate_grid,
                                                        l,h,
                                                        symmetry_cond)
    
    mesh.set_matrices(density_vec,V1_e,V3_e,THICKNESS_DEFAULT,RHO_DEFAULT,E0,Emin)

   
    # Compute the displacements and other metrics
    u,u_mean,_ = mesh.compute_displacements()
    
    # Compute the cost function
    if cost_function == COST_FUNCTIONS[0]:
        cost:float = u_mean 
    elif cost_function == COST_FUNCTIONS[1]:
        comp_vec = mesh.mesh_compute_compliance(disp=u,density_vector=density_vec,
                                                V1_e=V1_e,V3_e=V3_e,thickness=THICKNESS_DEFAULT,
                                                E0=E0,Emin=Emin)

        #Manipulate compliance
        ce:np.ndarray = comp_vec.reshape((mesh.MeshGrid.nely,mesh.MeshGrid.nelx),order='F')
        # Compute compliance value
        compliance = np.sum(ce)

        cost:float = compliance 
        
    
    # This part is meant for plotting purposes
    # -----------------------------------------------------------------------
    # N_static - Calculate the deformed global node matrix
    # Reshape the displacements
    u_r:np.ndarray = u.reshape((-1,2))
    N_static:np.ndarray = np.array([mesh.MeshGrid.coordinate_grid[:,0],
                                    mesh.MeshGrid.coordinate_grid[:,1]+u_r[:,0],
                                    mesh.MeshGrid.coordinate_grid[:,2]+u_r[:,1]])
    N_static = N_static.T 
    
    if (np.all((np.abs(u_r) < 50)) and plotVariables):
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
                                iterr=iterr,sample=sample,run_=run_)
            
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

            plot_LP_Parameters_pyvista(cost=cost,N_static=mesh.MeshGrid.coordinate_grid,
                            element_map=mesh.MeshGrid.E,
                            NN=mesh.MeshGrid.grid_point_number_total,
                            NN_l=mesh.MeshGrid.grid_point_number_X,
                            NN_h=mesh.MeshGrid.grid_point_number_Y,
                            mat_ind = mat_ind, V1_e = V1_e,V3_e=V3_e,
                                iterr=iterr,sample=sample,run_=run_)
    
    
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
    mesh:Mesh = Mesh(length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
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
    mesh:Mesh = Mesh(length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
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
    """
    This is a recursive function to check the neighbours of a given element index and then attach to the respective
    sets.
    ----------------------------
    Inputs:
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
    mesh:Mesh = Mesh(length=l,height=h,element_length=ELEMENT_LENGTH_DEFAULT,
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
    



def append_mass_elements_iterative(idx:int, elem_idx:int, material_elem_list:np.ndarray,
                                   visited_list:np.ndarray, set_A_tot:list,find_neighbours_function)->None:
    
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



    
    

