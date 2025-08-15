from typing import Optional, Union, Tuple, List
import numpy as np
from material_parameterizations.lp import V1_1_fun, V3_1_OPT, V3_2_OPT, VR_OPT, DV_DEFAULT
from meshers.MeshGrid2D import MeshGrid2D
import math
from dataclasses import dataclass
from scipy.optimize import brentq



r"""
This section is an implementation of the interpolation points as some class.
This is intended to be used as a parameterization for the lamination parameters."""

@dataclass
class LaminationMasterNode:
    r"""
    A class representing a lamination master node with x, y coordinates and V3 value.

    Attributes
    ----------
    -   x (`float`): The x-coordinate of the lamination master node.
    -   y (`float`): The y-coordinate of the lamination master node.
    -   V3 (`float`): The V3 value of the lamination master node, which should be between -1 and 1.
    """

    def __init__(self, x: float, y: float, V3: float):
        r"""Initialize a Lamination Master Node with x, y coordinates and V3 value."""
        self.x = x
        self.y = y
        self.V3 = V3

    ### Representation Methods
    def __str__(self):
        return f"Lamination Parameter Point(x={self.x}, y={self.y}, V1_1={self.V3})"

    def __repr__(self):
        return f"LaminationParameterPoint(x={self.x}, y={self.y}, V1_1={self.V3})"

    # Convert to a tuple for easier handling
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.V3)
    
    def compute_distance_to_arbitrary_point(self, x: float, y: float) -> float:
        """Compute the Euclidean distance to an arbitrary point (x, y)."""
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
    
    def compute_equivalent_V1(self, VR: float) -> float:
        """Compute the equivalent V1 values based on the V3 value and VR."""

        assert VR >= 0 and VR <= 1, "VR must be between 0 and 1"

        return (2*VR-1)*np.sqrt((self.V3+1)/2)
        
    
    ### Property Methods
    @property
    def x(self) -> float:
        """Get the x-coordinate of the lamination parameter point."""
        return self._x
    
    @x.setter
    def x(self, value: float)->None:
        """Set the x-coordinate of the lamination parameter point."""
        if not isinstance(value, (int, float)):
            raise TypeError("x must be a number")
        
        #if value < 0 or value > 1:
        #    raise ValueError("x must be between 0 and 1")
        
        self._x = value
    
    @property
    def y(self) -> float:
        """Get the y-coordinate of the lamination parameter point."""
        return self._y
    
    @y.setter
    def y(self, value: float)->None:
        """Set the y-coordinate of the lamination parameter point."""
        if not isinstance(value, (int, float)):
            raise TypeError("x must be a number")
        
        #if value < 0 or value > 1:
        #    raise ValueError("x must be between 0 and 1")
        
        self._y = value

    @property
    def V3(self) -> float:
        return self._V3
    
    @V3.setter
    def V3(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError("V3 must be a number")
        
        if value < -1 or value > 1:
            raise ValueError("V3 must be between -1 and 1")
        
        self._V3 = value




def CurveInterpolation(w1:float,V3_1:float,V3_2:float,VR:float)->float:
    '''

    Function which performs curve interpolation according to Miki's diagram
    
    Inputs:
    - w1:   Ratio of the dist. from the 1st point to the dist. 
        - sum 0 gives exactly the 1st point
        - 1 gives exactly the 2nd point
    - V3_1: V3 of the 1st master node
    - V3_2: V3 of the 2nd master node
    - VR:   Volumetric ratio of the layers

    Outputs:
    V3 of the current point
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
        #V1:float = c*math.sqrt((V3+1)/2)
    # w1=1 gives exactly the 2nd point
    elif abs(w1-1)< 1e-12:
        V3:float = y2
        #V1:float = c*math.sqrt((V3+1)/2)
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
                #V1 = c*math.sqrt((V3+1)/2)

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
                #V1 = c*math.sqrt((V3+1)/2)
        elif y2 == y1:
            V3 = y1
            #V1 = c*math.sqrt((V3+1)/2)

    return V3


def CurveInterpolation2(w1:float,V3_1:float,V3_2:float,VR:float)->float:
    '''

    Function which performs curve interpolation according to Miki's diagram 

    This function is implemented in a way that it is more efficient than the previous one.
    
    Inputs:
    - w1:   Ratio of the dist. from the 1st point to the dist. 
        - sum 0 gives exactly the 1st point
        - 1 gives exactly the 2nd point
    - V3_1: V3 of the 1st master node
    - V3_2: V3 of the 2nd master node
    - VR:   Volumetric ratio of the layers

    Outputs:
    V3 of the current point
    '''

    #Step size for the search
    dy:float = DV_DEFAULT

    c:float = 2*VR-1

    y1:float = V3_1
    y2:float = V3_2

    # Initialise output
    V1:float = 0.0
    V3:float = 0.0

    def F(y: float) -> float:
        """The integral function from Miki's formula."""
        term1 = math.sqrt(c**2 + 8*y + 8)
        return (1 / (8 * term1)) * math.sqrt((c**2 + 8*y + 8)/(y + 1)) * (
            2 * math.sqrt(2) * (y + 1) * term1 +
            c**2 * math.sqrt(y + 1) * math.log(math.sqrt(2) * term1 + 4 * math.sqrt(y + 1))
        )
    
    # Edge cases
    if abs(w1) < 1e-12:
        return V3_1
    if abs(w1 - 1) < 1e-12:
        return V3_2

    # Clamp y values if they hit singularity
    y1 = max(V3_1, -0.999999)
    y2 = max(V3_2, -0.999999)

    # Determine direction of integration
    F1 = F(y1)
    F2 = F(y2)

    target = F1 + w1 * (F2 - F1)

    def residual(y):
        return F(y) - target

    # Solve for V3 by root finding in interval [y1, y2] or [y2, y1]
    try:
        V3 = brentq(residual, y1, y2) if y2 > y1 else brentq(residual, y2, y1)
    except ValueError:
        # Fallback to closest endpoint
        V3 = y1 if abs(w1) < 0.5 else y2

    return V3


def setup_lamination_parameters(NE:int,
                                nelx:int,
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
    E_P:list = []
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
    r'''
    Calculate the points on the arc segment - discretization of the arc on
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



def compute_elemental_lamination_parameters(mesh_grid:MeshGrid2D,
                                            master_nodes_list:List[LaminationMasterNode],
                                            VR:float,
                                            symmetry_cond:bool,
                                            interpolation_function:Optional[int]=1)->list:

    '''
    Function to compute the elemental lamination parameters.

    Args
    ------------------
    - mesh_grid: `MeshGrid2D` object containing the mesh grid parameters
    - master_nodes_list: List of `LaminationMasterNode` objects representing the master nodes
    - VR: Volumetric ratio of the layers
    - symmetry_cond: boolean variable managing if the symmetry condition is "on"
    - interpolation_function: Optional integer to select the interpolation function (default is 1)

    Returns
    ------------------
    - V1_e: Array of V1 values for each element
    - V3_e: Array of V3 values for each element
    '''

    assert isinstance(mesh_grid, MeshGrid2D), "mesh_grid must be an instance of MeshGrid2D"
    assert isinstance(master_nodes_list, list), "master_nodes_list must be a list of LaminationMasterNode objects"

    assert len(master_nodes_list) >= 2, "master_nodes_list must contain at least two LaminationMasterNode objects"

    # Import itertools for combinations
    from itertools import combinations


    # Extract mesh grid parameters
    NE = mesh_grid.nel_total
    nelx = mesh_grid.nelx
    nely = mesh_grid.nely

    #N = mesh_grid.coordinate_grid
    #E = mesh_grid.E

    V1_e = np.zeros((NE,1))
    V3_e = np.zeros((NE,1))

    #V1_arc,V3_arc = calculate_points_on_arc_segment(V3_1,V3_2,VR)

    NP,E_P = setup_lamination_parameters(NE,nelx,nely,symmetry_cond)

    # Get the list of midpoints of the mesh grid
    mid_points = mesh_grid.compute_element_midpoints()


    # Initialize arrays to contain elemental lam. par.s
    #c = np.zeros((len(V3_arc),3))
    #c[:,0] = 1.0
    #c[:,1] = np.transpose(np.linspace(0.0,1.0,len(V3_arc)))

    # Extract the V3 values from all the master nodes
    V3_array = np.array([node.V3 for node in master_nodes_list])

    if np.all(V3_array == V3_array[0]):
        # If the two points are the same, then we can just return the same value
        V1_e[:] = V1_1_fun(VR,V3_array[0])
        V3_e[:] = V3_array[0]
        return V1_e,V3_e

    #Calculate elemental angles
    # Loop for each property
    for p in range(NP):
        
        # Element in the lower left rectangle
        # (The other 3 element will be mirrored from
        # this one using elemental property matrix)
        #e = E_P[p,0]

        
        
        #Element center in global coordinates
        x_C = mid_points[p,1]
        y_C = mid_points[p,2]

        # Compute the distances of the master nodes to the current element center
        # Initialize lists to store the weights and partial V1, V3 values
        # This will be used to compute the V1 and V3 values
        # for the current element based on the master nodes
        # and the distances to the current element center

        dist_arr = np.array([node.compute_distance_to_arbitrary_point(x_C, y_C) for node in master_nodes_list])

    

        if np.any(dist_arr < 1e-12):
            # If any of the master nodes is at the same point as the element center,
            # we can just use that node's V3 value
            idx = np.argmin(dist_arr)
            V3 = master_nodes_list[idx].V3
            V1 = (2*VR-1)*np.sqrt((V3+1)/2)

        else:

            partial_V3_list = []

            weights_arr = []
            # Loop through combinations of master nodes to define some V3_1 and V3_2
            for i, j in combinations(range(len(master_nodes_list)), 2):

                #Distance from the 1st point
                d1 = dist_arr[i]
                #Distance from the 2nd point
                d2 = dist_arr[j]

                v3_1 = master_nodes_list[i].V3
                v3_2 = master_nodes_list[j].V3
            
                # Weights of the points on the current element
                # 0 gives exactly the 1st point
                # 1 gives exactly the 2nd point
                # Weight of 1st point
                w1:float = d1/(d1+d2)
            
                # Calculate elemental lamination parameters
                v3 = CurveInterpolation(w1,v3_1,v3_2,VR) if interpolation_function == 1 else CurveInterpolation2(w1,v3_1,v3_2,VR)
                # Store the partial V1 and V3 values
                partial_V3_list.append(v3)

                # Get the distances which are not indexed by either i or j
                other_indices = [k for k in range(len(master_nodes_list)) if k != i and k != j]

                # Sum the distances of the other master nodes
                other_distances = np.sum(dist_arr[other_indices])

                # Calculate the weight for the current combination
                weights_arr.append(other_distances/ np.sum(dist_arr))


            if len(weights_arr) == 1 and weights_arr[0] == 0.0:
                # The loop before accounts for cases where there are more than 2 master nodes,
                # but if there is only two master nodes, the weights_arr will be 0.
                weights_arr = np.array([1.0])

            # Combine the partial V3 values using the partial v3 list and the weights
            V3 = np.sum(np.array(partial_V3_list) * np.array(weights_arr))
            
            # Calculate the corresponding V1 value using the V3 value
            V1 = (2*VR-1)*np.sqrt((V3+1)/2)

        # Store the computed V1 and V3 values in the elemental arrays
        V1_e[E_P[p,0]] = V1
        V3_e[E_P[p,0]] = V3

        # Mirror the lam. par. values for the symmetric elements
        # and store them
        
        V1_e[E_P[p,1]] = V1   
        V3_e[E_P[p,1]] = V3


    return V1_e,V3_e

def compute_angle_distribution(V3_e:np.ndarray,
                               mesh_grid:MeshGrid2D,
                               symmetry_condition:bool)->Tuple[np.ndarray, np.ndarray]:
    '''
    Function to compute the angle distribution of the lamination parameters.

    Args
    ------------------
    - V3_e: Array of V3 values for each element
    - mesh_grid: `MeshGrid2D` object containing the mesh grid parameters
    - symmetry_condition: `bool` object denoting if there's symmetry condition around x-axis.

    Returns
    ------------------
    - theta_l: Array of left angles for each element
    - theta_r: Array of right angles for each element
    '''

    assert isinstance(mesh_grid, MeshGrid2D), "mesh_grid must be an instance of MeshGrid2D"

    # Extract number of elements from the mesh grid
    NE = mesh_grid.nel_total

    # Initialize angles array
    theta_l = np.zeros((NE, 1))
    theta_r = np.zeros_like(theta_l)

    if not symmetry_condition:

        # Calculate angles based on V1 values
        for e in range(NE):
            V1_pos = np.sqrt(0.5*(V3_e[e]+1.0))
            V1_neg = -np.sqrt(0.5*(V3_e[e]+1.0))
            theta_l[e] = 0.5*np.arccos(V1_pos)
            theta_r[e] = 0.5*np.arccos(V1_neg)
    else:
        NP, EP = setup_lamination_parameters(NE, mesh_grid.nelx, mesh_grid.nely, symmetry_condition)

        # Calculate angles based on V1 values for symmetric elements
        for p in range(NP):
            e = EP[p, 0]
            V1_pos = np.sqrt(0.5*(V3_e[e]+1.0))
            V1_neg = -np.sqrt(0.5*(V3_e[e]+1.0))
            theta_l[e] = 0.5 * np.arccos(V1_pos)
            theta_r[e] = 0.5 * np.arccos(V1_neg)

            # Mirror the angles for the symmetric elements
            theta_l[EP[p, 1]] = -theta_l[e]
            theta_r[EP[p, 1]] = -theta_r[e]
        

    return theta_l, theta_r