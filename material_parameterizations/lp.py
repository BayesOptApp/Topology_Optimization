'''
This is a dedicated module for definition of the lamination parameters
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

'''

### IMPORTS ###
import math
import numpy as np



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



def CurveInterpolation(w1:float,V3_1:float,V3_2:float,VR:float)->list:
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

    #V1_arc,V3_arc = calculate_points_on_arc_segment(V3_1,V3_2,VR)

    NP,E_P = setup_lamination_parameters(NE,nelx,nely,symmetry_cond)

    xy_1:np.ndarray = xy_1_fun(length,height)
    xy_2:np.ndarray = xy_2_fun(length,height)

    # Initialize arrays to contain elemental lam. par.s
    #c = np.zeros((len(V3_arc),3))
    #c[:,0] = 1.0
    #c[:,1] = np.transpose(np.linspace(0.0,1.0,len(V3_arc)))

    if V3_1 == V3_2:
        # If the two points are the same, then we can just return the same value
        V1_e[:] = V1_1_fun(VR,V3_1)
        V3_e[:] = V3_1
        return V1_e,V3_e

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