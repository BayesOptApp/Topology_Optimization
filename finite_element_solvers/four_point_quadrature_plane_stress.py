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
from scipy import io,sparse
from scipy import linalg as scilinalg
from meshers.MeshGrid2D import MeshGrid2D
from typing import Union, List, Tuple, Optional
from .common import *



def compute_in_plane_C_matrix(E11:float,E22:float,G12:float,nu12:float)-> np.ndarray:
    r'''
    Compute the in-plane C-matrix for the Finite Element Method
    
    Args
    -----
        - E11: Young Modulus in 11 direction
        - E22: Young Modulus in 22 direction
        - G12: Torsional Modulus in 12 direction
        - nu12: Poisson's Ratios
    '''
    
    # Compute the material invariants
    u1,_,_,u4,u5 = compute_material_invariants(E11=E11,E22=E22,nu12=nu12,G12=G12)

    # C matrix entities
    c11:float = u1
    c12:float = u4
    c22:float = u1
    c33:float = u5
    
    # Assemble the in-plane stiffness matrix C from V1 and V3
    C_mat:np.ndarray = np.array([[c11, c12, 0  ],
                                 [c12, c22, 0  ],
                                 [0  , 0  , c33]])

    return C_mat


def apply_BC(K:np.ndarray,F:np.ndarray,NN:int,NN_l:int,NNDOF:int)->list:
    '''
    Function to apply the boundary conditions on Matrices.

    Inputs:
    - K: Global Stiffness Matrix
    - F: Global Force Vector
    - NN: Total number of Nodes
    - NN_l: Number of nodes in x-direction
    - NNDOF: Total Number of Degrees of Freedom
    '''

    # Array to contain node IDs where BCs are specified
    BCiN =[]
    NBcN:int = 0

    # Inserting displacement boundary conditions by modifying K
    # For modal and modal FR analyses the DoFs are removed from K, M, F
    # For static and direct FR analyses K and F rows modified as
    # K[i] = [0 0 ... 1 ... 0 0 0] and F(i) = 0

    for ii in range(NN):
        # ID for every NGDOF DoF (starts from 0)
        iNNDOF = ii*NNDOF
        
        # Cantilever BC
        # If the node is on the left edge
        if (np.fmod(ii+1,NN_l) == 1):
            for jj in range(NNDOF):
                    K[iNNDOF+jj,:] = 0.0
                    #K[:,iNNDOF+jj] = 0.0
                    K[iNNDOF+jj,iNNDOF+jj] = 1.0
                    F[iNNDOF+jj] = 0.0

            # Node IDs where BCs are specified
            BCiN.append(ii)
            NBcN = NBcN+1
    
    return np.array(BCiN),NBcN

def apply_BC_sparse(K:sparse.lil_matrix,F:sparse.lil_matrix,NN:int,NN_l:int,NNDOF:int)->list:
    '''
    Function to apply the boundary conditions on Matrices.

    Inputs:
    - K: Global Stiffness Matrix
    - F: Global Force Vector
    - NN: Total number of Nodes
    - NN_l: Number of nodes in x-direction
    - NNDOF: Total Number of Degrees of Freedom
    '''

    # Array to contain node IDs where BCs are specified
    BCiN =[]
    NBcN:int = 0

    # Inserting displacement boundary conditions by modifying K
    # For modal and modal FR analyses the DoFs are removed from K, M, F
    # For static and direct FR analyses K and F rows modified as
    # K[i] = [0 0 ... 1 ... 0 0 0] and F(i) = 0

    for ii in range(NN):
        # ID for every NGDOF DoF (starts from 0)
        iNNDOF = ii*NNDOF
        
        # Cantilever BC
        # If the node is on the left edge
        if (np.fmod(ii+1,NN_l) == 1):
            for jj in range(NNDOF):
                    K[iNNDOF+jj,:] = 0.0
                    #K[:,iNNDOF+jj] = 0.0
                    K[iNNDOF+jj,iNNDOF+jj] = 1.0
                    F[iNNDOF+jj] = 0.0

            # Node IDs where BCs are specified
            BCiN.append(ii)
            NBcN = NBcN+1
    
    return np.array(BCiN),NBcN


def retrieve_Strain_Stress(NN:int,NN_l:int,NN_h:int,E:np.ndarray,
                                    NE:int,u:np.ndarray,density_vector:np.ndarray,
                                    E11:float,E22:float,G12:float,nu12:float,
                                    dSdxy:np.ndarray,
                                    E0:float=1.00,
                                    Emin:float=1.00E-09)->list:
    r'''
    Function to generate the contours of strains and stresses of corresponding designs 

    Args:
    -------------------------------
    - NN: Total number of nodes
    - NN_l: Number of nodes in x-direction
    - NN_h: Number of nodes in y-direction
    - E: Element Freedom table
    - NE: Total Number of Elements
    - u: Vector of displacements of all DOFs
    - density_vector: density of all the finite elements in vector form.
    - E11: Young's Modulus in Direction 1,1
    - E22: Young's Modulus in Direction 2,2
    - G12: Torsional Modulus in Direction 1,2
    - nu12: Poisson's ratio in direction 1,2
    - dSdxy: Derivative of Shape functions with respect to x and y
    - Emin: Definition of void multiplier
    - E0: Definition of material multiplier
    '''
    
    # Initialize nodal strains
    epsxxN:np.ndarray = np.zeros((NN,1))
    epsyyN:np.ndarray = np.zeros((NN,1))
    epsxyN:np.ndarray = np.zeros((NN,1))
    
    # Initialize elemental strains
    epsxxE:np.ndarray = np.zeros((NE,1))
    epsyyE:np.ndarray = np.zeros((NE,1))
    epsxyE:np.ndarray = np.zeros((NE,1))

    # Initialize nodal stresses
    sigxxN:np.ndarray = np.zeros((NN,1))
    sigyyN:np.ndarray = np.zeros((NN,1))
    sigxyN:np.ndarray = np.zeros((NN,1))
    vonMisesN:np.ndarray = np.zeros((NN,1))
    
    # Initialize elemental stresses
    sigxxE:np.ndarray = np.zeros((NE,1))
    sigyyE:np.ndarray = np.zeros((NE,1))
    sigxyE:np.ndarray = np.zeros((NE,1))
    vonMisesE:np.ndarray = np.zeros((NE,1))
    
    sigN:np.ndarray = np.zeros((NN*3,1))
    
    GaussToCorner:np.ndarray = np.array([[1+math.sqrt(3)/2,  -1/2,  1-math.sqrt(3)/2, -1/2],
                     [-1/2,            1+math.sqrt(3)/2,   -1/2,  1-math.sqrt(3)/2],
                     [1-math.sqrt(3)/2,  -1/2,  1+math.sqrt(3)/2,  -1/2    ],
                     [-1/2,   1-math.sqrt(3)/2,  -1/2,  1+math.sqrt(3)/2]])
    
    C_mat_material = compute_in_plane_C_matrix(E11,E22,G12,nu12)
    C_mat_void = np.array([[1,1,0],[1,1,0],[0,0,1]])*Emin

    # Loop over each element
    for el in range(NE):
        
        #Find the grids of the element
        N1 = E[el,1]
        N2 = E[el,2]
        N3 = E[el,3]
        N4 = E[el,4]
        
        # Element property
        Ep = E[el,5]

        if abs(density_vector[0,el] - E0) <=  1e-12:
            C_mat:np.ndarray = C_mat_material.copy()
        else:
            C_mat:np.ndarray = C_mat_void.copy()
        
        Ne = np.array([N1,N2,N3,N4]).reshape((4,))
        
        # Get the nodal displacements
        u_uv = []

        for n in range(NUMBER_OF_NODES_X_ELEMENT):
            iN:int = Ne[n]
            iNNNDOF:int = int(iN*NUMBER_OF_NODAL_DOF)
            rang:np.ndarray = np.arange(start=iNNNDOF,stop=iNNNDOF+2)
            u_uv.append(u[rang])

        u_uv:np.ndarray = np.array(u_uv).reshape((-1,1))
    
        #Calculate the nodal stresses at each Gauss point
        # Loop over each Gauss quadrature point
        #eppsqp = np.zeros((3,4))
        eppsqp = np.array([])
        siggqp = np.array([])

        for gqp in range(4):

            # Assemble the B_matrix  
            B_mat:np.ndarray = part_B_matrix(dSdxy=dSdxy,gqp=gqp)
            
            #Sigma = C*B*u;
            # x = x1*S1 + x2*S2 + x3*S3 + x4*S4
            #  y = y1*S1 + y2*S2 + y3*S3 + y4*S4

            # Compute the Strains
            part = np.dot(B_mat,u_uv).reshape((3,-1))

             # Compute the Stresses
            part2 = np.dot(C_mat , part)

            # Attach to vectors
            if gqp == 0:
                eppsqp = part
                siggqp = part2
            else:
                eppsqp = np.hstack((eppsqp,part))
                siggqp = np.hstack((siggqp,part2))

        
        # Extrapolate the stresses to the corner nodes and save them as 'Sigma'       
        epsN = GaussToCorner @ np.transpose(eppsqp)
        sigN = GaussToCorner @ np.transpose(siggqp)

        vmN = np.sqrt(np.power(sigN[:,0],2)-np.multiply(sigN[:,0],sigN[:,1])+np.power(sigN[:,1],2)+3*np.power(sigN[:,2],2))
        
        
        epsxxE[el] = np.mean(epsN[:,0])
        epsyyE[el] = np.mean(epsN[:,1])
        epsxyE[el] = np.mean(epsN[:,2])
        
        sigxxE[el] = np.mean(sigN[:,0])
        sigyyE[el] = np.mean(sigN[:,1])
        sigxyE[el] = np.mean(sigN[:,2])
        vonMisesE[el] = np.mean(vmN)
        
        # Apply averaging where multiple elements share the node
        for n in range(4):
            
            iN = Ne[n]
            
            # If the grid is on a corner
            if ((iN == 0) or (iN == NN_l-1) or (iN == NN-NN_l) or (iN == NN-1)):
                
                sigxxN[iN] = sigxxN[iN]+sigN[n,0]
                sigyyN[iN] = sigyyN[iN]+sigN[n,1]
                sigxyN[iN] = sigxyN[iN]+sigN[n,2]
                vonMisesN[iN] = vonMisesN[iN]+vmN[n]

                epsxxN[iN] = epsxxN[iN]+epsN[n,0]
                epsyyN[iN] = epsyyN[iN]+epsN[n,1]
                epsxyN[iN] = epsxyN[iN]+epsN[n,2]
                
            # If the grid is on an edge
            elif (np.fmod(iN,NN_l) == 0) or ((0<iN and iN<NN_l-1)) or (np.fmod(iN+1,NN_l) == 0) or ((NN-NN_l<iN and iN<NN-1)):
                
                sigxxN[iN] = sigxxN[iN]+sigN[n,0]/2
                sigyyN[iN] = sigyyN[iN]+sigN[n,1]/2
                sigxyN[iN] = sigxyN[iN]+sigN[n,2]/2
                vonMisesN[iN] = vonMisesN[iN]+vmN[n]/2

                epsxxN[iN] = epsxxN[iN]+epsN[n,0]/2
                epsyyN[iN] = epsyyN[iN]+epsN[n,1]/2
                epsxyN[iN] = epsxyN[iN]+epsN[n,2]/2
                
            #If the grid is at the interior
            else:              
                sigxxN[iN] = sigxxN[iN]+sigN[n,0]/4
                sigyyN[iN] = sigyyN[iN]+sigN[n,1]/4
                sigxyN[iN] = sigxyN[iN]+sigN[n,2]/4
                vonMisesN[iN] = vonMisesN[iN]+vmN[n]/4

                epsxxN[iN] = epsxxN[iN]+epsN[n,0]/4
                epsyyN[iN] = epsyyN[iN]+epsN[n,1]/4
                epsxyN[iN] = epsxyN[iN]+epsN[n,2]/4
    
    
    return epsxxN, epsyyN, epsxyN, epsxxE, epsyyE, epsxyE, sigxxN, sigyyN, sigxyN, vonMisesN, sigxxE, sigyyE, sigxyE, vonMisesE

class Mesh:
    r'''
    Mesh Class definition
    '''
    def __init__(self,E11:float=E11_DEFAULT,E22:float=E22_DEFAULT,
                 G12:float=G12_DEFAULT,nu12:float=NU12_DEFAULT,
                 length:float=LENGTH_DEFAULT,height:float=HEIGHT_DEFAULT,
                 element_length:float=ELEMENT_LENGTH_DEFAULT,
                 element_height:float=ELEMENT_HEIGHT_DEFAULT,
                 SRI:bool=True, sparse_matrices:bool=False) -> None:
        
        # Build the meshgrid
        self.__mesh_grid:MeshGrid2D = MeshGrid2D(length=length,height=height,
                                      element_length=element_length,element_height=element_height)
        
        self.__sparse_matrices:bool = sparse_matrices
        # Set the material properties
        self.__E11:float = E11
        self.__E22:float = E22
        self.__G12:float = G12
        self.__nu12:float = nu12

        # Initialise Global Matrices for finite element method
        self._non_zero_matrices:bool = False

        if not self.sparse_matrices:
            self.__K:np.ndarray = np.zeros((self.MeshGrid.grid_point_number_total*NUMBER_OF_NODAL_DOF,
                                            self.MeshGrid.grid_point_number_total*NUMBER_OF_NODAL_DOF))
            self.__M:np.ndarray = np.zeros_like(self.__K)
            self.__F:np.ndarray = np.zeros((self.MeshGrid.grid_point_number_total*NUMBER_OF_NODAL_DOF,
                                            1))
        
        else:
            self.__fill_sparse_pattern__(nn=self.MeshGrid.grid_point_number_total*NUMBER_OF_NODAL_DOF)
        
        # initialise the shape functions
        self.__initialise_shape_functions__()

        # compute and initialise the Jacobian
        self.__initialise_jacobian__(SRI=SRI)
    
    def __initialise_shape_functions__(self)->None:

        self.__S1:np.ndarray = np.zeros((4,1))
        self.__S1[:,0]= shape(GQ_POINT_1[0])

        self.__dSdksieta1:np.ndarray = np.zeros((2,4,1))
        self.__dSdksieta1[:,:,0] = gradshape(GQ_POINT_1[0])

        self.__S4:np.ndarray = np.zeros((4,4))
        self.__dSdksieta4:np.ndarray = np.zeros((2,4,4))

        for gqp in range(4):


            self.__S4[:,gqp] = shape(GQ_POINT_4[gqp])

            self.__dSdksieta4[:,:,gqp] = gradshape(GQ_POINT_4[gqp])

    
    def __initialise_jacobian__(self,SRI:bool)->None:

        '''
        Calculates the Jacobian matrix, its inverse and determinant for the first
        element only (because all elements are identical) at all GQ points. Also
        evaluates and stores derivatives of shape functions wrt global
        coordinates x and y.

        ### Inputs:
        -   SRI: Determine if perform Selective Reduced Integration
        '''
        # To calculate Jacobian matrix of an element we need E_Nxy matrix of
        # size NEN=4*2. Each row of it stores x and y coordinates of the nodes
        # of an element.

        self.__E_Nxy:np.ndarray = np.zeros((NUMBER_OF_NODES_X_ELEMENT,2))
        element_data_matrix:np.ndarray = self.__mesh_grid.E
        xy_coords:np.ndarray = self.__mesh_grid.coordinate_grid

        self.__dSdxy4:np.ndarray = np.zeros((1,2,4,4))

        for ii in range (NUMBER_OF_NODES_X_ELEMENT):
            iN = element_data_matrix[0,ii+1] # Elemental grid IDs are stored at E's 2nd to 5th col.
            self.__E_Nxy[ii,0] = xy_coords[iN,1]; # x-coordinates are stored at N's 2nd col.
            self.__E_Nxy[ii,1] = xy_coords[iN,2]; # y-coordinates are stored at N's 3rd col.

        # For each GQ point calculate 2*2 Jacobian matrix, its inverse and its
        # determinant. Also calculate derivatives of shape functions wrt global
        # coordinates x and y. These are the derivatives that we'll use in
        # evaluating K and F integrals.

        self.__detJacob4:np.ndarray = np.zeros((1,4))
        self.__detJacob1:np.ndarray = np.zeros((1,1))

        for gqp in range(4):  # GQ loop
            #
            # |  dS  |   |  dx       dy  | |  dS  |
            # | ---- |   | ----     ---- | | ---- |
            # | dksi |   | dksi     dksi | |  dx  |
            # |      | = |               | |      |
            # |  dx  |   |  dx       dy  | |  dS  |
            # | ---- |   | ----     ---- | | ---- |
            # | deta |   | deta     deta | |  dy  |
            #
            #                    J
            #
            #  x = x1*S1 + x2*S2 + x3*S3 + x4*S4
            #  y = y1*S1 + y2*S2 + y3*S3 + y4*S4
            #
            # Then the Jacobian is:
            #
            #                                 | x1  y1 |
            #                                 |        |
            #     |  dS1   dS2   dS3   dS4  | |        |
            #     |  dksi  dksi  dksi  dksi | |        |
            # J = |                         | |        |
            #     |  dS1   dS2   dS3   dS4  | | x3  y3 |
            #     | ----- ----- ----- ----- | |        |
            #     |  deta  deta  deta  deta | |        |
            #                                 | x4  y4 |
            #
            self.__jacob:np.ndarray = self.__dSdksieta4[:,:,gqp] @ self.__E_Nxy
            self.__invJacob:np.ndarray = np.linalg.inv(self.__jacob)
            self.__detJacob4[0,gqp] = np.linalg.det(self.__jacob)  # We'll need this during GQ integration.

            self.__dSdxy4[0,:,:,gqp] = self.__invJacob[:,:] @ self.__dSdksieta4[:,:,gqp]
        

        if SRI:
            self.__jacob = self.__dSdksieta1[:,:,0] @ self.__E_Nxy
            self.__invJacob = np.linalg.inv(self.__jacob)
            self.__detJacob1[0,0] = np.linalg.det(self.__jacob)  # We'll need this during GQ integration.

            self.__dSdxy1:np.ndarray = np.zeros((1,2,4,1))
            self.__dSdxy1[0,:,:,0] = self.__invJacob[:,:] @ self.__dSdksieta1[:,:,0]

    
    def __fill_sparse_pattern__(self,nn:int)->None:
        '''
        This function helps filling the sparse pattern for sparse matrices in order
        to save computation time in the assembly of the matrices

        Inputs
            - nn: Size of the system
        '''

        # Create lists to save all the possible combination of indices of the vectors
        list_ii_arr:list = []

        # Create lists to save all the possible combination of indices of the matrices
        list_ii:list = []
        list_jj:list = []

        # Start looping all over the elements
        for elem in range(self.MeshGrid.nel_total):
            # Start looping for all the nodes per element
            for ii in range(NUMBER_OF_NODES_X_ELEMENT):
                # Get the nodal position referring to the element
                iN = int(self.MeshGrid.E[elem,ii+1])
                for jj in range(NUMBER_OF_NODAL_DOF):
                    # Add the row index of the vector list
                     list_ii_arr.append(iN*NUMBER_OF_NODAL_DOF+jj)
                
                for jj in range(NUMBER_OF_NODES_X_ELEMENT):
                    jN = int(self.MeshGrid.E[elem,jj+1])
                    list_ii.append((iN)*NUMBER_OF_NODAL_DOF)
                    list_ii.append((iN)*NUMBER_OF_NODAL_DOF)
                    list_ii.append((iN)*NUMBER_OF_NODAL_DOF+1)
                    list_ii.append((iN)*NUMBER_OF_NODAL_DOF+1)

                    list_jj.append((jN)*NUMBER_OF_NODAL_DOF)
                    list_jj.append((jN)*NUMBER_OF_NODAL_DOF+1)
                    list_jj.append((jN)*NUMBER_OF_NODAL_DOF)
                    list_jj.append((jN)*NUMBER_OF_NODAL_DOF+1)
        
        # Convert the lists of arrays
        arr_ii_arr:np.ndarray = np.array(list_ii_arr)

        arr_ii:np.ndarray = np.array(list_ii)
        arr_jj:np.ndarray = np.array(list_jj)

        # Delete the lists
        del list_jj, list_ii_arr, list_ii

        # Generate ones to match the size of the arrays
        ones_arr_arr:np.ndarray = np.ones_like(arr_ii_arr)
        zeros_arr_arr:np.ndarray = np.zeros_like(arr_ii_arr)

        # Generate ones to match the size of the arrays
        ones_arr:np.ndarray = np.ones_like(arr_ii)

        # Generate trial sparse matrix
        tt_sparse:sparse.coo_matrix = sparse.coo_matrix((ones_arr,(arr_ii,arr_jj)),
                                                        shape=(nn, nn),dtype=float)
        
        tt_sparse:sparse.csr_matrix = tt_sparse.tocsr()
 

        # Generate trial force vector
        tt_f_vec:sparse.csc_matrix = sparse.coo_matrix((ones_arr_arr,(arr_ii_arr,zeros_arr_arr)),
                                                       shape=(nn, 1),dtype=float)
        tt_f_vec:sparse.csr_matrix = tt_f_vec.tocsr()
        
        # change to zeros
        tt_sparse.data[:] =0.0
        tt_f_vec.data[:] =0.0

        # Fill the matrices
        self.__K:sparse.csr_matrix = tt_sparse.copy()
        self.__M:sparse.csr_matrix = tt_sparse.copy()
        self.__F:sparse.csr_matrix = tt_f_vec.copy()

        self.__K:sparse.lil_matrix =  self.__K.tolil()
        self.__M:sparse.lil_matrix =  self.__M.tolil()
        self.__F:sparse.lil_matrix =  self.__F.tolil()


    # Member return functions
    @property
    def MeshGrid(self)->MeshGrid2D:
        '''
        Property of the Mesh to return the Meshgrid (array) associated to the mesh
        ''' 
        return self.__mesh_grid
    
    @property
    def E11(self)->float:
        '''
        Young Modulus in 1,1 direction set to the mesh
        '''
        return self.__E11
    
    @property
    def E22(self)->float:
        '''
        Young Modulus in 2,2 direction set to the mesh
        '''
        return self.__E22
    
    @property
    def G12(self)->float:
        '''
        Torsional modulus in 1,2 direction set to the mesh
        '''
        return self.__G12
    
    @property
    def nu12(self)->float:
        '''
        Poisson ratio in 1,2 direction set to the mesh
        '''
        return self.__nu12
    
    @property
    def nu21(self)->float:
        '''
        Poisson ratio in 2,1 direction set to the mesh.
        Uses the function defined before by knowing the values of E11, E22 and nu12
        '''
        return compute_nu21(self.__E11,self.__E22,self.__nu12)
    

    def get_reduced_stiffness_terms(self)->list:
        '''
        Return the reduced stiffness (Q1,Q2,Q3,Q4)
        '''
        return compute_reduced_stiffness_terms(E11=self.__E11,
                                               E22=self.__E22,
                                               G12=self.__G12,
                                               nu12=self.__nu12)
    
    
    def get_material_invariants(self)->list:
        '''
        Return the the material invariants terms (U1,U2,U3,U4,U5)
        '''
        return compute_material_invariants(E11=self.__E11,
                                            E22=self.__E22,
                                            G12=self.__G12,
                                            nu12=self.__nu12)
    
    @property
    def S4(self)->np.ndarray:
        '''
        Return the 4 Gauss Point Shape Functions
        '''
        return self.__S4
    
    @property
    def S1(self)->np.ndarray:
        '''
        Return the 1 Gauss Point Shape Functions
        '''
        return self.__S1
    
    @property
    def E_nxy(self)->np.ndarray:
        return self.__E_Nxy
    
    @property
    def Jacobian(self)->np.ndarray:
        '''
        Return the Jacobian of the transformation
        '''
        return self.__jacob
    
    def get_inverse_Jacobian(self)->np.ndarray:
        return self.__invJacob
    
    def get_determinant_Jacobian_4(self)->np.ndarray:
        return self.__detJacob4
    
    def get_determinant_Jacobian_1(self)->np.ndarray:
        return self.__detJacob1
    
    @property
    def dSdksieta4(self)->np.ndarray:
        '''
        Return the 4 Gauss Point based derivative of shape functions w.r.t. ksi and eta
        '''
        return self.__dSdksieta4
    
    @property
    def dSdksieta1(self)->np.ndarray:
        '''
        Return the 1 Gauss Point based derivative of shape functions w.r.t. ksi and eta
        '''
        return self.__dSdksieta1
    
    @property
    def dSdxy4(self)->np.ndarray:
        '''
        Return the 4 Gauss Point based derivative of shape functions w.r.t. x and y
        '''
        return self.__dSdxy4
    
    @property
    def dSdxy1(self)->np.ndarray:
        '''
        Return the 1 Gauss Point based derivative of shape functions w.r.t. x and y
        '''
        return self.__dSdxy1
    
    @property
    def sparse_matrices(self)->bool:
        '''
        Boolean handling if sparse matrices are used for FE evaluation
        '''
        return self.__sparse_matrices
    
    @property
    def M(self):
        '''
        Mass matrix associated to mesh
        '''
        return self.__M
    
    @property
    def K(self):
        '''
        Stiffness matrix associated to mesh
        '''
        return self.__K
    
    @property
    def F(self):
        '''
        Mass matrix associated to mesh
        '''
        return self.__F
    
    # Computation member functions
    def _assemble_finite_element_matrices(self,density_vector:np.ndarray,
                                           thickness:float,
                                           rho:float,
                                           E0:float = 1.0,
                                           Emin:float=1e-09)->None:
        '''
        Function to assemble the matrices for Finite Element Analysis of a design.

        Inputs:
        - density_vector: Array with the corresponding density of each element
        - V1_e: array with values of parameter V1_e corresponding to each element
        - V3_e: array with values of parameter V1_e corresponding to each element
        - thickness: Arbitrary thickness (as a plate)
        - rho: material density
        - Emin: Minimum Material Density
        '''

        # Set the material and void matrices
        C_material:np.ndarray = compute_in_plane_C_matrix(self.__E11,
                                              self.__E22,
                                              self.__G12,
                                              self.__nu12)
        
        C_void = np.array([[1,1,0],[1,1,0],[0,0,1]])*Emin

        # Loop for each element
        for el in range(self.MeshGrid.nel_total):

            # IN THE ACTUAL VARIABLE STIFFNESS MODELING, ELEMENTAL V1 & V3 VALUES
            # SHOULD BE TAKEN
            # V1 = 0.0; V3 = 0.0;

            if abs(density_vector[0,el] -E0) < 1e-12:
                C:np.ndarray = C_material.copy()
            else:
                C:np.ndarray = C_void.copy()
               
            # Initialize Ke, Me and Fe to zero
            Ke = np.zeros((NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF,
                          NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF))
            Me = np.zeros_like(Ke)
            Fe = np.zeros((NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF,
                          1))
            
            # General B Matrix
            B_gen = gen_B_matrix(self.__dSdxy4,4)

            # Gauss quadrature
            for gqp in range(4):
                # Elemental membrane stiffness matrix
                B:np.ndarray = B_gen[:,:,gqp]

                Ke = Ke + thickness*self.get_determinant_Jacobian_4()[0,gqp]*GQ_WEIGHT_4[gqp] * (B.transpose() @ C @ B)

                # Elemental membrane mass matrix
                S:np.ndarray = np.array([[self.__S4[0,gqp], 0, self.__S4[1,gqp], 0,self.__S4[2,gqp], 0, self.__S4[3,gqp], 0],
                     [0, self.__S4[0,gqp], 0, self.__S4[1,gqp], 0, self.__S4[2,gqp], 0, self.__S4[3,gqp]]])

                Me = Me + (rho*thickness*self.get_determinant_Jacobian_4()[0,gqp] * GQ_WEIGHT_4[gqp])*(S.transpose() @ S)
            
            if self.sparse_matrices:
                assemble_global_spmatrices(Ke,Me,Fe,self.__K,self.__M,self.__F,el,self.MeshGrid.E,
                                     NUMBER_OF_NODES_X_ELEMENT,NUMBER_OF_NODAL_DOF)
            else:
                assemble_global_matrices(Ke,Me,Fe,self.__K,self.__M,self.__F,el,self.MeshGrid.E,
                                     NUMBER_OF_NODES_X_ELEMENT,NUMBER_OF_NODAL_DOF)

    def set_matrices(self,density_vector:np.ndarray,thickness:float,rho:float,
                     E0:float=1.0,Emin:float=1e-09)->None:
        '''
        Function to assemble and set BCs the matrices for Finite Element Analysis of a design.

        Inputs:
        - density_vector: Array with the corresponding density of each element
        - V1_e: array with values of parameter V1_e corresponding to each element
        - V3_e: array with values of parameter V1_e corresponding to each element
        - thickness: Arbitrary thickness (as a plate)
        - rho: material density
        - Emin: Minimum Material Density
        '''
        
        
        # Update the check on having "non-zero" valued matrices
        self._non_zero_matrices = True

        # Assemble K, F and M

        self._assemble_finite_element_matrices(density_vector,thickness,rho,E0,Emin)

        if not self.sparse_matrices:

            # Function to apply the boundary conditions
            BCiN,NBcN = apply_BC(self.__K,self.__F,self.__mesh_grid.grid_point_number_total,
                                self.__mesh_grid.grid_point_number_X,NUMBER_OF_NODAL_DOF)
        else:
            # Function to apply the boundary conditions
            BCiN,NBcN = apply_BC_sparse(self.__K,self.__F,self.__mesh_grid.grid_point_number_total,
                                self.__mesh_grid.grid_point_number_X,NUMBER_OF_NODAL_DOF)
        
        # Apply vertical load on middle right node
        self.__F[NUMBER_OF_NODAL_DOF*self.__mesh_grid.grid_point_number_X*
               math.ceil((self.__mesh_grid.grid_point_number_Y)/2)-1,0] = -0.1


    def compute_displacements(self)->list:

        if not self._non_zero_matrices:
            return np.zeros_like(self.__F),0,0
        
        else:
            NNDoF:int = NUMBER_OF_NODAL_DOF
            NN_l:int = self.__mesh_grid.grid_point_number_X
            
            # Solve the system

            if not self.sparse_matrices:
                u_vec:np.ndarray = scilinalg.solve(self.__K,self.__F,assume_a='gen',overwrite_a=True,
                                                   overwrite_b=True)
            else:
                K_conv:sparse.csr_matrix = self.__K.tocsr()
                F_conv:sparse.csr_matrix = self.__F.tocsr()
                u_vec:np.ndarray = sparse.linalg.spsolve(K_conv,F_conv)

            # Set the end value of the vector
            endd:int = u_vec.shape[0]

            # Get the x-direction DOF positions
            ran_1:np.ndarray = np.arange(start=0,step=NNDoF,stop=endd-1)

            # Get the y-direction DOF positions
            ran_2:np.ndarray = np.arange(start=1,step=NNDoF,stop=endd)

            # Get the x-direction DOF positions at the tip
            ran_3:np.ndarray = np.arange(start=NNDoF*(NN_l-1),step=NNDoF*NN_l,stop=endd-1)

            # Get the y-direction DOF positions at the tip
            ran_4:np.ndarray = np.arange(start=NNDoF*(NN_l-1)+1,step=NNDoF*NN_l,stop=endd)

            # Average displacement within the entire plate
            u_avg:float = np.mean(np.sqrt(np.power(u_vec[ran_1],2) + 
                                    np.power(u_vec[ran_2],2)))
            
            # Average tip displacement
            u_tip:float = np.mean(np.sqrt(np.power(u_vec[ran_3],2) + 
                                    np.power(u_vec[ran_4],2)))


            return u_vec, u_avg, u_tip

    
    def mesh_retrieve_Strain_Stress(self,
                                    density_vector:np.ndarray,
                                    disp:np.ndarray)->list:
        
        listt:list = retrieve_Strain_Stress(NN=self.__mesh_grid.grid_point_number_total,
                                            NN_l=self.__mesh_grid.grid_point_number_X,
                                            NN_h=self.__mesh_grid.grid_point_number_Y,
                                            E = self.__mesh_grid.E,
                                            NE = self.__mesh_grid.nel_total,
                                            u=disp,
                                            density_vector=density_vector,
                                            E11 = self.__E11,
                                            E22 = self.__E22,
                                            G12 = self.__G12,
                                            nu12=self.__nu12,
                                            dSdxy=self.__dSdxy4)
        return listt

    def mesh_compute_compliance(self,disp:np.ndarray,density_vector:np.ndarray,
                                            thickness:float,
                                            E0:float = 1.00,
                                            Emin:float = 1e-09)->np.ndarray:
        '''
        Member function wherein the compliance is computed "as a cost function"
        per each finite element

        Inputs:
        - density_vector: vector with the density values of each Finite Element
        - disp: Vector with the displacements of a FEA evaluation
        - V1_e: array with the values of V1 per each Finite Element
        - V3_e: array with the values of V3 per each Finite Element
        - thickness: default thickness of material
        - rho: density of the material
        - Emin: Minimum Material Density
        '''

        # Initialise a compliance vector
        comp_vec:np.ndarray = np.zeros((self.MeshGrid.nel_total,1))

        # Loop for each element
        for el in range(self.__mesh_grid.nel_total):

            # IN THE ACTUAL VARIABLE STIFFNESS MODELING, ELEMENTAL V1 & V3 VALUES
            # SHOULD BE TAKEN
            # V1 = 0.0; V3 = 0.0;

            if abs(density_vector[0,el] - E0) < 1e-12:
                C:np.ndarray = compute_in_plane_C_matrix(self.__E11,
                                              self.__E22,
                                              self.__G12,
                                              self.__nu12)
            else:
                C:np.ndarray = np.array([[1,1,0],[1,1,0],[0,0,1]])*Emin
               
            # Initialize Ke
            Ke = np.zeros((NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF,
                          NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF))
            
            # General B Matrix
            B_gen = gen_B_matrix(self.__dSdxy4,4)

            # Gauss quadrature
            for gqp in range(4):
                # Elemental membrane stiffness matrix
                B:np.ndarray = B_gen[:,:,gqp]

                Ke = Ke + thickness*self.get_determinant_Jacobian_4()[0,gqp]*GQ_WEIGHT_4[gqp] * (B.transpose() @ C @ B)
            
            # Get the nodal freedom table mapping the positions of the DOFS linked per element

            elem_nodes = self.MeshGrid.E[el,1:5]
            DOF_arr:np.ndarray = np.zeros((NUMBER_OF_NODAL_DOF*NUMBER_OF_NODES_X_ELEMENT,1))
            for pos, node in enumerate(elem_nodes.ravel()):
                for nnDOF in range(NUMBER_OF_NODAL_DOF):
                    DOF_arr[NUMBER_OF_NODAL_DOF*pos + nnDOF,0] = NUMBER_OF_NODAL_DOF*node + nnDOF
            
            # Reformulate the array of pointers of the DOF to be integers
            DOF_arr = DOF_arr.astype(int)
            tmp_DOF = [int(DOF_arr[i]) for i in range(len(DOF_arr))]

            # Extract the displacements of the element from global displacement vector
            u_el:np.ndarray = disp[tmp_DOF].reshape((NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF,1))
            
            comp_vec[el] = u_el.transpose() @ Ke @ u_el

    
        return comp_vec

