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
from .four_point_quadrature_plane_stress import Mesh



def compute_in_plane_C_matrix_composite(E11:float,E22:float,G12:float,nu12:float,
                                            V1:float,V3:float)-> np.ndarray:
    r'''
    Compute the in-plane C-matrix for the Finite Element Method for composite materials

    
    Args
    -----
        - E11: Young Modulus in 11 direction
        - E22: Young Modulus in 22 direction
        - G12: Torsional Modulus in 12 direction
        - nu12: Poisson's Ratios
        - V1: Fiber orientation vector 1
        - V3: Fiber orientation vector 2 
    '''
    
    # Compute the material invariants
    u1,u2,u3,u4,u5 = compute_material_invariants(E11=E11,
                                                 E22=E22,
                                                 nu12=nu12,
                                                 G12=G12)

    # C matrix entities
    c11:float = u1+u2*V1+u3*V3
    c12:float = u4-u3*V3
    c22:float = u1-u2*V1+u3*V3
    c33:float = u5-u3*V3
    
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



def retrieve_Strain_Stress_Composite(NN:int,NN_l:int,NN_h:int,E:np.ndarray,
                                    NE:int,u:np.ndarray,density_vector:np.ndarray,
                                    V1_e:np.ndarray,V3_e:np.ndarray,
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
    - density_vector: density of all the finite elements in vector form
    - V1_e: Array with the values of V1 of each element
    - V3_e: Array with the values of V3 of each element
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
            V1 = V1_e[el,0]
            V3 = V3_e[el,0]
            C_mat:np.ndarray = compute_in_plane_C_matrix_composite(E11,E22,G12,nu12,V1,V3)
        else:
            C_mat:np.ndarray = np.array([[1,1,0],[1,1,0],[0,0,1]])*Emin
        
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


class CompositeMaterialMesh(Mesh):
    r"""
    This is a class which overloads the Mesh class to include the definition of a composite material.
    """

    # Computation member functions
    def _assemble_finite_element_matrices(self,
                                          density_vector:np.ndarray,
                                           V1_e:np.ndarray,
                                           V3_e:np.ndarray,
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

        # Loop for each element
        for el in range(self.MeshGrid.nel_total):

            # IN THE ACTUAL VARIABLE STIFFNESS MODELING, ELEMENTAL V1 & V3 VALUES
            # SHOULD BE TAKEN
            # V1 = 0.0; V3 = 0.0;

            if abs(density_vector[0,el] -E0) < 1e-12:
                V1 = V1_e[el,0]
                V3 = V3_e[el,0]
                C:np.ndarray = compute_in_plane_C_matrix_composite(self.E11,
                                              self.E22,
                                              self.G12,
                                              self.nu12,
                                              V1,V3)
            else:
                C:np.ndarray = np.array([[1,1,0],[1,1,0],[0,0,1]])*Emin
               
            # Initialize Ke, Me and Fe to zero
            Ke = np.zeros((NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF,
                          NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF))
            Me = np.zeros_like(Ke)
            Fe = np.zeros((NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF,
                          1))
            
            # General B Matrix
            B_gen = gen_B_matrix(self.dSdxy4,4)

            # Gauss quadrature
            for gqp in range(4):
                # Elemental membrane stiffness matrix
                B:np.ndarray = B_gen[:,:,gqp]

                Ke = Ke + thickness*self.get_determinant_Jacobian_4()[0,gqp]*GQ_WEIGHT_4[gqp] * (B.transpose() @ C @ B)

                # Elemental membrane mass matrix
                S:np.ndarray = np.array([[self.S4[0,gqp], 0, self.S4[1,gqp], 0,self.S4[2,gqp], 0, self.S4[3,gqp], 0],
                     [0, self.S4[0,gqp], 0, self.S4[1,gqp], 0, self.S4[2,gqp], 0, self.S4[3,gqp]]])

                Me = Me + (rho*thickness*self.get_determinant_Jacobian_4()[0,gqp] * GQ_WEIGHT_4[gqp])*(S.transpose() @ S)
            
            if self.sparse_matrices:
                assemble_global_spmatrices(Ke,Me,Fe,self.K,self.M,self.F,el,self.MeshGrid.E,
                                     NUMBER_OF_NODES_X_ELEMENT,NUMBER_OF_NODAL_DOF)
            else:
                assemble_global_matrices(Ke,Me,Fe,self.K,self.M,self.F,el,self.MeshGrid.E,
                                     NUMBER_OF_NODES_X_ELEMENT,NUMBER_OF_NODAL_DOF)
                

    def set_matrices(self,
                     density_vector:np.ndarray,
                     V1_e:np.ndarray,
                     V3_e:np.ndarray,
                     thickness:float,
                     rho:float,
                     E0:float=1.0,
                     Emin:float=1e-09)->None:
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

        self._assemble_finite_element_matrices(density_vector,V1_e,V3_e,thickness,rho,E0,Emin)

        if not self.sparse_matrices:

            # Function to apply the boundary conditions
            BCiN,NBcN = apply_BC(self.K,self.F,self.MeshGrid.grid_point_number_total,
                                self.MeshGrid.grid_point_number_X,NUMBER_OF_NODAL_DOF)
        else:
            # Function to apply the boundary conditions
            BCiN,NBcN = apply_BC_sparse(self.K,self.F,self.MeshGrid.grid_point_number_total,
                                self.MeshGrid.grid_point_number_X,NUMBER_OF_NODAL_DOF)
        
        # Apply vertical load on middle right node
        self.F[NUMBER_OF_NODAL_DOF*self.MeshGrid.grid_point_number_X*
               math.ceil((self.MeshGrid.grid_point_number_Y)/2)-1,0] = -0.1
    

    def mesh_compute_compliance(self,disp:np.ndarray,density_vector:np.ndarray,
                                            V1_e:np.ndarray,
                                            V3_e:np.ndarray,
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
        for el in range(self.MeshGrid.nel_total):

            # IN THE ACTUAL VARIABLE STIFFNESS MODELING, ELEMENTAL V1 & V3 VALUES
            # SHOULD BE TAKEN
            # V1 = 0.0; V3 = 0.0;

            if abs(density_vector[0,el] - E0) < 1e-12:
                V1 = V1_e[el,0]
                V3 = V3_e[el,0]
                C:np.ndarray = compute_in_plane_C_matrix_composite(self.E11,
                                              self.E22,
                                              self.G12,
                                              self.nu12,
                                              V1,V3)
            else:
                C:np.ndarray = np.array([[1,1,0],[1,1,0],[0,0,1]])*Emin
               
            # Initialize Ke
            Ke = np.zeros((NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF,
                          NUMBER_OF_NODES_X_ELEMENT*NUMBER_OF_NODAL_DOF))
            
            # General B Matrix
            B_gen = gen_B_matrix(self.dSdxy4,4)

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
    
    def mesh_retrieve_Strain_Stress(self,V1_e:np.ndarray,V3_e:np.ndarray,
                                    density_vector:np.ndarray,disp:np.ndarray,
                                    E0:float = 1.00,
                                    Emin:float = 1.00E-09)->list:
        
        listt:list = retrieve_Strain_Stress_Composite(NN=self.MeshGrid.grid_point_number_total,
                                            NN_l=self.MeshGrid.grid_point_number_X,
                                            NN_h=self.MeshGrid.grid_point_number_Y,
                                            E = self.MeshGrid.E,
                                            NE = self.MeshGrid.nel_total,
                                            u=disp,
                                            density_vector=density_vector,
                                            V1_e=V1_e,
                                            V3_e=V3_e,
                                            E11 = self.E11,
                                            E22 = self.E22,
                                            G12 = self.G12,
                                            nu12=self.nu12,
                                            dSdxy=self.dSdxy4,
                                            E0=E0,
                                            Emin=Emin)
        return listt

       

        
    
    

