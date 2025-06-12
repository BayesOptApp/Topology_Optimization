import numpy as np
import math
from scipy import sparse

# ----------------------------------------------------------------------------------------------------
# ---------------------------------------------CONSTANTS----------------------------------------------
# ----------------------------------------------------------------------------------------------------

# Meshgrid properties constants
LENGTH_DEFAULT:float = 100.0 # Default length of plate
HEIGHT_DEFAULT:float = 50 # Default height of plate

ELEMENT_LENGTH_DEFAULT:float = 1.0 # Default length of Finite 2D Plate Element
ELEMENT_HEIGHT_DEFAULT:float = 1.0 # Default height of Finite 2D Plate Element

# Material property constants
E11_DEFAULT:float = 25
E22_DEFAULT:float = 1
G12_DEFAULT:float = 0.5
NU12_DEFAULT:float = 0.25

# Finite Element Parameters
NUMBER_OF_NODES_X_ELEMENT:int = 4 # Number of nodes per element
'''
Number of nodes per element
'''
NUMBER_OF_NODAL_DOF:int = 2 # u,v

# 1 point Gauss Quadrature integration parameters
GQ_POINT_1:np.ndarray = np.zeros((1,2))

GQ_WEIGHT_1:np.ndarray = np.zeros((1,1))
GQ_WEIGHT_1[0] = 4.0

# 4-point quadrature
GQ_POINT_4:np.ndarray = np.array([[-1,-1],[1,-1],[-1,1],[1,1]]) / math.sqrt(3.0)

GQ_WEIGHT_4:np.ndarray = np.zeros((4,))

GQ_WEIGHT_4[0] = 1.0
GQ_WEIGHT_4[1] = 1.0
GQ_WEIGHT_4[2] = 1.0
GQ_WEIGHT_4[3] = 1.0

# ----------------------------------------------------------------------------------------
# Additional (helper) functions
# ----------------------------------------------------------------------------------------

def shape(xi_inp:np.ndarray):
    
    """Shape functions for a 4-node, isoparametric element N_i(xi,eta) where i=[1,2,3,4]
    Input: 1x2,  
    Output: 4x1"""
        
    xi,eta = tuple(xi_inp)
    N = [(1.0-xi)*(1.0-eta), (1.0+xi)*(1.0-eta), (1.0+xi)*(1.0+eta), (1.0-xi)*(1.0+eta)]
    return 0.25 * np.array(N).T

def gradshape(xi_inp:np.ndarray):
	"""Gradient of the shape functions for a 4-node, isoparametric element.
		dN_i(xi,eta)/dxi and dN_i(xi,eta)/deta
		Input: 1x2,  Output: 2x4"""
	xi,eta = tuple(xi_inp)
	dN = [[-(1.0-eta),  (1.0-eta), (1.0+eta), -(1.0+eta)],
		  [-(1.0-xi), -(1.0+xi), (1.0+xi),  (1.0-xi)]]
	return 0.25 * np.array(dN)

def part_B_matrix(dSdxy:np.ndarray,gqp:int)->np.ndarray:
    '''
    Compute the particular B Matrix for a defined Gauss Point
    '''
    B_mat:np.ndarray = np.array([[dSdxy[0,0,0,gqp], 0, dSdxy[0,0,1,gqp], 0, dSdxy[0,0,2,gqp], 0, dSdxy[0,0,3,gqp], 0 ],
                                     [0, dSdxy[0,1,0,gqp], 0, dSdxy[0,1,1,gqp], 0, dSdxy[0,1,2,gqp], 0, dSdxy[0,1,3,gqp]],
                                     [dSdxy[0,1,0,gqp], dSdxy[0,0,0,gqp], dSdxy[0,1,1,gqp], dSdxy[0,0,1,gqp], dSdxy[0,1,2,gqp], dSdxy[0,0,2,gqp], dSdxy[0,1,3,gqp], dSdxy[0,0,3,gqp]]])

    return B_mat

def gen_B_matrix(dSdxy:np.ndarray,tot_num_GP:int=0)->np.ndarray:
    '''
    Compute array compiling all combinations of B Matrix for several Gauss Points
    '''
    if tot_num_GP <= 0:
        tot_num_GP= dSdxy.shape[3]
    
    elif dSdxy.shape[3] < tot_num_GP:
        tot_num_GP= dSdxy.shape[3]
    

    B_mat_gen:np.ndarray = np.zeros((3,8,tot_num_GP))

    for gqp in range(tot_num_GP):
        B_mat_gen[:,:,gqp] = part_B_matrix(dSdxy=dSdxy,gqp=gqp)

    return B_mat_gen

def compute_nu21(E11:float,E22:float,nu12:float)->float:
    return E22*nu12/E11

def compute_reduced_stiffness_terms(E11:float,E22:float,G12:float,nu12:float)->list:
    nu21:float = compute_nu21(E11=E11,E22=E22,nu12=nu12)
    q11:float = E11/(1-nu12*nu21)
    q22:float = E22/(1-nu12*nu21)
    q12:float = nu12*E22/(1-nu12*nu21)
    q66:float = G12
    return q11,q22,q12,q66

def compute_material_invariants(E11:float,E22:float,G12:float,nu12:float)->list:
    q11,q22,q12,q66 = compute_reduced_stiffness_terms(E11=E11,E22=E22,nu12=nu12,G12=G12)

    u1:float = 1/8*(3*q11+3*q22+2*q12+4*q66)
    u2:float = 1/2*(q11-q22)
    u3:float = 1/8*(q11+q22-2*q12-4*q66)
    u4:float = 1/8*(q11+q22+6*q12-4*q66)
    u5:float = 1/8*(q11+q22-2*q12+4*q66)

    return u1,u2,u3,u4,u5


def assemble_global_matrices(element_stiffness_mat:np.ndarray,
                             element_mass_mat:np.ndarray,
                             element_force_vec:np.ndarray,
                             global_stiffness_mat:np.ndarray,
                             global_mass_mat:np.ndarray,
                             global_force_vec:np.ndarray,
                             elem_pos:int,
                             E:np.ndarray,
                             NEN:int,
                             NNDOF:int)->None:
    '''
    Function to assemble dense FEM matrices.

    The function receives the inputs:

    
    '''
    
    for ii in range(NEN):
        iN = int(E[elem_pos,ii+1])
        for jj  in range(NNDOF):
            global_force_vec[iN+jj,0] = global_force_vec[iN+jj,0] + element_force_vec[jj+NNDOF*(ii),0]

        for jj in range(NEN):
            jN = int(E[elem_pos,jj+1])
            KeNNDOF = element_stiffness_mat[(ii)*NNDOF:(ii+1)*NNDOF,(jj)*NNDOF:(jj+1)*NNDOF]
            #MeNNDOF = element_mass_mat[(ii)*NNDOF:(ii+1)*NNDOF,(jj)*NNDOF:(jj+1)*NNDOF]
            
            global_stiffness_mat[(iN)*NNDOF:(iN+1)*NNDOF,(jN)*NNDOF:(jN+1)*NNDOF] = global_stiffness_mat[(iN)*NNDOF:(iN+1)*NNDOF,(jN)*NNDOF:(jN+1)*NNDOF] + KeNNDOF
            #global_mass_mat[(iN)*NNDOF:(iN+1)*NNDOF,(jN)*NNDOF:(jN+1)*NNDOF] =  global_mass_mat[(iN)*NNDOF:(iN+1)*NNDOF,(jN)*NNDOF:(jN+1)*NNDOF] +  MeNNDOF


def assemble_global_spmatrices(element_stiffness_mat:np.ndarray,
                               element_mass_mat:np.ndarray,
                               element_force_vec:np.ndarray,
                               global_stiffness_mat:sparse.lil_matrix,
                               global_mass_mat:sparse.lil_matrix,
                               global_force_vec:sparse.lil_matrix,
                               elem_pos:int,E:np.ndarray,NEN:int,
                               NNDOF:int)->None:
    
    for ii in range(NEN):
        iN = int(E[elem_pos,ii+1])
        for jj  in range(NNDOF):
            global_force_vec[iN+jj,0] = global_force_vec[iN+jj,0] + element_force_vec[jj+NNDOF*(ii),0]

        for jj in range(NEN):
            jN = int(E[elem_pos,jj+1])
            KeNNDOF = element_stiffness_mat[(ii)*NNDOF:(ii+1)*NNDOF,(jj)*NNDOF:(jj+1)*NNDOF]
            #MeNNDOF = element_mass_mat[(ii)*NNDOF:(ii+1)*NNDOF,(jj)*NNDOF:(jj+1)*NNDOF]
            
            for ii_ind in range(NNDOF):
                for jj_ind in range(NNDOF):
                    #global_mass_mat[(iN)*NNDOF+ii_ind,(jN)*NNDOF+jj_ind] = MeNNDOF[ii_ind,jj_ind] + global_mass_mat[(iN)*NNDOF+ii_ind,(jN)*NNDOF+jj_ind]
                    global_stiffness_mat[(iN)*NNDOF+ii_ind,(jN)*NNDOF+jj_ind] = KeNNDOF[ii_ind,jj_ind] + global_stiffness_mat[(iN)*NNDOF+ii_ind,(jN)*NNDOF+jj_ind]