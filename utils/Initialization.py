'''
Reinterpretation of code for Structural Optimisation
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

'''

# Import libraries
import numpy as np
import numpy.matlib
from scipy import sparse
from typing import List, Union

'''
The following function takes as inputs the number of elements in -x and 
-y directions and the test case to set the variables to set the FEA of
the desired case

'''
def prepare_FEA(nelx:int,nely:int,test_case:str='cant-beam',
                nu:float=0.3)->List[np.ndarray]:

    # Preparation of the Finite Element analysis
    # Sub-blocks of Stiffness Matrix
    A11:np.ndarray = np.array([[12 , 3, -6, -3],[ 3, 12,  3,  0],[-6,  3, 12, -3],[-3 , 0, -3, 12]])
    A12:np.ndarray = np.array([[-6, -3,  0,  3],[-3, -6, -3, -6],[ 0, -3, -6,  3],[ 3, -6,  3, -6]])
    B11:np.ndarray = np.array([[-4,  3, -2,  9],[ 3, -4, -9,  4],[-2, -9, -4, -3],[ 9,  4, -3, -4]])
    B12:np.ndarray = np.array([[ 2, -3,  4, -9],[-3,  2,  9, -2],[ 4,  9,  2,  3],[-9, -2,  3,  2]])

    # Stiffness Matrix
    temp1:np.ndarray = np.block([[A11,A12],[np.transpose(A12),A11]])
    temp2:np.ndarray = np.block([[B11,B12],[np.transpose(B12),B11]])

    KE:np.ndarray = (1/(1-nu**2)/24)*(temp1+nu*temp2)
    del temp1,temp2,A11, A12, B11, B12 # Delete the auxiliary variables
    
    # Node numbers
    nodenrs=np.arange(0,(1+nelx)*(1+nely)).reshape(1+nely,1+nelx,order='F')
    
    edofVec:np.ndarray = ((2*nodenrs[0:-1,0:-1])).reshape(nelx*nely,1,order='F')
    edofMat:np.ndarray = np.matlib.repmat(edofVec,1,8)+np.matlib.repmat(np.concatenate([np.array([0,1]),2*nely+np.array([2,3,4,5]),np.array([2,3])],axis=0),nelx*nely,1)

    iK=np.kron(edofMat,np.ones((8,1))).T
    jK=np.kron(edofMat,np.ones((1,8))).T
    
    # Definition of loads and supports

    if test_case.find('MBB-beam')!=-1:
        #(half MBB-beam)
        aux1 = np.arange(0,2*(nely+1),2,dtype=int)
        aux2 = 2*(nelx+1)*(nely+1)
        fixeddofs:np.ndarray = np.union1d(aux1,aux2)
        F = sparse.coo_matrix(([-1],([1],[0])),shape=(2*(nely+1)*(nelx+1),1))
        del aux1, aux2 # delete auxiliary variables

    elif test_case.find('cant-beam')!=-1:
        #%(cantilever beam)   
        fixeddofs:np.ndarray = np.arange(0,(2*(nely+1)))
        loaddof=2*(nely+1)*nelx+nely+1
        F=sparse.csc_matrix(([-1], ([loaddof], [0])), shape=(2*(nely+1)*(nelx+1), 1))

    else:
        raise ValueError("The test case is neither 'MBB-beam' nor 'cant-beam' ")
    
    U:np.ndarray = np.zeros((2*(nelx+1)*(nely+1),1))
    alldofs=np.arange(0,2*(nely+1)*(nelx+1)) # All degrees of freedom
    freedofs:np.ndarray = np.setdiff1d(alldofs,fixeddofs)

    return KE,iK,jK,F,U,freedofs,edofMat


