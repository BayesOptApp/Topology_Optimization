'''
Reinterpretation of code for Structural Optimisation
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

'''

__authors__ = ['Elena Raponi', 'Ivan Olarte Rodriguez']

# import usual Python libraries
import numpy as np
from scipy import sparse
import math

# import the copy library
from copy import copy, deepcopy

# Import evaluate FEA function
from FEA import evaluate_FEA_LP, return_element_midpoint_positions, compute_number_of_joined_bodies, compute_number_of_joined_bodies_2
from FEA import compute_objective_function

# Import DataClasses
from dataclasses import dataclass

# Import Typing library
from typing import List, Tuple, Union, Optional

# Import the MMC Library
from geometry_parameterizations.MMC import MMC

# Import the Topology library
from utils.Topology import Topology

from Design import Design, SCALATION_MODES, CONTINUITY_CHECK_MODES

# Import IOH Real Problem
import ioh

# Import the Initialization
#from Initialization import prepare_FEA

# ----------------------------------------------------------------------------------------------------
# ---------------------------------------------CONSTANTS----------------------------------------------
# ----------------------------------------------------------------------------------------------------

# Names to check for optimisation of Lamination Parameters
KEY_NAMES = ("VR","V3_1","V3_2")

# Default optimization modes
OPT_MODES = ('TO','TO+LP','LP')

# -----------------------------------------------------------------------------------------------------
# --------------------------------------------CLASS DEFINITION-----------------------------------------
# -----------------------------------------------------------------------------------------------------
@dataclass
class Design_LP(Design):
    '''
    Class which is generated to contain a design for the optimisation procedure
    and its properties with Lamination Parameters
    '''
    def __init__(self, 
                 nmmcsx:int, 
                 nmmcsy:int, 
                 nelx:int, 
                 nely:int,
                 VR:float = 0.5, 
                 V3_1:float = 0.0, 
                 V3_2:float = 0.0,
                 mode:str=OPT_MODES[0],
                 symmetry_condition:bool=False,
                 scalation_mode:str = SCALATION_MODES[0],
                 initialise_zero:bool=False,
                 add_noise:bool = False,
                 E0:float = 1.00,
                 Emin:float = 1e-09,
                 continuity_check_mode:Optional[str]=CONTINUITY_CHECK_MODES[0],
                 **kwargs):
        '''
        Constructor of the class
        Inputs:
            - nmmcsx: number of Morphable Moving Components (MMCs) in x-direction
            - nmmcsy: number of Morphable Moving Components (MMCs) in y-direction
            - nelx: number of finite elements in x-direction
            - nely: number of finite elements in y-direction
            - mode: Optimisation mode: 'TO', 'LP' or 'TO+LP'
            - VR: VR parameter set for Lamination
            - V3_1: V3_1 parameter set for Lamination
            - V3_2: V3_2 parameter set for Lamination
            - symmetry_condition: Impose a symmetry condition on the design on the x-axis.
                                  If the symmetry condition is imposed, only half of the 
                                  supposed MMC's are saved.
            - initialise_zero: Initialise the table of attributes as zeros
            - add_noise: boolean to control if noise is added to default initialisation
            - scalation_mode: Select a scalation mode: Set values for 'Bujny' or 'unitary'
            - inverted_init: this is to invert the order of the MMC for the default initialisation.
            - **kwargs: keyworded arguments in case the Optimisation mode is set to 'TO+LP'
        '''
        
        #Assign the mode
        self.mode:str = mode

        # Call the constructor of the parent class
        Design.__init__(self,
                        nmmcsx=nmmcsx,
                         nmmcsy=nmmcsy,
                         nelx=nelx,
                         nely=nely,
                         E0=E0,
                         Emin=Emin,
                         scalation_mode=scalation_mode,
                         symmetry_condition=symmetry_condition,
                         initialise_zero=initialise_zero,
                         add_noise=add_noise,
                         continuity_check_mode=continuity_check_mode,
                         **kwargs)
        

        # Set the values of the Lamination Parameters     
        self.__VR:float = VR
        self.__V3_1:float = V3_1
        self.__V3_2:float = V3_2


        
            

    def return_mutable_properties_in_array(self,scaled:bool=True)->np.ndarray:

        '''
        Returns the mutable properties of the Design 

        Inputs:
        - scaled: Determine to output the scaled
        '''
        
        if scaled:
            # If scaled is chosen, extract the array of scaled MMC properties 

            if self.mode.find("TO") != -1:
                compl_array:np.ndarray = self.get_array_of_scaled_MMC_properties()
            else:
                compl_array = np.array([]).reshape((1,-1))
            

            lam_params:np.ndarray =  self.get_scaled_lamination_parameters().reshape((1,3))

            if self.mode.find("LP") !=-1:
                compl_array = np.hstack((compl_array,lam_params))
            
            return compl_array

        else:
            # If scaled is not chosen,extract the array of unscaled MMC properties

            if self.mode.find("TO") != -1:
                compl_array:np.ndarray = self.get_array_of_unscaled_MMC_properties()
            else:
                compl_array:np.ndarray = np.array([]).reshape((1,-1))

            lam_params:np.ndarray =  self.get_lamination_parameters().reshape((1,3))   
            if self.mode.find("LP") !=-1:
                compl_array:np.ndarray = np.hstack((compl_array,lam_params))
            
            return compl_array

    
    # ------------------------------------------------------------------------
    # Evaluate Finite Element Method with current Design
    # ------------------------------------------------------------------------

    
    def evaluate_FEA_design(self,volfrac:float,iterr:int,sample:int,run_:int,
                            use_sparse_matrices:bool=False,
                            plotVariables=False, cost_function:str="compliance",
                            penalty_factor:float=0.0,
                            avoid_computation_for_not_compliance:bool=True)->float:
        '''
        Sets the cost function of the design (from Finite Element Method)

        Inputs:
        - volfrac: The volume fraction to be considered
        - iter: current iteration
        - sample: sample to be processed
        - Emin: Minimum density (to avoid numerical errors) and set the "Ersatz" material
        - E0: Reference elastic modulus of element
        - run_: Current optimisation loop run
        - use_sparse_matrices: Control the usage of Sparse Matrices for the FEM
        - plotVariables: Control to plot the variables of interest of designs
        - cost_function: string to determine the type of cost function. Two options available: "compliance" or "mean_displacement"
        - penalty_factor: a floating value indicating a factor to penalise designs which exceed the volume constraint.
        - avoid_computation_for_not_compliance: This variable is to avoid computation of the function.
        Outputs:
        - Cost of the design
        '''

        if avoid_computation_for_not_compliance:
            # Get the compliance array
            comp_array: np.ndarray = self.identify_natural_constraints_violation()

            # Check if any of the constraints are violated
            if np.any(comp_array==False):
                # Update the cost
                self.score_FEA = 1e20
                return 1e20

        # Compute the topology optimisation matrix
        TO_mat:np.ndarray = self._topo.return_floating_topology()

        # Compute the cost of the Design
        cost:float = evaluate_FEA_LP(x=np.array([self.VR,self.V3_1,self.V3_2]),
                                    TO_mat=TO_mat,
                                    iterr = iterr,
                                    sample = sample,
                                    volfrac=volfrac,
                                    Emin=self.Emin,E0=self.E0,
                                    run_=run_,
                                    sparse_matrices_solver=use_sparse_matrices,
                                    plotVariables=plotVariables,
                                    symmetry_cond=self.symmetry_condition_imposed,
                                    cost_function=cost_function,
                                    penalty_factor=penalty_factor,
                                    mode=self.mode)
            
        # Update the cost
        #self.__score_FEA = cost

        return cost

    # ------------------------------------------------------------------------
    # Member functions to return the private attributes
    # ------------------------------------------------------------------------
    
    
    @property
    def mode(self)->str:
        return self.__mode
    
    @mode.setter
    def mode(self,new_mode:str)->None:
        if isinstance(new_mode,str):
            if new_mode == "TO" or new_mode =="TO+LP" or new_mode =="LP":
                self.__mode = new_mode
        else:
            raise ValueError("The mode should be a string value")
    

    def problem_name(self)->str:
        r"""
            This function returns the Full_Name of the problem given the 
            mode
        """
        if self.mode == "TO":
            return "Topology_Optimization_Without_Lamination_Parameters"
        elif self.mode == "TO+LP":
            return "Topology_Optimization_With_Lamination_Parameters"
        elif self.mode == "LP":
            return "Lamination_Parameters Optimization"
        
    
    @property
    def VR(self)->float:
        return self.__VR

    @property           
    def V3_1(self)->float:    
        return self.__V3_1
    
    @property 
    def V3_2(self)->float:
        return self.__V3_2
    
    def get_lamination_parameters(self)->np.ndarray:
        return np.array([self.__VR, self.__V3_1,self.__V3_2 ])
    
    def get_scaled_lamination_parameters(self)->np.ndarray:
        return np.array([self.__VR, 0.5*(self.__V3_1+1),0.5*(self.__V3_2+1)])
    

    @property
    def problem_dimension(self)-> int:
        '''
        Returns the dimension of the optimisation problem
        '''

        if self.mode == "TO":
            return len(self.list_of_MMC)*5
        elif self.mode == "TO+LP":
            return len(self.list_of_MMC)*5+3
        elif self.mode == "LP":
            return 3
        else:
            raise AttributeError()
    

    

    
    
    # --------------------------------------------------------------------------------------------
    # Member functions used to change properties of the design at each MMC level
    # --------------------------------------------------------------------------------------------

    def __change_values_of_MMCs_from_array(self, samp_array:np.ndarray, repair_level:int=2, 
                                         tol:float = 0.5, min_thickness:float = 1.0,**kwargs)->None:
        '''
        Given an array of values (in the same format as the array of properties)
        change the values of each MMC based on the values of the received array
        by parameter

        Inputs:
        - samp_array: array with new values (result from optimisation/modification process)
        - repair_level: level of repair desired in case the parameters given on the array
                        lie outside the admissible range
        - kwargs: keyword arguments (tolerance, min_thickness)
        '''
        if self._zero_valued== True:
            self._zero_valued = False

        # Modify the pre-score
        self._pre_score = np.nan

        # Check if the mode is just for Lamination Parameter modification. If so raise an error
        if self.mode == "LP":
            raise PermissionError("The mode is set to 'LP'. Then, the MMC cannot be modified!")
        
        # Check if the repair level is an acceptable value
        if not (repair_level ==0 or repair_level == 1 or repair_level == 2):
            raise ValueError("The repair level should be an integer equal to 0, 1 or 2")

        # Check the sample array has the number of elements required to modify all the properties
        # of each of the MMCs

        samp_array_manip:np.ndarray = samp_array.ravel()

        # Check if the array contains all the required number of attributes to modify each MMC
        if samp_array_manip.size != (5*len(self.__list_of_MMC)):
            raise AttributeError("The size of the array does not correspond" 
                                 + " to the total number of parameters of each MMC")
        
        # Insert the values as given in the array
        for ii in range(0,len(samp_array_manip),5):
            jj:int = math.floor(ii/5)

            curMMC:MMC = self.__list_of_MMC[jj]

            curMMC.pos_X = samp_array_manip[ii]
            curMMC.pos_Y = samp_array_manip[ii+1]
            curMMC.angle =samp_array_manip[ii+2]
            curMMC.length = samp_array_manip[ii+3]
            curMMC.thickness = samp_array_manip[ii+4]

            if repair_level ==1 or repair_level == 2:
                curMMC.repair_MMC(self.__nelx,self.__nely,repair_level=repair_level,
                                  min_thickness=min_thickness,tol=tol)
                

    def __change_values_of_MMCs_from_unscaled_array(self, samp_array:np.ndarray,
                                                  tol:float = 0.5, min_thickness:float = 1.0,
                                                  repair_level:int=2,**kwargs)->None:
        '''
        Given an array of values (in the same format as the array of properties)
        change the values of each MMC based on the values of the received array
        by parameter

        Inputs:
        - samp_array: array with new values (result from optimisation/modification process)
        - repair_level: level of repair desired in case the parameters given on the array
                        lie outside the admissible range
        - kwargs: keyword arguments (tolerance, min_thickness)
        '''

        if self._zero_valued== True:
            self._zero_valued = False

        # Modify the pre-score
        self._pre_score = np.nan

        # Check if the mode is just for Lamination Parameter modification. If so raise an error
        if self.mode == "LP":
            raise PermissionError("The mode is set to 'LP'. Then, the MMC cannot be modified!")
        
        # Check if the repair level is an acceptable value
        if not (repair_level ==0 or repair_level == 1 or repair_level == 2):
            raise ValueError("The repair level should be an integer equal to 0, 1 or 2")

        # Check the sample array has the number of elements required to modify all the properties
        # of each of the MMCs

        samp_array_manip:np.ndarray = samp_array.ravel()

        # Check if the array contains all the required number of attributes to modify each MMC
        if samp_array_manip.size != (5*len(self.list_of_MMC)):
            raise AttributeError("The size of the array does not correspond" 
                                 + " to the total number of parameters of each MMC")
        

        for ii in range(0,len(samp_array_manip),5):
            jj:int = math.floor(ii/5)

            curMMC:MMC = self.list_of_MMC[jj]

            # Change the values
            curMMC.change_pos_X_from_scaled_value(samp_array_manip[ii],self._pos_X_norm)
            curMMC.change_pos_Y_from_scaled_value(samp_array_manip[ii+1],self._pos_Y_norm)
            curMMC.change_angle_from_scaled_value(samp_array_manip[ii+2],self._angle_norm)
            curMMC.change_length_from_scaled_value(samp_array_manip[ii+3],self._length_norm)
            curMMC.change_thickness_from_scaled_value(samp_array_manip[ii+4],self._thickness_norm)
            
            if repair_level ==1 or repair_level == 2:
                curMMC.repair_MMC(self.nelx,self.nely,repair_level=repair_level,
                                  min_thickness=min_thickness,tol=tol)
            
            
    def __modify_lamination_parameters_from_array(self,new_LM_array:np.ndarray,scaled:bool)->None:
        '''
        Modify the lamination parameters of the Design from given array.
        The array should hold 3 elements at least.

        Inputs:
        - new_LM_array: array with the new values of lamination parameters
        - scaled: boolean variable controlling whether the values are scaled or not
        '''

        if self.mode == OPT_MODES[0]:
            raise AttributeError("The optimisation mode is just set to topology optimization \n" +
                                 "Therefore the lamination parameters cannot be modified")
        else:

            # Check the score is not nan
            # if not np.isnan(self.__score_FEA):
            #     self.__score_FEA = np.nan
            
            # if type(new_LM_array) != np.ndarray:
            #     new_LM_array = np.array(new_LM_array)
            
            # For safety flatten the array and check if the number of elements is equal to 3
            new_LM_array = new_LM_array.ravel()

            if len(new_LM_array) != 3:
                raise ValueError("The number of elements of new array shall be equal to 3")
            
            # Modify values if scaled value is used
            if scaled:
                new_LM_array = [new_LM_array[0],new_LM_array[1]/0.5-1,new_LM_array[2]/0.5-1]
        
            # Apply the "repair" operator
            if new_LM_array[0] < 0.0 or new_LM_array[0] > 1.0:
                if new_LM_array[0] < 0.0:
                    new_LM_array[0] = 0.0
                else:
                    new_LM_array[0] = 1.0
            
            if new_LM_array[1] < -1.0 or new_LM_array[1] > 1.0:
                if new_LM_array[1] < -1.0:
                    new_LM_array[1] = -1.0
                else:
                    new_LM_array[1] = 1.0
            
            if new_LM_array[2] < -1.0 or new_LM_array[2] > 1.0:
                if new_LM_array[2] < -1.0:
                    new_LM_array[2] = -1.0
                else:
                    new_LM_array[2] = 1.0
            
            # Assign the values of the array
            self.__VR = new_LM_array[0]
            self.__V3_1 = new_LM_array[1]
            self.__V3_2 = new_LM_array[2]
    
    def modify_mutable_properties_from_array(self,new_properties_array:np.ndarray,
                                             scaled:bool,repair_level:int=2)->None:
        '''
        Method to replace the array of properties of a design based on a list with 
        all the properties to modify

        Inputs:
        - new_properties_array: array with the replaceable properties
        - scaled: Boolean determining if the modified array is scaled or not
        - repair_level: Value between 0 and 2 defining the level of repair
        '''

        if not isinstance(new_properties_array,np.ndarray):
            raise ValueError("The properties table is not of type 'numpy.ndarray'")
        
        if not isinstance(scaled,bool):
            raise ValueError("The variable 'scaled' must be of type 'bool'")
        
        # Modify the score
        #self.__score_FEA = np.nan
        
        # For safety, set the array to be a flat 
        new_properties_array_mod:np.ndarray = new_properties_array.ravel()

        if new_properties_array_mod.size != self.problem_dimension:
            raise ValueError("The dimension of the array is different than the dimension of " + 
                             "allowed mutable properties")
        
        if self.mode == "TO+LP":
                # Split the array into two
                newArr:np.ndarray = new_properties_array_mod[0:new_properties_array_mod.size-3]
                otherArr:np.ndarray = new_properties_array_mod[new_properties_array_mod.size-3:
                                                               new_properties_array_mod.size]

                # Modify the Lamination Parameters
                self.__modify_lamination_parameters_from_array(otherArr,scaled)
                # Modify the values of MMC parameters
                if scaled:           
                    self.__change_values_of_MMCs_from_unscaled_array(newArr,repair_level=repair_level)              
                else:
                    self.__change_values_of_MMCs_from_array(newArr,repair_level=repair_level)

                # Recompute the topology
                self._topo = self._topo.from_floating_array(self._density_mapping(self.E0,self.Emin))

        elif self.mode == "TO":
            if scaled:
                self.__change_values_of_MMCs_from_unscaled_array(new_properties_array_mod,
                                                                    repair_level=repair_level)  
            else:                
                self.__change_values_of_MMCs_from_array(new_properties_array_mod,repair_level=repair_level)
            
            # Recompute the topology
            self._topo = self._topo.from_floating_array(self._density_mapping(self.E0,self.Emin))

        else:
            # Modify the Lamination Parameters
            self.__modify_lamination_parameters_from_array(new_properties_array_mod,scaled)
            
    
    
    
    



        