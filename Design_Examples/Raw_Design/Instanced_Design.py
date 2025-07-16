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
from Design_Examples.utils.FEA import (evaluate_FEA, 
                 return_element_midpoint_positions, 
                 compute_number_of_joined_bodies, 
                 compute_number_of_joined_bodies_2)

from Design_Examples.Raw_Design.Design import Design


# Import DataClasses
from dataclasses import dataclass

# Import Typing library
from typing import List, Tuple, Optional, Union

# Import the MMC Library
from geometry_parameterizations.MMC import MMC

# Import the Topology library
from utils.Topology import Topology


# Import the Initialization
from utils.Initialization import prepare_FEA

# Import all the packages from boundary conditions
from boundary_conditions import PointDirichletBC, PointNeumannBC, LineDirichletBC, LineNeumannBC, BoundaryConditionList

# Import the default material properties
from finite_element_solvers.common import (E11_DEFAULT,E22_DEFAULT,G12_DEFAULT,NU12_DEFAULT)

# Default plot Modifier dict
PLOT_MODIFIER_DEFAULT:dict = {'rotate':False,
                              'rotate_angle':0.0,
                              'scale':1.0}

# Default scalation modes
SCALATION_MODES = ("Bujny","unitary")

# Continuity check modes
CONTINUITY_CHECK_MODES = ("continuous","discrete")



class InstancedDesign(Design):
    """
    InstancedDesign is a subclass of Design that represents a design
    which can be instantiated multiple times with different parameters.
    It inherits from the Design class and can include additional methods
    or properties specific to instanced designs.
    """

    def __init__(self, 
                 nmmcsx:int, 
                 nmmcsy:int, 
                 nelx:int, 
                 nely:int,
                 instance:int=0,
                 symmetry_condition:bool=False,
                 scalation_mode:str = "unitary",
                 initialise_zero:bool=False,
                 add_noise:bool = False,
                 E0:float = 1.00,
                 Emin:float = 1e-09,
                 continuity_check_mode:Optional[str]=CONTINUITY_CHECK_MODES[0],
                 boundary_conditions_list:Optional[BoundaryConditionList]=None,
                 material_properties_dict:Optional[dict]=None,
                 **kwargs):
        """
        Initializes the InstancedDesign with a name and optional parameters.
        
        :param name: The name of the design.
        :param parameters: Optional parameters for the design.
        """
        super().__init__(nmmcsx, 
                         nmmcsy,
                         nelx, 
                         nely,
                         symmetry_condition,
                         scalation_mode,
                         initialise_zero,
                         add_noise,
                         E0,
                         Emin,
                         continuity_check_mode,
                         boundary_conditions_list,
                         material_properties_dict,
                         **kwargs)
        
        # Instance number
        assert isinstance(instance, int), "Instance must be an integer."
        assert instance >= 0, "Instance must be a non-negative integer."

        # Store the instance number 
        self._instance = instance

        # -------------------------------------------------------
        # Set the ranges for the MMCs based on the instance number
        # -------------------------------------------------------

        # Instantiate a number generator by using the instance number as a seed
        rng = np.random.default_rng(self._instance)

        # Get a random number between 0 and 0.2
        # to set the lower bound of the MMCs
        rand_lb = rng.uniform(0, 0.2, size=(4,))

        # Set the upper bound of the MMCs
        rand_ub = rand_lb + 0.8

        # Generate a random number for the angle
        rand_angle = rng.uniform(-np.pi, 2*np.pi)/np.pi

         # Generate a range of values for the MMCs
        self._range_MMCs:np.ndarray = np.array([[rand_lb[0], rand_ub[0]],  # pos_X
                                                [rand_lb[1], rand_ub[1]],  # pos_Y
                                                [0 + rand_angle, 1 + rand_angle], # angle
                                                [rand_lb[2], rand_ub[2]],  # length
                                                [rand_lb[3], rand_ub[3]]]) # thickness
    

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
        

        if scaled:
            self.__change_values_of_MMCs_from_unscaled_array(new_properties_array_mod,
                                                                repair_level=repair_level)  
        else:                
            self.__change_values_of_MMCs_from_array(new_properties_array_mod,repair_level=repair_level)
        
        # Recompute the topology
        self._topo = self._topo.from_floating_array(self._density_mapping(self.E0,self.Emin))


    def __change_values_of_MMCs_from_unscaled_array(self, samp_array:np.ndarray,
                                                  tol:float = 0.5, min_thickness:float = 1.0,
                                                  repair_level:int=0,**kwargs)->None:
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

        if self.zero_valued== True:
            self.zero_valued = False

        # Modify the pre-score
        self._pre_score = np.nan

        
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
            curMMC.change_pos_X_from_scaled_value(self.range_MMCs[0,0] + 0.8*samp_array_manip[ii],self._pos_X_norm)

            curMMC.change_pos_Y_from_scaled_value(self.range_MMCs[1,0] + 0.8*samp_array_manip[ii+1],self._pos_Y_norm)
            curMMC.change_angle_from_scaled_value(self.range_MMCs[2,0] + samp_array_manip[ii+2],self._angle_norm)
            curMMC.change_length_from_scaled_value(self.range_MMCs[3,0] + 0.8*samp_array_manip[ii+3],self._length_norm)
            curMMC.change_thickness_from_scaled_value(self.range_MMCs[4,0] + 0.8*samp_array_manip[ii+4],self._thickness_norm)
            
            if repair_level ==1 or repair_level == 2:
                curMMC.repair_MMC(self.__nelx,self.__nely,repair_level=repair_level,
                                  min_thickness=min_thickness,tol=tol)
                
    
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
        self.pre_score = np.nan
        
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
        
        # Insert the values as given in the array
        for ii in range(0,len(samp_array_manip),5):
            jj:int = math.floor(ii/5)

            curMMC:MMC = self.list_of_MMC[jj]

            curMMC.pos_X = samp_array_manip[ii]
            curMMC.pos_Y = samp_array_manip[ii+1]
            curMMC.angle =samp_array_manip[ii+2]
            curMMC.length = samp_array_manip[ii+3]
            curMMC.thickness = samp_array_manip[ii+4]

            if repair_level ==1 or repair_level == 2:
                curMMC.repair_MMC(self.__nelx,self.__nely,repair_level=repair_level,
                                  min_thickness=min_thickness,tol=tol)

    @property
    def instance(self) -> int:
        """
        Returns the instance number of the design.
        
        :return: The instance number.
        """
        return self._instance
    
    @property
    def range_MMCs(self) -> np.ndarray:
        """
        Returns the range of MMCs for the design.
        
        :return: The range of MMCs as a numpy array.
        """
        return self._range_MMCs