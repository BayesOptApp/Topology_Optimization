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
from torch import Tensor
import math

# import the copy library
from copy import copy, deepcopy

# Import evaluate FEA function
#from FEA import evaluate_FEA, return_element_midpoint_positions, compute_number_of_joined_bodies, compute_number_of_joined_bodies_2
#from FEA import compute_objective_function


# Import Typing library
from typing import List, Tuple, Union, Optional

# Import the MMC Library
from geometry_parameterizations.MMC import MMC

# Import the Topology library
from utils.Topology import Topology

# Import IOH Real Problem
import ioh
from ioh.iohcpp import RealConstraint

# Import the Boundary Condition List
from boundary_conditions import BoundaryConditionList, LineDirichletBC, LineNeumannBC, PointNeumannBC, PointDirichletBC

# Import the Initialization
from utils.Initialization import prepare_FEA

from Design_Examples.Raw_Design.Design_LP import Design_LP, OPT_MODES, CONTINUITY_CHECK_MODES
from Design_Examples.utils.FEA import COST_FUNCTIONS

import time


class Design_LP_IOH_Wrapper(Design_LP,ioh.problem.RealSingleObjective):
    r"""
    This is a double class inherited object which will merge the attributes
    from the Design class and the ioh Real Single Objective 
    """

    def __init__(self, 
                 nmmcsx:int, 
                 nmmcsy:int, 
                 nelx:int, 
                 nely:int,
                 instance:int = 0, 
                 VR: float= 0.5, 
                 V3_list:List[float] = [0.0, 0.0],
                 volfrac:float = 0.5,
                 mode:str = OPT_MODES[0], 
                 symmetry_condition:bool = False, 
                 scalation_mode:str = "unitary",  
                 E0:float = 1.0, 
                 Emin:float = 1e-9,
                 use_sparse_matrices:bool = True,
                 plot_variables:bool= True,
                 cost_function:str = "compliance",
                 run_:int = 0,
                 continuity_check_mode:Optional[str]=CONTINUITY_CHECK_MODES[0],
                 boundary_conditions_list:Optional[BoundaryConditionList]=None,
                 interpolation_points:Optional[List[Tuple[Union[float,int], Union[float,int]]]] = [(0,0), (1,0.5)],
                 material_properties_dict:Optional[dict] = None,
                 standard_weight:Optional[float] = 200.0,
                 **kwargs):
        
        r"""
        The initializer of this class initializes the same variables as the `Design` class
        and set ups the conditions to handle the solver properties and plotting handles.

        ----------
        Inputs:
            - nmmcsx: number of Morphable Moving Components (MMCs) in x-direction
            - nmmcsy: number of Morphable Moving Components (MMCs) in y-direction
            - nelx: number of finite elements in x-direction
            - nely: number of finite elements in y-direction
            - mode: Optimisation mode: 'TO', 'LP' or 'TO+LP'
            - VR: VR parameter set for Lamination
            - V3_1: V3_1 parameter set for Lamination
            - V3_2: V3_2 parameter set for Lamination
            - volfrac: The volume limit to set
            - symmetry_condition: Impose a symmetry condition on the design on the x-axis.
                                  If the symmetry condition is imposed, only half of the 
                                  supposed MMC's are saved.
            - initialise_zero: Initialise the table of attributes as zeros
            - add_noise: boolean to control if noise is added to default initialisation
            - scalation_mode: Select a scalation mode: Set values for 'Bujny' or 'unitary'
            - Emin: Setting of the Ersatz Material; to be numerically close to 0
            - E0: Setting the Material interpolator (close to 1)
            - use_sparse_matrices: Check to use sparse matrices to run the optimisation algorithm
            - plot_variables: set to plot the variables generated in the process
            - cost_function: the definition of the cost function to compute the target (so far only two options)
            - run_: the run number of the problem
            - continuity_check_mode: the mode to check the continuity of the design
            - boundary_conditions_list: a list of boundary conditions to apply to the design
            - interpolation_points: a list of points to interpolate the design
            - material_properties_dict: a dictionary with the material properties to use
            - standard_weight: the standard weight to use for the constraints
        """

        # Get the kwargs
        if kwargs is not None:
            if isinstance(kwargs,dict):
                # If the kwargs is a dictionary, then unpack it
                kwargs_copy = kwargs.copy()
            else:
                raise ValueError("The kwargs must be a dictionary with the additional parameters")
        
        # Get the problem auxiliary name from the kwargs
        prob_aux_name:str = kwargs_copy.pop("problem_aux_name", "")

        
        # This initialises the Design_LP Class
        Design_LP.__init__(self,nmmcsx=nmmcsx, 
                         nmmcsy=nmmcsy, 
                         nelx=nelx, 
                         nely=nely, 
                         VR=VR, 
                         V3_List=V3_list, 
                         mode=mode, 
                         symmetry_condition=symmetry_condition, 
                         scalation_mode=scalation_mode, 
                         initialise_zero=True, 
                         add_noise=False, 
                         E0=E0, 
                         Emin=Emin,
                         continuity_check_mode=continuity_check_mode,
                         boundary_conditions_list=boundary_conditions_list,
                         interpolation_points=interpolation_points,
                         material_properties_dict= material_properties_dict,
                         **kwargs)
        
        # Append the fractional volume constraint
        self.__volfrac:float = volfrac
        bounds = ioh.iohcpp.RealBounds(self.problem_dimension, 0.0, 1.0)
        optimum = ioh.iohcpp.RealSolution([0]* self.problem_dimension, 0.0)

        # Initialize the `actual_volume_excess`
        self.actual_volume_excess = 0.0

        if prob_aux_name == "":
            # If the problem auxiliary name is not set, then use the problem name
            partial_new_name = self.problem_name
        else:
            # If the problem auxiliary name is set, then use it
            partial_new_name =  self.problem_name + "_" + prob_aux_name

        # Initialize the IOH class dependency
        ioh.problem.RealSingleObjective.__init__(self,
            name= partial_new_name,
            n_variables=self.problem_dimension,
            instance=instance,
            is_minimization=True,
            bounds= bounds,
            optimum=optimum
        )

        # Set the number of actual function evaluations
        self._n_evals:int = 0
        
        self.__use_sparse_matrices:bool = use_sparse_matrices
        self.__plot_variables:bool = plot_variables

        if cost_function.lower() in COST_FUNCTIONS:
            self.__cost_function:str = cost_function
        else:
            raise ValueError(f"The cost function '{cost_function}' is not part of the allowed cost functions")
        
        # Submit the run
        self.__run:int = run_

         # Compute the minimal possible volume fraction penalty
        min_vol_frac_penalty = 1/self.nelx/self.nely

        assert standard_weight > 0 and isinstance(standard_weight,(int,float)), "The standard weight must be greater than 0.0"

        # Store the standard weight
        self._standard_weight:float = standard_weight

        # Set the penalty factor
        weight_volume_penalty_factor:float = standard_weight/min_vol_frac_penalty

        if self.symmetry_condition_imposed:
            weight_volume_penalty_factor = weight_volume_penalty_factor/2.0

        # Register the different constraints
        constr1:RealConstraint = RealConstraint(self.dirichlet_boundary_condition, name="Dirichlet Boundary Condition",
                                                        enforced=ioh.ConstraintEnforcement.HIDDEN,
                                                        weight =  standard_weight, 
                                                        exponent=1.0)
        constr2:RealConstraint = RealConstraint(self.neumann_boundary_condition, name="Neumann Boundary Condition",
                                                        enforced=ioh.ConstraintEnforcement.HIDDEN,
                                                        weight =  standard_weight, 
                                                        exponent=1.0)
        constr3:RealConstraint = RealConstraint(self.connectivity_condition, name="Connectivity Condition",
                                                        enforced=ioh.ConstraintEnforcement.HIDDEN,
                                                        weight =  standard_weight, 
                                                        exponent=1.0)
        constr4:RealConstraint = RealConstraint(self.volume_fraction_cond, name="Volume Fraction Condition",
                                                        enforced=ioh.ConstraintEnforcement.HIDDEN,
                                                        weight =  weight_volume_penalty_factor, 
                                                        exponent=1.0)
        # This part will automatically initialize the pointers to the constraints
        super(Design_LP,self).add_constraint(constr1)
        super(Design_LP,self).add_constraint(constr2)
        super(Design_LP,self).add_constraint(constr3)
        super(Design_LP,self).add_constraint(constr4)
    
    # This is a re-definition of the create function from IOH
    def create(self, id, iid, dim):
        raise NotImplementedError
    
    def add_constraint(self, constraint):
        raise NotImplementedError("This function is restricted for this kind of object")
    
    def reset(self)->None:
        # Call the super-class reset function
        super(Design_LP,self).reset()

        # Reset the number of evaluations
        self._n_evals = 0
    
    # Set the constraint functions
    # Start with the Dirichlet Boundary Condition

    def dirichlet_boundary_condition(self,x:np.ndarray):
        """
        This function corresponds to a typical signature to wrap.

        ------------ 
        Inputs:
        x: An array with the current new input. This array is normalised [0,1] 
        """

        if self.problem_dimension != x.size:
            raise ValueError("The dimension of the input does not match the dimension of the problem!")

        # Change the variable dependency
        x_arr:np.ndarray = np.array(x).ravel()
        
        # For the sake of the properties do not perform reparation (just meant
        # for CMA-ES)
        self.modify_mutable_properties_from_array(x_arr,scaled=True,repair_level=0)
        

        resp = self.dirichlet_boundary_conditions_compliance()

        return resp
    

    # Now the Neumann Boundary Condition
    def neumann_boundary_condition(self, x:np.ndarray):
        """
        This function corresponds to a typical signature to wrap.
        
        Inputs:
        ------------
        - x: An array with the current new input. This array is normalised [0,1] 
        """

        if self.problem_dimension != x.size:
            raise ValueError("The dimension of the input does not match the dimension of the problem!")

        

        # Change the variable dependency
        x_arr:np.ndarray = np.array(x).ravel()
        
        # For the sake of the properties do not perform reparation (just meant
        # for CMA-ES)
        self.modify_mutable_properties_from_array(x_arr,scaled=True,repair_level=0)

        resp = self.neumann_boundary_conditions_compliance()

        return resp
    
    # Now the Connectivity Condition
    def connectivity_condition(self, x:np.ndarray):
        """
        This function corresponds to a typical signature to wrap.

        ------------ Inputs:
        x: An array with the current new input. This array is normalised [0,1] 
        """

        if self.problem_dimension != x.size:
            raise ValueError("The dimension of the input does not match the dimension of the problem!")

        # Change the variable dependency
        x_arr:np.ndarray = np.array(x).ravel()
        
        # For the sake of the properties do not perform reparation (just meant
        # for CMA-ES)
        self.modify_mutable_properties_from_array(x_arr,scaled=True,repair_level=0)


        # Compute the connectivity condiction compliance
        resp = self.continuity_check_compliance()


        return resp
    
    # Now the volume fraction
    def volume_fraction_cond(self,x:np.ndarray):
        """
        This function corresponds to a typical signature to wrap.

        ------------ Inputs:
        x: An array with the current new input. This array is normalised [0,1] 
        """

        if self.problem_dimension != x.size:
            raise ValueError("The dimension of the input does not match the dimension of the problem!")

        # Change the variable dependency
        x_arr:np.ndarray = np.array(x).ravel()
        
        # For the sake of the properties do not perform reparation (just meant
        # for CMA-ES)
        self.modify_mutable_properties_from_array(x_arr,scaled=True,repair_level=0)
    

        # Compute the number of disjoint bodies of the design
        resp = self.volume_constrain_violation(volfrac_=self.__volfrac)
        
        
        return resp
    
    def compute_actual_volume_excess(self,x:np.ndarray)->float:
        """
        This function computes the actual volume excess of the design
        given the current design.

        ---------------
        Inputs:
        - x (`np.ndarray`): an array with the input of the problem to evaluate the target.

        ---------------
        Output:
        - target (`float`): target value evaluation (raw)
        """

        # Change the variable dependency
        x_arr:np.ndarray = np.array(x).ravel()
        
        # For the sake of the properties do not perform reparation (just meant
        # for CMA-ES)
        self.modify_mutable_properties_from_array(x_arr,scaled=True,repair_level=0)

        # Compute the actual objective
        target = super().compute_actual_volume_excess(volfrac_=self.volfrac)

        return target
    
    
    def evaluate(self, x:np.ndarray)->float:
        """
        This is an overload of the default
        evaluate function

        ---------------
        Inputs:
        - x (`np.ndarray`): an array with the input of the problem to evaluate the target.

        ---------------
        Output:
        - target (`float`): target value evaluation (raw)
        """

        # Update the volume fraction
        self.actual_volume_excess = self.compute_actual_volume_excess(x)

        # Start the timer
        start_time = time.perf_counter()

        # Loop all over the first 3 constraints
        penalty_array = []
        for i in range(3):
            penalty_array.append(self.constraints[i].penalty())
        
        pen_sum = sum(penalty_array)

        if pen_sum > 1e-12:
            # If the penalty is greater than 0, then the target is not computed
            # and the penalty is returned

            # Stop the timer
            self.evaluation_time = time.perf_counter() - start_time
            return pen_sum
        
        # Change the variable dependency
        x_arr:np.ndarray = np.array(x).ravel()
        
        # For the sake of the properties do not perform reparation (just meant
        # for CMA-ES)
        self.modify_mutable_properties_from_array(x_arr,scaled=True,repair_level=0)

        # Compute the actual objective
        target = self.evaluate_FEA_design(volfrac=self.volfrac,
                                             iterr=self.state.evaluations+1,
                                             run_ = self.current_run,
                                             sample=1,
                                             use_sparse_matrices=self.use_sparse_matrices,
                                             plotVariables=self.plot_variables,
                                             cost_function=self.cost_function,
                                             penalty_factor=0.0,  # This is for not computing the penalty
                                             avoid_computation_for_not_compliance=False,
                                             )
        
        # Extract the evaluation time
        self.evaluation_time = time.time() - start_time
        # Update the number of evaluations
        self._n_evals += 1

        return target
    
    def __call__(self, x:Union[np.ndarray,Tensor])->float:

        #TODO: This is a temporary solution to avoid the error of the ioh
        # library. The idea is to use the evaluate function from the ioh
        # library, which is the one that will be used in the future.

        if isinstance(x,Tensor):
            # Convert the tensor to a numpy array
            x = x.detach().tolist()
        elif isinstance(x, np.ndarray):
            # Convert the numpy array to a list
            x = x.tolist()

        
        value = super().__call__(x)

        return value
    #def enforce_bounds(self, weight, enforced, exponent):
    #    return super(Design,self).enforce_bounds(weight, enforced, exponent)
    
    @staticmethod
    def get_transformed_constraint_type(type_int:int)->object:
        
        r"""
        This is a static method, which will act as a both a class helper and 
        for setting the corresponding constraints manually by the user for any
        type of constrained optimization algorithm.

        -------------------
        Inputs:
        - type_int (`int`): An integer that takes the values of the set {1,2,3,4}
                          which identify each of the different Constraint Enforcement
                          types in the IOH context.
        
        -------------------
        Outputs:
        - Out: An object referred to any constant from `ioh.iohcpp.ConstraintEnforcement`
                        
        """

        ### ----------------------
        ### INPUT CHECKS ---------
        ### ----------------------
        if not isinstance(type_int,int):
            raise ValueError("The input must be an integer")
        
        if not type_int in (1,2,3,4,5):
            raise ValueError("The input is not included in the set {0}".format((1,2,3,4,5)))
        
        # Now define the output
        if type_int ==1:
            return ioh.ConstraintEnforcement.NOT
        elif type_int==2:
            return ioh.ConstraintEnforcement.HIDDEN
        elif type_int ==3:
            return ioh.ConstraintEnforcement.SOFT
        elif type_int ==4:
            return ioh.ConstraintEnforcement.HARD
        elif type_int ==5:
            return ioh.ConstraintEnforcement.OVERRIDE
    
    def change_values_of_MMCs_from_unscaled_array(self, samp_array:np.ndarray,
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
        
        # Update the Topology
        self._topo = self._topo.from_floating_array(self._density_mapping(self.E0,self.Emin))
        

    def convert_defined_constraint_to_type(self, iddx:int,new_type:int)->None:
        r"""
        This function sets a new type for a constraint given by an index.

        -------------
        Inputs:
        - iddx (`int`): Integer from the set {0,1,2,3} to identify each of, the 4 constraints stored in the problem.
        - new_type (`int`): Integer from the set {1,2,3,4} to map according to the type of constraint.
        """

        # Perform the Input Validation
        if iddx not in (0,1,2,3):
            raise ValueError("The index is not from the set{0}".format((0,1,2,3)))
        
        else:

            # Set the constraint type
            self.constraints[iddx].enforced = self.get_transformed_constraint_type(new_type)
        



    ### --------------------------------------
    ### Properties
    ### --------------------------------------

    @property
    def volfrac(self)->float:
        return self.__volfrac
    
    @volfrac.setter
    def volfrac(self,new_volfrac:float)->None:
        if ((isinstance(new_volfrac,float) or isinstance(new_volfrac,int))):
            if new_volfrac > 0 or new_volfrac <=1:
                # Set the value in this case
                self.__volfrac = new_volfrac
            else:
                raise ValueError(f"The value of the fractional volume is {new_volfrac}, which is not between 0 and 1")
        else:
            raise ValueError(f"The fractional volume is not of a numerical type; it is {type(new_volfrac)}")
        
    
    @property
    def use_sparse_matrices(self)->bool:
        return self.__use_sparse_matrices
    
    @use_sparse_matrices.setter
    def use_sparse_matrices(self, new_definition:bool)->None:
        # Reinterpet the input as some boolean (in case is an integer)
        new_definition = bool(new_definition)

        # Set the new value
        self.__use_sparse_matrices = new_definition
    
    @property
    def plot_variables(self)->bool:
        return self.__plot_variables
    
    @plot_variables.setter
    def plot_variables(self,new_definition)->None:
        # Reinterpet the input as some boolean (in case is an integer)
        new_definition = bool(new_definition)

        # Set the new value
        self.__plot_variables = new_definition

    
    @property
    def cost_function(self)->str:
        return self.__cost_function
    
    @cost_function.setter
    def cost_function(self,new_definition:str)->None:

        # Ensure the new definition is a string value
        if isinstance(new_definition,str) and new_definition in COST_FUNCTIONS:
            self.__cost_function = new_definition

        else:
            raise ValueError("This value is not allowed")
        
    @property
    def current_run(self)->int:
        if not isinstance(self.__run,int):
            raise AttributeError("The current run variable is not an integer")
        return self.__run
    
    @current_run.setter
    def current_run(self,new_run:int)->None:
        if not isinstance(self.__run,int) or new_run < 0 :
            raise AttributeError("The new setting must be an integer and gerater than 0.")
        else:
            self.__run = new_run

    @property
    def n_evals(self)->int:
        r"""
        Return the number of function evaluations-so-far.
        """
        return self._n_evals
    
    @property
    def evaluation_time(self)->float:
        """
        This property returns the time spent in the last evaluation.
        """
        return self._evaluation_time
    
    @evaluation_time.setter
    def evaluation_time(self,new_time:float)->None:
        """
        This property sets the time spent in the last evaluation.
        """
        if isinstance(new_time,float) and new_time >= 0:
            self._evaluation_time = new_time
        else:
            raise ValueError("The evaluation time must be a positive float value")
    
    @property
    def actual_volume_excess(self)->float:
        """
        This property returns the actual volume excess of the design
        given the current design.
        """
        return self._actual_volume_excess
    
    @actual_volume_excess.setter
    def actual_volume_excess(self,new_value:float)->None:
        """
        This property sets the actual volume excess of the design
        given the current design.
        """
        if isinstance(new_value,float):
            self._actual_volume_excess = new_value
        else:
            raise ValueError("The actual volume excess must be a float value")
    

    @property
    def standard_weight(self)->float:
        """
        This property returns the standard weight used for the constraints.
        """
        return self._standard_weight
    
    @standard_weight.setter
    def standard_weight(self,new_weight:float)->None:
        """
        This property sets the standard weight used for the constraints.
        """
        if isinstance(new_weight,float) and new_weight > 0:
            self._standard_weight = new_weight

            # Change the weight of the "Constraint Set"
            for idx in range(len(self.constraints)):
                actual_constraint:ioh.iohcpp.RealConstraint = self.constraints[idx]
                if idx < 3:
                    # For the first three constraints, set the standard weight
                    actual_constraint.weight = new_weight
                else:
                    weight_volume_penalty_factor:float = new_weight/( 1/self.nelx/self.nely) #0.05/min_vol_frac_penalty

                    if self.symmetry_condition_imposed:
                        weight_volume_penalty_factor = weight_volume_penalty_factor/2.0


        else:
            raise ValueError("The standard weight must be a positive float value")
    
    
