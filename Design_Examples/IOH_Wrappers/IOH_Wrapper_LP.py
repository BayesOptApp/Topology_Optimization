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
from torch import Tensor
import math

# import the copy library
from copy import copy, deepcopy

# Import evaluate FEA function
#from FEA import evaluate_FEA, return_element_midpoint_positions, compute_number_of_joined_bodies, compute_number_of_joined_bodies_2
#from FEA import compute_objective_function

# Import DataClasses
from dataclasses import dataclass

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

        # Initialize the IOH class dependency
        ioh.problem.RealSingleObjective.__init__(self,
            name=self.problem_name()+ "_" + prob_aux_name,
            n_variables=self.problem_dimension,
            instance=0,
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

        # Set the penalty factor
        weight_volume_penalty_factor:float = 0.05/min_vol_frac_penalty

        # Register the different constraints
        constr1:RealConstraint = RealConstraint(self.dirichlet_boundary_condition, name="Dirichlet Boundary Condition",
                                                        enforced=ioh.ConstraintEnforcement.HIDDEN,
                                                        weight =  200, 
                                                        exponent=1.0)
        constr2:RealConstraint = RealConstraint(self.neumann_boundary_condition, name="Neumann Boundary Condition",
                                                        enforced=ioh.ConstraintEnforcement.HIDDEN,
                                                        weight =  200, 
                                                        exponent=1.0)
        constr3:RealConstraint = RealConstraint(self.connectivity_condition, name="Connectivity Condition",
                                                        enforced=ioh.ConstraintEnforcement.HIDDEN,
                                                        weight =  200, 
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

        # Loop all over the first 3 constraints
        penalty_array = []
        for i in range(3):
            penalty_array.append(self.constraints[i].penalty())
        
        pen_sum = sum(penalty_array)

        if pen_sum > 0:
            # If the penalty is greater than 0, then the target is not computed
            # and the penalty is returned
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
                                             avoid_computation_for_not_compliance=False)
        
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
    
