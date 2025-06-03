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
from FEA import (evaluate_FEA, 
                 return_element_midpoint_positions, 
                 compute_number_of_joined_bodies, 
                 compute_number_of_joined_bodies_2)

from FEA import compute_objective_function

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

# ----------------------------------------------------------------------------------------------------
# ---------------------------------------------CONSTANTS----------------------------------------------
# ----------------------------------------------------------------------------------------------------

# Default scalation modes
SCALATION_MODES = ("Bujny","unitary")

# Continuity check modes
CONTINUITY_CHECK_MODES = ("continuous","discrete")

# -----------------------------------------------------------------------------------------------------
# --------------------------------------------CLASS DEFINITION-----------------------------------------
# -----------------------------------------------------------------------------------------------------
class Design:
    '''
    Class which is generated to contain a design for the optimisation procedure
    and its properties
    '''
    def __init__(self, 
                 nmmcsx:int, 
                 nmmcsy:int, 
                 nelx:int, 
                 nely:int,
                 symmetry_condition:bool=False,
                 scalation_mode:str = "Bujny",
                 initialise_zero:bool=False,
                 add_noise:bool = False,
                 E0:float = 1.00,
                 Emin:float = 1e-09,
                 continuity_check_mode:Optional[str]=CONTINUITY_CHECK_MODES[0],
                 boundary_conditions_list:Optional[BoundaryConditionList]=None,
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
            - boundary_conditions_list: List of boundary conditions to be applied to the design.
            - **kwargs: keyworded arguments in case the Optimisation mode is set to 'TO+LP'
        '''
        

        # The following lines are input checks to ensure for the intersection of symmetry condition and the number of MMCs in y-direction
        if symmetry_condition and np.remainder(nmmcsy,2) !=0:
            raise ValueError("In case the symmetry condition is active, then the number of MMCs in y-direction should be an even number")

        self.__nmmcsx:int = nmmcsx
        self.__nmmcsy:int = nmmcsy
        self.__nelx:int = nelx
        self.__nely:int = nely
        self.__symmetry_condition:bool = symmetry_condition
        self.__E0:float = E0
        self.__Emin:float = Emin
        self.continuity_check_mode:str = continuity_check_mode

        if scalation_mode in SCALATION_MODES:
            self.__scalation_mode:str = scalation_mode

        else:
            raise ValueError("The scalation mode is not allowed")


        # Initialise score
        self._score_FEA:float = np.nan

        #Initialise pre-scores
        self._pre_score:float = np.nan

        lx:float = nelx/nmmcsx

        ly:float = nely/nmmcsy
        angle:float = math.atan2(ly,lx)
        length:float = math.sqrt(lx**2.0 + ly**2.0)
        thickness:float = max([1.0, 0.5*self.nelx*self.nely/
                                    (2*self.__nmmcsx*self.__nmmcsy*length)])

        
        # Store the references as class properties
        self.__default_length:float = length
        self.__default_thickness:float = thickness
        self.__default_angle:float = angle
        self.__default_lx:float = lx
        self.__default_ly:float = ly

        # Set the scalation mode
        self.scalation_mode = scalation_mode

        # Set the linked topology
        self._topo:Topology = Topology(np.zeros((nely,nelx)),E0,Emin)

        # Set the boundary conditions
        if boundary_conditions_list is None:

            self._boundary_conditions:BoundaryConditionList = BoundaryConditionList(
                LineDirichletBC((0.0, 0.0), (0.0, 1.0), 0.0),
                PointNeumannBC()

            )

        
        # Proceed with the non-zero initialisation
        if initialise_zero == False:
            
            # Run the default non zero initialisation
            self.__default_non_zero_initialisation__(add_noise)
            

            # Recompute the topology
            self._topo = self._topo.from_floating_array(self._density_mapping(E0=E0,Emin=Emin))

        else:
            # Run the zero initialisation
            self.__zero_initialisation__()
            
    
    # ------------------------------------------------------------------------
    # MEMBER INITIALISATION METHODS
    # ------------------------------------------------------------------------
    
    def __default_non_zero_initialisation__(self,
                                            add_noise:bool)->None:

        # Write again the values
        lx = self.__default_lx
        ly = self.__default_ly
        angle = self.__default_angle
        length = self.__default_length
        thickness = self.__default_thickness

        # Generate Member variable storing the MMCs
        self.__list_of_MMC:List[MMC] = []

        self._zero_valued:bool = False
        if self.__nmmcsy == 1:
            for ii in range(1,self.__nmmcsx+1):
                

                # Generate a new MMC
                mmC_1:MMC = MMC(pos_X=0.5*lx + lx*(ii-1),
                            pos_Y= 0.5*ly ,
                            angle = 0,
                            length = lx,
                            thickness=thickness)
                
                if add_noise:
                    # Add noise to the variables
                    noise_arr:np.ndarray = np.random.normal(0,0.01,size=(5,1))

                    mmC_1.pos_X += noise_arr[0,0]*mmC_1.pos_X
                    mmC_1.pos_Y += noise_arr[1,0]*mmC_1.pos_Y
                    mmC_1.angle += noise_arr[2,0]*mmC_1.angle
                    mmC_1.length += noise_arr[3,0]*mmC_1.length
                    mmC_1.thickness += noise_arr[4,0]*mmC_1.thickness
                
                # Append the MMC to the list
                self.__list_of_MMC.append(mmC_1)



        elif np.remainder(self.__nmmcsy,2)==0:
            for ii in range(1,self.__nmmcsx+1):
                for jj in range(1,math.floor(self.__nmmcsy/2)+1):
                    
                    if ii < self.__nmmcsx:
                        # Generate a new MMC
                        mmC:MMC = MMC(pos_X=0.5*lx + lx*(ii-1),
                                    pos_Y= 0.5*ly + ly*(jj-1),
                                    angle = 0,
                                    length = length,
                                    thickness=thickness)
                    else:
                        mmC:MMC = MMC(pos_X=0.5*lx + lx*(ii-1),
                                    pos_Y= 0.5*ly + ly*(jj-1),
                                    angle = 0,
                                    length = length,
                                    thickness=thickness)
                    
                    if add_noise:
                        # Add noise to the variables
                        noise_arr:np.ndarray = np.random.normal(0,0.01,size=(5,1))

                        mmC.pos_X += noise_arr[0,0]*mmC.pos_X
                        mmC.pos_Y += noise_arr[1,0]*mmC.pos_Y
                        mmC.angle += noise_arr[2,0]*mmC.angle
                        mmC.length += noise_arr[3,0]*mmC.length
                        mmC.thickness += noise_arr[4,0]*mmC.thickness
                    

                    # Append the MMC to the list
                    self.__list_of_MMC.append(mmC)
                    
                    
            
            if self.__symmetry_condition == False:

                
                # length of actual mmc
                actual_length:int = len(self.__list_of_MMC)

                
                for ii in range(actual_length):
                    actual_MMC:MMC = self.__list_of_MMC[ii] 
                    mmC_2:MMC = MMC(pos_X= actual_MMC.pos_X,
                                    pos_Y= self.__nely-actual_MMC.pos_Y,
                                    length=actual_MMC.length,
                                    angle = 0,
                                    thickness=actual_MMC.thickness)
                    
                    # Append the symmetry MMC to array
                    self.__list_of_MMC.append(mmC_2)
        
        else:
            for ii in range(1,self.__nmmcsx+1):
                for jj in range(1,self.__nmmcsy+1):

                    if ii == self.nmmcsx and jj == np.median(np.arange(1,self.__nmmcsy+1)):
      
                        # Generate a new MMC
                        mmC:MMC = MMC(pos_X=0.75*lx + lx*(ii-1),
                                    pos_Y= 0.5*ly + ly*(jj-1),
                                    angle = 0,
                                    length = 0.5*lx,
                                    thickness=thickness)
                        
                        if add_noise:
                            # Add noise to the variables
                            noise_arr:np.ndarray = np.random.normal(0,0.01,size=(5,1))

                            mmC.pos_X += noise_arr[0,0]*mmC.pos_X
                            mmC.pos_Y += noise_arr[1,0]*mmC.pos_Y
                            mmC.angle += noise_arr[2,0]*mmC.angle
                            mmC.length += noise_arr[3,0]*mmC.length
                            mmC.thickness += noise_arr[4,0]*mmC.thickness
                        

                        # Append the MMC to the list
                        self.__list_of_MMC.append(mmC)
                        
                       

                    else:
                        
                        # Generate a new MMC
                        mmC:MMC = MMC(pos_X=0.5*lx + lx*(ii-1),
                                    pos_Y= 0.5*ly + ly*(jj-1),
                                    angle = angle,
                                    length = length,
                                    thickness=thickness)
                        
                        if add_noise:
                            # Add noise to the variables
                            noise_arr:np.ndarray = np.random.normal(0,0.01,size=(5,1))

                            mmC.pos_X += noise_arr[0,0]*mmC.pos_X
                            mmC.pos_Y += noise_arr[1,0]*mmC.pos_Y
                            mmC.angle += noise_arr[2,0]*mmC.angle
                            mmC.length += noise_arr[3,0]*mmC.length
                            mmC.thickness += noise_arr[4,0]*mmC.thickness
                        

                        # Append the MMC to the list
                        self.__list_of_MMC.append(mmC)
                        
                       
    def __zero_initialisation__(self)->None:
        
        ' Perform the initialisation of all MMC to zero values'
        self.__list_of_MMC:List[MMC] = []

        self._zero_valued:bool = True
        
        # Determine the number of MMCs to generate
        
        if self.__symmetry_condition:
            total_number_MMC:int = int(self.__nmmcsx*self.__nmmcsy/2)
        else:
            total_number_MMC:int = int(self.__nmmcsx*self.__nmmcsy)
        
        # Perform a loop to generate all MMC
        for _ in range(total_number_MMC):
            # Append zero valued list
            self.__list_of_MMC.append(MMC())
    
    #-------------------------------------------------------------------------
    # Normalisation setups
    #-------------------------------------------------------------------------
    
    def __set_Bujny_normalisation__(self)->None:
        '''
        Sets the norms according to Bujny (2018).
        Better suited for Evolutionary-Mutation Approaches
        '''

        self._pos_X_norm:float = self.__nelx

        if self.__symmetry_condition == True:
            self._pos_Y_norm:float = self.__nely/2
        else:
            self._pos_Y_norm:float = self.__nely

        self._angle_norm:float = math.pi
        self._length_norm:float = 0.25*math.sqrt(self.__nelx**2+self.__nely**2)
        self._thickness_norm:float = 0.25*math.sqrt(self.__nelx**2+self.__nely**2)
    
    def __set_unitary_normalisation__(self)->None:
        '''
        Sets the norms to be unitary.
        Better suited for BO Optimisation
        '''

        self._pos_X_norm:float = self.__nelx

        if self.__symmetry_condition == True:
            self._pos_Y_norm:float = self.__nely/2
        else:
            self._pos_Y_norm:float = self.__nely
        
        self._angle_norm:float = math.pi

        ## TODO: Set the effect of this normalization
        
        #self.__length_norm:float = 4*self.__default_length
        self._thickness_norm:float = 4*self.__default_thickness

        if self.__symmetry_condition == True:
            self._length_norm:float = math.sqrt(self.__nelx**2+(self.__nely/2)**2)
        else:
            self._length_norm:float = math.sqrt(self.__nelx**2+self.__nely**2)
        #self.__length_norm:float = 0.25*math.sqrt(self.__nelx**2+self.__nely**2)
        #self.__thickness_norm:float = 0.25*math.sqrt(self.__nelx**2+self.__nely**2)

    #-------------------------------------------------------------------------
    # Member function to load a topology from file
    #-------------------------------------------------------------------------

    def load_topology_from_file(self,file_path:str)->None:
        '''
        This function receives a file from parameter and loads the file to be
        the topology to be analyzed. Automatically, the analysis type is set to
        'LP' to analyze the variation of the lamination parameters.

        Inputs:
        - file_path: path specified of a saved topology file 
        '''

        # Read the topology
        try:
            self._topo = self._topo.from_file(file_path=file_path)
        except:
            print("The topology could not be loaded")
        else:
            # Check the dimensions are the same
            if self._topo.nelx != self.__nelx or self._topo.nely != self.__nely:
                raise ValueError(f"The number of elements do not match with the ones of the design")

            # set everything to zero
            self.__zero_initialisation__()

            #Change E0 and Emin by the ones of the new topology
            self.__E0 = self._topo.E0
            self.__Emin = self._topo.Emin

            # Set the cost and pre cost to nan
            #self.__pre_score = np.nan
            #self.__score_FEA = np.nan

    
    #-------------------------------------------------------------------------
    # Member functions returning arrays with the properties of all MMCs
    #-------------------------------------------------------------------------
    
    def get_array_of_unscaled_MMC_properties(self)->np.ndarray:
        '''
        Returns a numpy array with unscaled attributes of each MMC corresponding to the design
        '''
        
        # Create the array 
        param_array:np.ndarray = np.zeros((1,5*len(self.__list_of_MMC)),
                                          dtype=float)
        
        if self._zero_valued:
            return param_array
        
        else:
            # Perform a loop all over the list of MMCs
            for ii in range(len(self.list_of_MMC)):
                # Store in the following order
                # 1. X_position
                # 2. Y_position
                # 3. Angle
                # 4. Length
                # 5. Thickness
                
                param_array[0,5*ii] = self.list_of_MMC[ii].pos_X
                param_array[0,5*ii+1] = self.list_of_MMC[ii].pos_Y
                param_array[0,5*ii+2] = self.list_of_MMC[ii].angle
                param_array[0,5*ii+3] = self.list_of_MMC[ii].length
                param_array[0,5*ii+4] = self.list_of_MMC[ii].thickness
        
            return param_array
    
    def get_array_of_scaled_MMC_properties(self)->np.ndarray:
        
        # Create the array 
        param_array:np.ndarray = np.zeros((1,5*len(self.list_of_MMC)),
                                          dtype=float)
        
        if self._zero_valued:
            return param_array
        
        else:
            # Perform a loop all over the list of MMCs
            for ii in range(len(self.list_of_MMC)):
                # Store in the following order
                # 1. X_position
                # 2. Y_position
                # 3. Angle
                # 4. Length
                # 5. Thickness
                
                scaled_arr:np.ndarray = self.list_of_MMC[ii].return_all_scaled_parameters(self._pos_X_norm,
                                                                                            self._pos_Y_norm,
                                                                                            self._angle_norm,
                                                                                            self._length_norm,
                                                                                            self._thickness_norm)
                
                param_array[0,5*ii:5*ii+5] = scaled_arr
        
            return param_array
    
    def get_array_of_unscaled_MMC_properties_with_symmetry(self)->np.ndarray:
        
        if self.__symmetry_condition == False:
            raise AttributeError("The symmetry condition is not imposed")
        else:
            # Create the array
            param_array:np.ndarray = np.zeros((1,2*5*len(self.__list_of_MMC)),
                                              dtype=float)
            
            if self._zero_valued:
                return param_array
            
            else:
            
                # Parameter to indicate the starting point of second half
                init_point:int = 5*len(self.__list_of_MMC)
                
                # Fill the first part of the properties
                # Perform a loop all over the list of MMCs
                for ii in range(len(self.__list_of_MMC)):
                    # Store in the following order
                    # 1. X_position
                    # 2. Y_position
                    # 3. Angle
                    # 4. Length
                    # 5. Thickness
                    
                    unscaled_arr:np.ndarray = self.list_of_MMC[ii].return_all_parameters()
                    
                    unscaled_arr_2:np.ndarray = np.copy(unscaled_arr)
                    
                    unscaled_arr_2[1] = self.__nely - unscaled_arr_2[1]
                    
                    unscaled_arr_2[2] = math.pi - unscaled_arr_2[2]
                    
                    # Fill the first part
                    param_array[0,5*ii:5*ii+5] = unscaled_arr
                    
                    # Fill the second part
                    param_array[0,5*ii+init_point:5*ii+5+init_point] = unscaled_arr_2
                    
                
                return param_array
    
    def __build_ghost_MMC_array_with_symmetry_conditions__(self)->List:
        if self.__symmetry_condition == False:
            raise AttributeError("The symmetry condition is not imposed!")
        
        else:
            if self._zero_valued:
                raise AttributeError("All the MMC properties are zero valued!")
            else:
                # Generate a deep copy of the list of saved MMC
                copy_list_of_MMC:List[MMC] = deepcopy(self.__list_of_MMC)
                
                # return list
                return_list_of_MMC:List[MMC] = deepcopy(self.__list_of_MMC)

                # Fill the first part of the properties
                # Perform a loop all over the list of MMCs
                for ii in range(len(self.__list_of_MMC)):
                    
                    # Extract a copy of the current MMC
                    curMMC:MMC = deepcopy(copy_list_of_MMC[ii])
                    # Modify in the copy list
                    curMMC.pos_Y = self.__nely - curMMC.pos_Y
                    
                    curMMC.angle = math.pi - curMMC.angle
                    
                    # Append element to return list
                    return_list_of_MMC.append(curMMC)

                return return_list_of_MMC.copy()


    def return_mutable_properties_in_array(self,scaled:bool=True)->np.ndarray:

        '''
        Returns the mutable properties of the Design 

        Inputs:
        - scaled: Determine to output the scaled
        '''
        
        if scaled:
            # If scaled is chosen, extract the array of scaled MMC properties 
            compl_array:np.ndarray = self.get_array_of_scaled_MMC_properties()


        else:
            # If scaled is not chosen,extract the array of unscaled MMC properties
            compl_array:np.ndarray = self.get_array_of_unscaled_MMC_properties()
 
        return compl_array

     


    def get_array_of_scaled_MMC_properties_with_symmetry(self)->np.ndarray:
        
        if self.__symmetry_condition == False:
            raise AttributeError("The symmetry condition is not imposed!")
        else:
            # Create the array
            param_array:np.ndarray = np.zeros((1,2*5*len(self.list_of_MMC)),
                                              dtype=float)
            
            if self.zero_valued:
                return param_array
            
            else:
            
                # Parameter to indicate the starting point of second half
                init_point:int = 5*len(self.list_of_MMC)
                
                # Generate a deep copy of the list of saved MMC
                copy_list_of_MMC:List[MMC] = deepcopy(self.list_of_MMC)
                
                # Fill the first part of the properties
                # Perform a loop all over the list of MMCs
                for ii in range(len(self.list_of_MMC)):
                    
                    # Modify in the copy list
                    copy_list_of_MMC[ii].pos_Y =self.__nely - self.list_of_MMC[ii].pos_Y
                    
                    copy_list_of_MMC[ii].angle = math.pi- self.list_of_MMC[ii].angle
                    
                    # Store in the following order
                    # 1. X_position
                    # 2. Y_position
                    # 3. Angle
                    # 4. Length
                    # 5. Thickness
                    
                    unscaled_arr:np.ndarray = self.list_of_MMC[ii].return_all_scaled_parameters(pos_X_ref= self._pos_X_norm,
                                                                                                  pos_Y_ref= self._pos_Y_norm,
                                                                                                  angle_ref=self._angle_norm,
                                                                                                  length_ref=self._length_norm,
                                                                                                  thickness_ref=self._thickness_norm
                                                                                                  )
                    
                    unscaled_arr_2:np.ndarray = copy_list_of_MMC[ii].return_all_scaled_parameters(pos_X_ref= self._pos_X_norm,
                                                                                                  pos_Y_ref= self._pos_Y_norm,
                                                                                                  angle_ref=self._angle_norm,
                                                                                                  length_ref=self._length_norm,
                                                                                                  thickness_ref=self._thickness_norm
                                                                                                  )
                    
                    # Fill the first part
                    param_array[0,5*ii:5*ii+5] = unscaled_arr
                    
                    # Fill the second part
                    param_array[0,5*ii+init_point:5*ii+5+init_point] = unscaled_arr_2
                    
                
                return param_array
    
    # ------------------------------------------------------------------------------------------------
    # DENSITY MAPPING FUNCTION 
    # ------------------------------------------------------------------------------------------------
     
    def _density_mapping(self,E0:float=1.0,Emin:float=1.0e-09)->np.ndarray:
        r'''
        Density-based mapping of the global level-set function on the FE mesh.

        Inputs:
        --------
            - Emin: Elastic modulus of empty density element.
            - E0: Reference elastic modulus of full density element
        '''

        #x_ref,y_ref = np.meshgrid('''0.5:1.0:nelx-0.5''' np.arange(), '''0.5:1.0:nely-0.5''' np.arange())
        # Generate the meshgrid
        x_ref,y_ref = np.meshgrid(np.arange(start=0.5,step=1.0,stop=self.__nelx),  
                                 np.arange(start=0.5,step=1.0,stop=self.__nely))
        
        # Call the mesh to generate the sampling positions (in a general setting)
        #trial_ = return_element_midpoint_positions(self.topology,Emin,E0)
        #x_ref= np.reshape(trial_[:,1],(self.nely,self.nelx))
        #y_ref= np.reshape(trial_[:,2],(self.nely,self.nelx))
        
        
        if self.__symmetry_condition == False:
            mmC_complete_list:List[MMC] = self.list_of_MMC[:]
        else:
            mmC_complete_list:List[MMC] = self.__build_ghost_MMC_array_with_symmetry_conditions__()
        
        # Generate array to know the results

        L_arr:np.ndarray = np.zeros((self.__nely,self.__nelx,len(mmC_complete_list)))

        # Loop all over the list of MMC
        for ii in range(len(mmC_complete_list)):
            L_arr[:,:,ii] = mmC_complete_list[ii].get_MMC_local_level_set_function(x_ref=x_ref,
                                                                                   y_ref=y_ref)
        
        xPhys:np.ndarray=np.sign(np.amax(L_arr, axis = 2))*E0

        xPhys[xPhys < 0] = Emin

        return xPhys
    
    # ------------------------------------------------------------------------
    # Evaluate Finite Element Method with current Design
    # ------------------------------------------------------------------------

    # This is the default function used by supposing the material is isotropic
    def evaluate_pre_cost_FEA_design(self,volfrac:float,KE:np.ndarray,iK:np.ndarray,
                                     jK:np.ndarray,F:sparse.coo_matrix,
                                     U:np.ndarray,freedofs:np.ndarray,
                                     edofMat:np.ndarray,
                                     penalty_factor:float = 0.0,
                                     avoid_computation_for_not_compliance:bool=True)->float:
        
        '''
        This function computes the pre-cost to evaluate then the FEA cost function
        
        Inputs:
        - volfrac: Fractional mass of the Element
        - KE: Element stiffness matrix
        - iK: rows to store non-zero values of global stiffness matrix
        - jK: columns to store non-zero values of global stiffness matrix
        - F: Force vector
        - U: vector of displacements (with all DOFs)
        - freedofs: Free (unconstrained) degrees of freedom
        - edofmat: matrix of degrees of freedom
        - Emin: Elastic modulus of full density element
        - E0: Reference elastic modulus of non full density element
        - penalty_factor: penalty factor to penalize the design if this exceeds the volume constraint.
        - avoid_computation_for_not_compliance: This variable is to avoid computation of the function.

        Outputs:
        - cost: floating value with the pre-cost assessment
        '''

        if avoid_computation_for_not_compliance:
            # Get the compliance array
            comp_array: np.ndarray = self.identify_natural_constraints_violation()

            # Check if any of the constraints are violated
            if np.any(comp_array==False):
                # Update the pre cost
                self._pre_score = 1e20
                return 1e20

        # Compute the density mapping
        xPhys:np.ndarray = self._topo.return_floating_topology()
        xPhys_arr:np.ndarray = np.ravel(xPhys,order='F')
        aux:np.ndarray = self.__Emin+np.multiply(xPhys_arr.reshape(-1,1),self._topo.E0
                                                 -self._topo.Emin)
        KE_aux:np.ndarray = np.ravel(KE,order='F')
        sK:np.ndarray =  np.expand_dims(KE_aux.flatten(order='F'),axis=1) @ (np.expand_dims(aux.flatten(order='F'),axis=1).T)
        del aux, KE_aux

        # Compute displacements
        K = sparse.coo_matrix((sK.flatten(order='F'),(iK.flatten(order='F'),jK.flatten(order='F'))),
                       shape=(2*(self.nely+1)*(self.nelx+1),2*(self.nely+1)*(self.nelx+1))).tocsc()
        U_c:np.ndarray = np.copy(U)
        aux1 = K[freedofs,:]
        aux1b = aux1[:,freedofs]
        aux2 = F.tocsr()[freedofs,0]
        answ = sparse.linalg.spsolve(aux1b,aux2)
        U_c[freedofs] = answ.reshape((-1,1))
        
        del aux1, aux1b, aux2, answ
        aux:np.ndarray = np.squeeze(U_c[edofMat],axis=2)
        aux1:np.ndarray = np.sum(np.multiply(aux @ KE,aux),1)
        ce:np.ndarray = aux1.reshape((self.nely,self.nelx),order='F')


        del aux, aux1

        # Compute compliance value
        compliance = np.sum(np.multiply(self.__Emin+xPhys*(self._topo.E0-self._topo.Emin),ce))

        cost:float = compliance + penalty_factor*max(0.0, self.compute_volume_ratio() - volfrac)
        
        # Update the pre cost
        #self.__pre_score = cost

        return cost
    
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
                self._score_FEA = 1e20
                return 1e20

        # Compute the topology optimisation matrix
        TO_mat:np.ndarray = self._topo.return_floating_topology()

        # Compute the cost of the Design
        cost:float = evaluate_FEA( TO_mat=TO_mat,
                                    iterr = iterr,
                                    sample = sample,
                                    volfrac= volfrac,
                                    Emin=self.Emin,
                                    E0=self.E0,
                                    run_=run_,
                                    sparse_matrices_solver=use_sparse_matrices,
                                    plotVariables=plotVariables,
                                    symmetry_cond=self.symmetry_condition_imposed,
                                    cost_function=cost_function,
                                    penalty_factor=penalty_factor)
            
        # Update the cost
        #self.__score_FEA = cost

        return cost

    # ------------------------------------------------------------------------
    # Member functions to return the private attributes
    # ------------------------------------------------------------------------
    
    @property
    def nmmcsx(self)->int:
        return self.__nmmcsx
    
    @property
    def nmmcsy(self)->int:
        return self.__nmmcsy
    
    @property
    def nelx(self)->int:
        return self.__nelx
    
    @property
    def nely(self)->int:
        return self.__nely
    

    def problem_name(self)->str:
        r"""
            This function returns the Full_Name of the problem given the 
            mode
        """
        
        return "Topology_Optimization_MMC"
    
    @property
    def zero_valued(self)->bool:
        return self._zero_valued
    
    @zero_valued.setter
    def zero_valued(self,zero_valued:bool)->None:
        '''
        Sets the zero valued initialisation to True or False
        '''
        if not isinstance(zero_valued,bool):
            raise ValueError("The zero_valued parameter should be a boolean")
        else:
            self._zero_valued = zero_valued

    @property
    def list_of_MMC(self):
        return copy(self.__list_of_MMC)
    
    
    @property
    def symmetry_condition_imposed(self)->bool:
        return self.__symmetry_condition
    
    @property
    def score_FEA(self)->float:
        return self._score_FEA
    
    @property
    def pre_score(self)->float:
        return self._pre_score
    
    @property
    def scalation_mode(self)->str:
        '''
        Property defining the kind of scalation to use for computations
        '''
        return self.__scalation_mode
    
    @scalation_mode.setter
    def scalation_mode(self,new_scalation_mode:str)->None:
        '''
        Sets a new scalation mode
        '''
        if not isinstance(new_scalation_mode,str):
            raise ValueError("The new scalation mode should be a string")
        else:
            if new_scalation_mode not in SCALATION_MODES:
                raise ValueError("The new_scalation_mode is not allowed")
            else:
                if new_scalation_mode == "Bujny":
                    # Run the Bujny Setting Function
                    self.__set_Bujny_normalisation__()            
                elif new_scalation_mode == "unitary":
                    self.__set_unitary_normalisation__()

    @property
    def problem_dimension(self)-> int:
        '''
        Returns the dimension of the optimisation problem
        '''

        return len(self.__list_of_MMC)*5

    
    @property
    def topology(self)->np.ndarray:
        '''
        Returns the inherent topology of the design
        '''

        return self._topo.return_floating_topology()
    
    @property
    def Emin(self)->float:
        '''
        Returns the Emin parameter to define the topology
        '''

        return self.__Emin
    
    @Emin.setter
    def Emin(self,new_Emin:float)->None:
        '''
        Modifies the Emin parameter to define the topology

        Inputs:
        - new_Emin: new Emin parameter (floating point number)
        '''
        #Check if there is no error setting the new Emin
        try:
            self._topo.Emin = new_Emin
        except:
            print("Some error occurred")
        else:
            # Change the value of the Design Class
            self.__Emin = new_Emin

    @property
    def E0(self)->float:
        '''
        Returns the E0 parameter to define the topology
        '''

        return self.__E0
    
    @E0.setter
    def E0(self,new_E0:float)->None:
        '''
        Modifies the Emin parameter to define the topology

        Inputs:
        - new_Emin: new Emin parameter (floating point number)
        '''
        #Check if there is no error setting the new Emin
        try:
            self._topo.E0 = new_E0
        except:
            print("Some error occurred")
        else:
            # Change the value of the Design Class
            self.__E0 = new_E0
    
    @property
    def continuity_check_mode(self)->str:
        '''
        Returns the continuity check mode
        '''
        return self.__continuity_check_mode
    
    @continuity_check_mode.setter
    def continuity_check_mode(self,new_mode:str)->None:
        '''
        Sets the continuity check mode

        Inputs:
        - new_mode: `str`: new mode to set 
        '''
        if not isinstance(new_mode,str):
            raise ValueError("The new scalation mode should be a string")
        else:
            if new_mode.strip().lower() not in CONTINUITY_CHECK_MODES:
                raise ValueError("The new scalation_mode is not allowed")
            else:
                self.__continuity_check_mode = new_mode.strip().lower()
    

    
    
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
        self.pre_score = np.nan
        
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
        if samp_array_manip.size != (5*len(self.__list_of_MMC)):
            raise AttributeError("The size of the array does not correspond" 
                                 + " to the total number of parameters of each MMC")
        

        for ii in range(0,len(samp_array_manip),5):
            jj:int = math.floor(ii/5)

            curMMC:MMC = self.__list_of_MMC[jj]

            # Change the values
            curMMC.change_pos_X_from_scaled_value(samp_array_manip[ii],self._pos_X_norm)
            curMMC.change_pos_Y_from_scaled_value(samp_array_manip[ii+1],self._pos_Y_norm)
            curMMC.change_angle_from_scaled_value(samp_array_manip[ii+2],self._angle_norm)
            curMMC.change_length_from_scaled_value(samp_array_manip[ii+3],self._length_norm)
            curMMC.change_thickness_from_scaled_value(samp_array_manip[ii+4],self._thickness_norm)
            
            if repair_level ==1 or repair_level == 2:
                curMMC.repair_MMC(self.__nelx,self.__nely,repair_level=repair_level,
                                  min_thickness=min_thickness,tol=tol)
        
        # +++++
        #for ii in range(len(self.__list_of_MMC)):
        #    print(self.__list_of_MMC[ii])
        # +++++
            
    
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
          
    
    def compute_volume_ratio(self)-> float:
        """
        This function extends the
        is a member which given this topology computes the volume ratio of the
        current design by extending the function of the topology object 
        associated to this design.

        Returns
        -------
        volume_ratio (float): Value between 0 to 1 which is the infill
        ratio of the topology.

        """
        
        
        return self._topo.compute_volume_ratio()
    
    
    def dirichlet_boundary_conditions_compliance(self)-> float:
        """
        This function returns a boolean value to tell whether the Dirichlet 
        Boundary conditions of the associated topology are at fulfilled at 
        least to check if the design is compliant with the imposed 
        Dirichlet boundary conditions of the problem.
        
        Returns
        -------
        A distance that is non zero if there are no elements on the boundary
        
        """
        

        if self.continuity_check_mode == "continuous":
            import utils.lc as lame_curves

            ### NOTE: JELLE WESTRA'S THESIS INJECTION
            ### IMPORT THE LIBRARIES
            lcs: List[lame_curves.LameCurveConfig] = lame_curves.construct_lame_configs(self.list_of_MMC, 
                                                                                        self.symmetry_condition_imposed, 
                                                                                        self.nely)

            d_yaxis = min(
                lame_curves.solve_distance_to_bounded_yaxis(lc, yaxis_bounds=(1/2, self.nely-1/2))[1] 
                for lc in lcs
            )
            
            ### NOTE: THIS NOTATION WAS MODIFIED
            response_yaxis = d_yaxis if (d_yaxis > 1/2) else 0.
            return response_yaxis if not np.isnan(response_yaxis) else np.sqrt(self.nelx**2 + self.nely**2)

        elif self.continuity_check_mode == "discrete":
            
            ### NOTE: JELLE WESTRA'S THESIS INJECTION
            ### IMPORT THE LIBRARIES
            from utils.lame_curves import geo_from_binary_image
            from shapely.geometry import MultiPolygon,  LineString

            ### EXTRACT THE geometry 
            geo: MultiPolygon = geo_from_binary_image(self.topology > 1/2)
        
            line: LineString = LineString([(0,0), (0,self.nely)])
            dist = geo.distance(line)
            return dist if not np.isnan(dist) else np.sqrt(self.nelx**2 + self.nely**2)
        else:

            # Get a copy of the associated topology
            assoc_topology:np.ndarray = self._topo.return_floating_topology()

            # Check there is at least one active element at the leftmost 
            
            # Given the problem is the cantilever beam, the inspection is based
            # on checking at least the left-most elements of the topology is one
            if np.sum(assoc_topology[:,0],dtype=int) >= 1:
                return 0.0
            else:
                #Start looping the topology
                distance:float = 0.0
                for ii in range(1,self.nelx):
                    if np.sum(assoc_topology[:,ii],dtype=int) >= 1:
                        distance = float(ii)
                        break
                
                return distance
        
        
    
    
    def neumann_boundary_conditions_compliance(self)-> float:
        """
        This function returns a boolean value to tell whether the Neumann 
        Boundary conditions of the associated topology are at fulfilled at 
        least to check if the design is compliant with the imposed 
        Neumann boundary conditions of the problem.
        
        Returns
        -------
        A `float`, with some distance 
        
        """

        def smallest_distance(arr, position):
            """
            Find the smallest distance from a bit at a given position in a binary 2D array.
            
            Args:
            --------------
            - arr (list of list of int): The binary 2D array.
            - position (`tuple`): A tuple (x, y) representing the position in the array.
            
            Returns:
            --------------
            - `int`: The smallest distance to a bit with value 1.
            """
            x, y = position
            rows, cols = len(arr), len(arr[0])
            
            # Ensure the position is valid
            if x < 0 or x >= rows or y < 0 or y >= cols:
                raise ValueError("Position is out of bounds.")
            
    
            # Find all positions with value 1
            ones_positions = [(i, j) for i in range(rows) for j in range(cols) if arr[i][j] == 1]
            if not ones_positions:
                return self.nelx*self.nely  # No other 1s in the array
            
            # Calculate Manhattan distance to each one
            distances = [abs(x - i) + abs(y - j) for i, j in ones_positions]
            return min(distances)
            
   
        if self.continuity_check_mode == "continuous":

            ### NOTE: JELLE WESTRA'S THESIS INJECTION
            ### IMPORT THE LIBRARIES
            import utils.lc as lame_curves

            lcs: List[lame_curves.LameCurveConfig] = lame_curves.construct_lame_configs(self.list_of_MMC, 
                                                                                        self.symmetry_condition_imposed, 
                                                                                        self.nely)
            
            pt: lame_curves.Point = lame_curves.Point(self.nelx, self.nely/2)

            d_pt = min(
                lame_curves.solve_distance_to_point(lc, pt)[1] 
                for lc in lcs
            )

            reponse_pt = d_pt if (d_pt > 1/2) else 0.
            return reponse_pt if not np.isnan(reponse_pt) else np.sqrt(self.nelx**2 + self.nely**2)
        

        elif self.continuity_check_mode == "discrete":
            ### NOTE: JELLE WESTRA'S THESIS INJECTION
            ### IMPORT THE LIBRARIES
            from utils.lame_curves import geo_from_binary_image
            from shapely.geometry import MultiPolygon, Point


            ### EXTRACT THE geometry
            geo: MultiPolygon = geo_from_binary_image(self.topology > 1/2)

            pt: Point = Point(self.nelx, self.nely/2)
            dist = geo.distance(pt)
            return dist if not np.isnan(dist) else np.sqrt(self.nelx**2 + self.nely**2)

        else:
            # Get a copy of the associated topology
            assoc_topology:np.ndarray = self._topo.return_floating_topology()
            
            # Given the problem is the cantilever beam, the inspection is based
            # on checking at least there is material in contact with the imposed 
            
            # Case for even number of elements in y-direction
            if self.nelx % 2 == 0:
                # Get the indices of all the points of interest
                interest_points = [(int(self.nely/2)-1,self.nelx-1),
                                    (int(self.nely/2),self.nelx-1)]
                
                dist_arr = [smallest_distance(assoc_topology, iTuple) for iTuple in interest_points]

                # Return the minimum distance
                return min(dist_arr)
            
            # Case for odd number of elements in y-direction
            else:
                if ((assoc_topology[np.floor(self.nely/2)-1,self.nelx-1]==1)
                or (assoc_topology[np.floor(self.nely/2),self.nelx-1]==1)
                or (assoc_topology[np.floor(self.nely/2)+1,self.nelx-1])):
                    return True
                else:
                    return False
    
    def continuity_check_compliance(self)->int:
        """
        This function computes the number of disjoint structures or level sets.
        The method is the burning sites, which calls iteratively adjoint material 
        elements.
        """
        if self.continuity_check_mode == "continuous":

            ### NOTE: JELLE WESTRA'S THESIS INJECTION
            ### IMPORT THE LIBRARIES

            from utils.lame_curves import construct_geometry
            from shapely.geometry import MultiPolygon, Point, LineString
            from scipy.sparse.csgraph import minimum_spanning_tree

            geo: MultiPolygon = construct_geometry(self.list_of_MMC, 
                                                   self.symmetry_condition_imposed, 
                                                   self.nelx, 
                                                   self.nely)

            components = list(geo.geoms)

            n = len(components)
            D = np.zeros((n,n))
            for i in range(n):
                for j in range(i) :
                        d = components[i].distance(components[j])
                        # making sure the MST understands there is an edge, for 0 it thinks it's not connected,
                        # also if the values get too low (<1e-8) this can happen
                        D[i,j] = D[j,i] = d if (d > 1e-6) else -1

            # this does not perform MST since it does not overwrite D (first its converted to sparse, it overwrites that instance)
            # TODO : make a boolean for this to activate or not
            minimum_spanning_tree(D, overwrite=True).toarray()
            D[D < 1/2] = 0
            d_MST = D.sum()

            pt: Point = Point(self.nelx, self.nely/2)
            line: LineString = LineString([(0,1/2), (0,self.nely-1/2)])

            if (d_pt := geo.distance(pt)) > 1/2 : d_MST += d_pt
            if (d_line := geo.distance(line)) > 1/2 : d_MST += d_line
            
            response_disc = d_MST
            return response_disc if not np.isnan(response_disc) else np.sqrt(self.nelx**2 + self.nely**2)
        

        elif self.continuity_check_mode == "discrete":

            ### NOTE: JELLE WESTRA'S THESIS INJECTION
            ### IMPORT THE LIBRARIES
            from utils.lame_curves import construct_geometry, geo_from_binary_image
            from shapely.geometry import MultiPolygon, Point, LineString
            from scipy.sparse.csgraph import minimum_spanning_tree

            geo: MultiPolygon = geo_from_binary_image(self.topology > 1/2)

            components = list(geo.geoms)

            n = len(components)
            D = np.zeros((n,n))
            for i in range(n):
                for j in range(i) :
                        d = components[i].distance(components[j])
                        # making sure the MST understands there is an edge, for 0 it thinks it's not connected,
                        # also if the values get too low (<1e-8) this can happen
                        D[i,j] = D[j,i] = d if (d > 1e-6) else -1
            # this does not perform MST since it does not overwrite D (first its converted to sparse, it overwrites that instance)
            # TODO : make a boolean for this to activate or not
            minimum_spanning_tree(D, overwrite=True).toarray()
            # D = minimum_spanning_tree(D).toarray()
            D[D < 1/2] = 0
            d_MST = D.sum()

            pt: Point = Point(self.nelx, self.nely/2)
            line: LineString = LineString([(0,1/2), (0,self.nely-1/2)])

            if (d_pt := geo.distance(pt)) > 1/2 : d_MST += d_pt
            if (d_line := geo.distance(line)) > 1/2 : d_MST += d_line
            
            return d_MST if not np.isnan(d_MST) else np.sqrt(self.nelx**2 + self.nely**2)
        
        else:
            # Get the topology
            topo:Topology = self.topology

            # Pass it to the function (Don't return the array with bodies)
            numBodies, _ = compute_number_of_joined_bodies_2(topo,self.Emin,self.E0)

            return numBodies
    

    def identify_natural_constraints_violation(self)->List[bool]:
        """
        This member function returns a 3-element array informing which of the "natural constraints"
        is violated by current design.
        The order is the following:
        [1]: Dirichlet Boundary Condition
        [2]: Neumann Boundary Condition
        [3]: Connectivity Condition
        """

        # Generate a zeros array
        constraints_arr = np.zeros(shape=(3,),dtype=float)

        constraints_arr[0] = self.dirichlet_boundary_conditions_compliance()
        constraints_arr[1] = self.neumann_boundary_conditions_compliance()

        # The restriction is based upon that the number of solid connected bodies of the design 
        # must be equal to 1. This condition is to avoid the generation of "floating" beams
        # which may not badly condition the stiffness matrix because of the "Ersatz material" formulation,
        # yet for practical reasons this leads to physical unfeasibility.
        constraints_arr[2] = self.continuity_check_compliance()

        return constraints_arr
    
    def volume_constrain_violation(self,volfrac_:float)->float:
        return max(self.compute_volume_ratio() - volfrac_,0)
    
    def compute_actual_volume_excess(self, volfrac_)->float:
        """
        This function computes the actual volume excess of the design
        """
        return self.compute_volume_ratio() - volfrac_
    



        