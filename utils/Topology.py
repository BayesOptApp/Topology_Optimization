'''
Reinterpretation of code for Structural Optimisation
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

'''

# ------------------------------------------------------------------------------------------------------
# -------------------------------- LIBRARIES -----------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

import numpy as np # Import numpy

# Import the dataclass method
from dataclasses import dataclass


# ------------------------------------------------------------------------------------------------------
# ----------------------------------- CLASS DEFINITION -------------------------------------------------
# ------------------------------------------------------------------------------------------------------

@dataclass
class Topology:
    '''
    Class holder to define a topology 
    '''

    def __init__(self,binary_topo:np.ndarray, E0:float =1.0, Emin:float = 1e-09)-> None:
        '''
        Class constructor method

        Inputs:
        - binary_topo
        - Emin: Elastic modulus of empty density element
        - E0: Reference elastic modulus of full density element
        '''
        if not isinstance(binary_topo,np.ndarray):
            raise ValueError("The binary topology is not of a numpy array")
        
        # Check there are just two entries of the topology matrix 
        if len(binary_topo.shape) !=2:
            raise ValueError("The topology is not 2-Dimensional")
        
        # Check entries
        if binary_topo.dtype == float or binary_topo.dtype == int:

            binary_topo_copy_1 = np.round(binary_topo,0).astype(int)
            unique_vals:np.ndarray = np.unique(binary_topo_copy_1)

            if not unique_vals.size <= 2:
                raise ValueError("The binary topology given as input has more than two entries")
        
        elif binary_topo.dtype == bool:
            binary_topo_copy_1 = binary_topo.astype(int)

        else:
            raise("The datatype given in the array is not numeric")
        
        # Check the E0 and Emin parameters are positive
        if E0 <= 0 or Emin <= 0:
            raise ValueError("The E0 and/or and Emin parameters should be positive")
        
        if Emin >= E0:
            raise ValueError("E0 should be more than Emin")

        # Store the topology
        self.__TO_mat = binary_topo_copy_1.copy()

        # Store the properties
        self.__E0:float = E0
        self.__Emin:float = Emin

    
    ## -------------------------------------------------------------------------------------------------
    ## ------------ Property definitions ---------------------------------------------------------------
    ## -------------------------------------------------------------------------------------------------

    @property
    def nelx(self)->int:
        '''
        Number of elements in x-direction
        '''
        return self.__TO_mat.shape[1]
    
    @property
    def nely(self)->int:
        '''
        Number of elements in y-direction
        '''
        return self.__TO_mat.shape[0]
    
    @property
    def Emin(self)->float:
        '''
        Elastic modulus of empty density element
        '''
        return self.__Emin
    
    @Emin.setter
    def Emin(self,new_Emin:float)->None:
        '''
        Setter function to set a new Emin value
        '''
        if isinstance(new_Emin,float) and new_Emin > 0.0:
            if new_Emin > self.__E0:
                raise ValueError("The new value of Emin should be less than E0")
            else:
                self.__Emin = new_Emin
        else:
            raise ValueError("The Emin given must be a positive number!")
    
    @property
    def E0(self)->float:
        '''
        Reference elastic modulus of full density element
        '''
        return self.__E0
    
    @E0.setter
    def E0(self,new_E0:float)->None:
        '''
        Setter function to set a new E0 value
        '''
        if isinstance(new_E0,float) and new_E0 > 0.0:
            if new_E0 < self.__Emin:
                raise ValueError("The new E0 value should be more than Emin")
            else:
                self.__E0 = new_E0
        else:
            raise ValueError("The E0 given must be a positive number!")

    @property
    def TO_mat(self)->np.ndarray:
        '''
        Topology array
        '''
        return self.__TO_mat.copy()
    
    ## -------------------------------------------------------------------------------------------------
    ## ------------------------------- Class methods ---------------------------------------------------
    ## -------------------------------------------------------------------------------------------------

    @classmethod
    def from_floating_array(cls,floating_arr:np.ndarray):
        '''
        Creates a new topology based on a numerical 2D numpy array given by parameter.

        Inputs:
        - floating_arr: Array of floating numbers (should have only two different entries)
        '''
        if not isinstance(floating_arr,np.ndarray):
            raise ValueError("The array given is not a numpy array")
        
        if not (floating_arr.dtype==float or floating_arr.dtype == int):
            raise ValueError("The datatype of the array is not numerical")
        
        # Check there are just two entries of the topology matrix 
        if len(floating_arr.shape) !=2:
            raise ValueError("The array is not 2-Dimensional")
        
        unique_vals:np.ndarray = np.unique(floating_arr)

        E_0:float = np.amax(unique_vals)
        Emin:float = np.amin(unique_vals)

        if abs(E_0 - Emin) > 1e-12 and unique_vals.size ==2:
            binary_arr:np.ndarray = floating_arr > Emin
        else:
            if unique_vals < 2:
                #Apply some repair
                if E_0 < 1.0:
                    binary_arr:np.ndarray = np.zeros_like(floating_arr,dtype=int)
                    E_0 = 1.0
                elif Emin > 0:
                    binary_arr:np.ndarray = np.ones_like(floating_arr,dtype=int)
                    Emin = 1e-09
            else:
                raise ValueError("There are more than two different entries")

        # Call the constructor and return
        return cls(binary_arr,E_0,Emin)
    
    @classmethod
    def from_file(cls,file_path:str):
        '''
        Extracts an array saved in a file, whose path is given by parameter and 
        generates a new Topology entity

        Inputs:
        - file_path: string with the given path of the array
        '''

        # Check the file path is an instance of string
        if not isinstance(file_path,str):
            raise ValueError("The file path should be of type string")
        
        # Load the array
        try:
            floating_arr:np.ndarray = np.load(file=file_path,
                                     fix_imports=True)
        except OSError:
            print("The file: ", file_path, " could not be read")
        except:
            print("Something else went wrong when reading the file")
        else:
            # If no errors occurred, then use the array to generate a new instance
            if not isinstance(floating_arr,np.ndarray):
                raise ValueError("The array given is not a numpy array")
        
            if not (floating_arr.dtype==float or floating_arr.dtype == int):
                raise ValueError("The datatype of the array is not numerical")
            
            # Check there are just two entries of the topology matrix 
            if len(floating_arr.shape) !=2:
                raise ValueError("The array is not 2-Dimensional")
            
            unique_vals:np.ndarray = np.unique(floating_arr)

            E_0:float = np.amax(unique_vals)
            Emin:float = np.amin(unique_vals)

            if abs(E_0 - Emin) > 1e-12 and unique_vals.size ==2:
                binary_arr:np.ndarray = floating_arr > Emin
            else:
                if unique_vals < 2:
                    #Apply some repair
                    if E_0 < 1.0:
                        binary_arr:np.ndarray = np.zeros_like(floating_arr,dtype=int)
                        E_0 = 1.0
                    elif Emin > 0:
                        binary_arr:np.ndarray = np.ones_like(floating_arr,dtype=int)
                        Emin = 1e-09
                else:
                    raise ValueError("There are more than two different entries")

            # Call the constructor and return
            return cls(binary_arr,E_0,Emin)

    ## -------------------------------------------------------------------------------------------------
    ## ----------------------------- Member functions --------------------------------------------------
    ## -------------------------------------------------------------------------------------------------
    
    def return_floating_topology(self)->np.ndarray:
        '''
        Returns a binary array showing the presence of material
        '''
        
        pos_arr:np.ndarray = np.array(self.__TO_mat==1).astype(int)
        neg_arr:np.ndarray = np.array(self.__TO_mat==0).astype(int)

        result_arr:np.ndarray = pos_arr*self.__E0 + neg_arr*self.__Emin

        return result_arr.astype(float)
    
    
    def compute_volume_ratio(self)-> float:
        """
        This function is a member which given this topology computes the 
        volume ratio of itself (proportion of filled elements)

        Returns
        -------
        volume_ratio (float): Value between 0 to 1 which is the infill
        ratio of the topology.

        """
        
        # Extract the binary topology
        binary_array:np.ndarray = self.__TO_mat.copy()
        
        # Compute the total number of elements
        n_elem:int = self.nelx * self.nely
        
        
        # Count the number of "ones" in the binary topology
        counter:int = np.sum(binary_array,axis=None,dtype=int)
        
        return counter/n_elem
    

    def compute_elementwise_gradient_norm(self):
        """
        This function computes the norm of the gradient per each element.
        According to the notation of Guo, this will be the norm of the gradient of the resulting heaviside function
        operated over the Level-Set Function phi.
        """

        # Compute the gradient of the Topology matrix in both directions (-x and -y)
        gradient_y:np.ndarray = np.gradient(self.__TO_mat,axis=0)
        gradient_x:np.ndarray = np.gradient(self.__TO_mat,axis=1)

        # Compute the Magnitude and the angle (based on the Canny Filter for edge in images)
        gradient_magnitude:np.ndarray = np.hypot(gradient_x,gradient_y)
        angle_tan:np.ndarray = np.arctan2(gradient_y,gradient_x)
        
        # Normalize the Gradient
        normalized_gradient_magnitude:np.ndarray = gradient_magnitude/np.max(gradient_magnitude,axis=None)
        
        #Convert the angle in degrees (for viewing purposes)
        #angle_tan_2:np.ndarray = np.rad2deg(angle_tan)
        
        return normalized_gradient_magnitude, angle_tan

        
        


