'''
Reinterpretation of code for Structural Optimisation
Based on work by @Elena Raponi

@Authors:
    - Elena Raponi
    - Ivan Olarte Rodriguez

'''

# import usual Python libraries
import numpy as np
import math

# Import DataClasses
from dataclasses import dataclass


# ----------------------------------------------------------------------------------------------------
# ---------------------------------------------CONSTANTS----------------------------------------------
# ----------------------------------------------------------------------------------------------------

# Names to check for optimisation of Lamination Parameters
KEY_NAMES = ("VR","V3_1","V3_2")


# Default optimization modes
OPT_MODES = ('TO','TO+LP','LP')

# Default scalation modes
SCALATION_MODES = ("Bujny","unitary")

#------------------------------------------------------------------------------------------------------
#-----------------------------------------HELPER FUNCTIONS---------------------------------------------
#------------------------------------------------------------------------------------------------------

def local_level_set_function(x_ref:np.ndarray, y_ref:np.ndarray, posX:float, 
                             posY:float, angle:float, length:float, thickness:float,m:float=6.0)->np.ndarray:
    
    '''
    Compute the local level set function of the given MMC
    Inputs:
        - x_ref: Array of x - positions of centroids of the Finite Elements
        - y_ref: Array of y - positions of centroids of the Finite Elements 
        - posX: x-position of the centroid of a corresponding MMC
        - posY: y-position of the centroid of a corresponding MMC
        - angle: angle of corresponding MMC
        - length: length of corresponding MMC
        - thickness: thickness of corresponding MMC
        - m: exponent controlling the LSF (set to 6.0 by default)
    
    Outputs:
        - array for corresponding evaluation of level set function of the MMC
    '''
    
    aux1:np.ndarray = np.multiply(np.cos(angle),np.subtract(x_ref,posX)) + np.multiply(np.sin(angle),np.subtract(y_ref,posY))

    # Avoid the warning when the length is zero
    with np.errstate(divide='ignore'):
        res:np.ndarray = np.divide(aux1,0.5*length)
    
    res[np.isnan(res)] = np.inf
    aux2:np.ndarray = res
    del res

    aux3:np.ndarray = np.power(aux2,m)

    aux4:np.ndarray = np.multiply(-np.sin(angle),np.subtract(x_ref,posX)) + np.multiply(np.cos(angle),np.subtract(y_ref,posY))

    # Avoid the warning when the thickness is zero
    with np.errstate(divide='ignore'):
        res = np.divide(aux4,0.5*thickness)

    res[np.isnan(res)] = np.inf
    aux5:np.ndarray = res
    del res
    
    
    aux6:np.ndarray = np.power(aux5,m)

    aux7:np.ndarray = aux3 + aux6

    aux8:np.ndarray = -1*(aux7-np.ones_like(aux7,dtype=float))

    # ----------------------------- MATLAB Formula --------------------------------------------------------------------------------------------------------------
    # return -((((cos(angle).*(x-posX) + sin(angle).*(y-posY))./(0.5.*length)).^m + ((-sin(angle).*(x-posX) + cos(angle).*(y-posY))./(0.5.*thickness)).^m) - 1.0)
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

    return aux8



@dataclass
class MMC:
    
    """
    Class Definition for each Individual MMC
    """
    def __init__(self,pos_X:float=0.0,pos_Y:float=0.0,length:float=0.0,angle:float=0.0,
                 thickness:float=0.0):
        '''
        Constructor of the class
        Inputs:
            - pos_X: Unscaled X position of the centroid of the MMC
            - pos_Y: Unscaled Y position of the centroid of the MMC
            - length: Unscaled Length of the MMC
            - angle: Unscaled angle of the MMC
            - thickness: Unscaled thickness of the MMC
            - mode: Optimisation Mode ("TO" or "TO+LP")
            - **kwargs: keyworded arguments in case the Optimisation mode is set to 'TO+LP'
        '''
        
        # Store the unscaled parameter values into the member variables/properties
        self.__pos_X:float = pos_X
        self.__pos_Y:float = pos_Y
        self.__angle:float = angle
        self.__length:float = length
        self.__thickness:float = thickness
        
    
    # System Section
    def __str__(self)->str:
        return ('MMC with: X_pos: {self.pos_X}; Y_pos: {self.pos_Y}; angle: {self.angle}; ' + 
                'length: {self.length}; thickness: {self.thickness}').format(self=self)
    
    def __repr__(self)->str:
       return 'MMC(%.5f, %.5f, %.5f, %.5f, %.5f)' % (self.__pos_X, self.__pos_Y, self.__angle,
                                                     self.__length, self.__thickness) 

    def __iter__(self):
        self.__n = 0
        return self
    
    def __next__(self):
        return_val:int = -1
        if self.__n <= 4:
            
            if self.__n == 0:
                return_val = self.__pos_X
            elif self.__n == 1:
                return_val = self.__pos_Y
            elif self.__n == 2:
                return_val = self.__angle
            elif self.__n == 3:
                return_val = self.__length
            else:
                return_val = self.__thickness
            self.__n += 1
            return return_val
        else:
            raise StopIteration

    '''
    Section of member functions returning member variables
    '''

    @property
    def pos_X(self)->float:
        return self.__pos_X
    
    def get_scaled_pos_X(self,pos_X_ref:float)->float:
        return self.__pos_X/pos_X_ref
    
    @property
    def pos_Y(self)->float:
        return self.__pos_Y
    
    def get_scaled_pos_Y(self,pos_Y_ref:float)->float:
        return self.__pos_Y/pos_Y_ref
    
    @property
    def angle(self)->float:
        return self.__angle
    
    def get_scaled_angle(self,angle_ref:float)->float:
        return self.__angle/angle_ref
    
    @property
    def length(self)->float:
        return self.__length
    
    def get_scaled_length(self,length_ref:float)->float:
        return self.__length/length_ref
    
    @property
    def thickness(self)->float:
        return self.__thickness
    
    def get_scaled_thickness(self,thickness_ref:float)->float:
        return self.__thickness/thickness_ref
    
    
    """
    Return all parameters
    """
    
    def return_all_parameters(self)->np.ndarray:
        
        """
        Return all basic parameters of the class
        """

        return np.array([self.__pos_X,self.__pos_Y,self.__angle,self.__length,
                self.__thickness])

    
    def return_all_scaled_parameters(self,
                                     pos_X_ref:float,
                                     pos_Y_ref:float,
                                     angle_ref:float,
                                     length_ref:float,
                                     thickness_ref:float)->np.ndarray:
        
        """
        Return all basic parameters of the class
        """
        

        return np.array([self.get_scaled_pos_X(pos_X_ref),
                self.get_scaled_pos_Y(pos_Y_ref),
                self.get_scaled_angle(angle_ref),
                self.get_scaled_length(length_ref),
                self.get_scaled_thickness(thickness_ref)])

    
    '''
    Section of modification of individual member variables
    '''
    
    @pos_X.setter
    def pos_X(self,new_pos_X:float)->None:
        if isinstance(new_pos_X, float):
            self.__pos_X = new_pos_X
   
    def change_pos_X_from_scaled_value(self,new_scaled_pos_X:float,
                                         pos_X_ref:float)->None:
        self.__pos_X = new_scaled_pos_X*pos_X_ref
        
    @pos_Y.setter
    def pos_Y(self,new_pos_Y:float)->None:
        if isinstance(new_pos_Y, float):
            self.__pos_Y = new_pos_Y 
    
    def change_pos_Y_from_scaled_value(self,new_scaled_pos_Y:float,
                                         pos_Y_ref:float)->None:
        self.__pos_Y = new_scaled_pos_Y*pos_Y_ref       
    
    @angle.setter
    def angle(self,new_angle:float)->None:
        self.__angle = new_angle
  
    def change_angle_from_scaled_value(self,new_scaled_angle:float,angle_ref:float)->None:
        self.__angle = new_scaled_angle*angle_ref

    @length.setter    
    def length(self,new_length:float)->None:
        self.__length = new_length

    def change_length_from_scaled_value(self,new_scaled_length:float,
                                          length_ref:float)->None:
        self.__length = new_scaled_length*length_ref
    
    @thickness.setter
    def thickness(self, new_thickness:float)->None:
        self.__thickness = new_thickness
    

    def change_thickness_from_scaled_value(self,new_scaled_thickness:float,
                                          thickness_ref:float)->None:
        self.__thickness = new_scaled_thickness*thickness_ref
        
    
    def change_parameters(self,**kwargs)->None:
        for key, value in kwargs.items():
            
            if key == 'pos_X':
                self.__pos_X = value
            elif key == 'pos_Y': 
                self.__pos_Y = value
            elif key == 'angle':
                self.__angle = value
            elif key == 'length':
                self.__length = value
            elif key == 'thickness':
                self.__thickness = value
            else:
                raise NameError("Key '{0}' not represented as a parameter \n".format(key))
    
    def repair_MMC(self, nelx:int, nely:int, repair_level:int = 2, 
                   min_thickness:float = 1.0, tol:float = 0.5)->None:
        '''
        This function repairs the MMC in case required to ensure the parameters yields a MMC
        within the boundaries.

        Inputs:
        - nelx: Number of Finite Elements in x-direction
        - nely: Number of Finite Elements in y-direction
        - repair_level: Repair level to be performed; Integer between 1 or 2
        - min_thickness: Minimum Thickness to be considered
        - tol: tolerance to set by in case the MMC is out of bounds
        
        '''

        # Check if the repair level is within bounds
        if repair_level > 2 or repair_level < 0:
            raise ValueError("The repair level is badly set")
        
        # Perform the repairs
        if repair_level == 1: # Just the thickness is corrected

            if self.__thickness < min_thickness:
                self.__thickness = min_thickness

            # Repair the length 
            if self.__length < tol:
                self.__length = tol

            # Repair angle
            if self.__angle > np.pi:
                self.__angle = self.angle-np.pi
            elif self.__angle < 0:
                self.__angle = np.pi - np.pi

        elif repair_level == 2:

            # Correct the thickness
            if self.__thickness < min_thickness:
                self.__thickness = min_thickness
            
            # Repair the length 
            if self.__length < tol:
                self.__length = tol

            # Repair angle
            if self.__angle > np.pi:
                self.__angle = self.angle-np.pi
            elif self.__angle < 0:
                self.__angle = np.pi - np.pi

            # Compute the ends
            x_pos_end_1:float = self.__pos_X - 0.5*self.__length*math.cos(self.__angle)
            y_pos_end_1:float = self.__pos_Y - 0.5*self.__length*math.sin(self.__angle)
            end_1:np.ndarray = np.array([x_pos_end_1,y_pos_end_1]).ravel()

            x_pos_end_2:float = self.__pos_X + 0.5*self.__length*math.cos(self.__angle)
            y_pos_end_2:float = self.__pos_Y + 0.5*self.__length*math.sin(self.__angle)
            end_2:np.ndarray = np.array([x_pos_end_2,y_pos_end_2]).ravel()

            # Check if the MMC is within bounds
            if end_1[0]< -tol: end_1[0]=-tol
            elif end_1[0]> nelx +tol: end_1[0]= nelx +tol 

            if end_1[1]< -tol: end_1[1]=-tol
            elif end_1[1]> nely +tol: end_1[1]= nely +tol

            if end_2[0]< -tol: end_2[0]=-tol
            elif end_2[0]> nelx +tol: end_2[0]= nelx +tol 

            if end_2[1]< -tol: end_2[1]=-tol
            elif end_2[1]> nely +tol: end_2[1]= nely +tol

            # Store the new values
            self.__pos_X = (end_1[0]+end_2[0])/2.0
            self.__pos_Y = (end_1[1]+end_2[1])/2.0
            self.__length = math.sqrt((end_1[0]-end_2[0])**2.+(end_1[1]-end_2[1])**2.)

            # Correct the value of the angle
            if (abs(end_1[0]-end_2[0])+abs(end_1[1]-end_2[1])) > 1e-12:
                self.__angle = math.acos((end_1[0]-end_2[0])/self.__length)
                if abs(math.sin(self.__angle) - (end_1[1]-end_2[1])/self.__length) > 1e-5:
                    self.__angle= math.pi-self.__angle
            else:
                self.__angle = 0.0

    

    
    def get_MMC_local_level_set_function(self,x_ref:np.ndarray,y_ref:np.ndarray)->np.ndarray:
        '''
        Compute the local level set function for the actual MMC
        Inputs:
        - x_ref: Array with the x-position of the centroids of the finite elements
        - y_ref: Array with the y-position of the centroids of the finite elements
        ''' 
        lsf:np.ndarray = local_level_set_function(x_ref=x_ref, y_ref=y_ref, posX=self.__pos_X, 
                             posY=self.__pos_Y, angle=self.__angle, length=self.__length,
                             thickness=self.__thickness)
        
        return lsf



