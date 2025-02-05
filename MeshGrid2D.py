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

'''
MeshGrid Class Definition
'''
class MeshGrid2D:
    '''
    Manages the Grid Properties of the mesh. To construct the object,
    the length and height of the beam shall be specified as well as the
    height and length of the element. It is assumed the element length and
    element height are constant
    '''

    #The Class constructor
    def __init__(self,length:float,height:float,
                 element_length:float,
                 element_height:float) -> None:
        
        self.__length:float = length
        self.__height:float = height

        # Number of horizontal, vertical and total grids
        self.__grid_point_number_X:int = math.ceil(self.__length/element_length)+1
        self.__grid_point_number_Y:int = math.ceil(self.__height/element_height)+1
        self.__grid_point_number_total:int = self.__grid_point_number_X*self.__grid_point_number_Y

        # Calculate number of elements in -x and -y directions
        self.__nelx:int = self.__grid_point_number_X-1
        self.__nely:int = self.__grid_point_number_Y-1
        self.__nel_total:int = self.__nelx*self.__nely

        # Set element length and height as member variables
        self.__element_length:float = self.__length/self.__nelx
        self.__element_height:float = self.__height/self.__nely

        # Call function to set coordinate points
        self.__set_coordinate_points__()
        
        # Call function to set element data matrix
        self.__set_element_data_matrix__()
        
    
    # Set the coordinate points

    #==========================================================================
    #
    #        Nodal Coordinate System
    #
    #           y
    #           ^
    #           |______________
    #           |              |
    #           |              |
    #           |              |
    #         (0,0)------------  --> x
    #
    #==========================================================================

    # Generation of grid data matrix
    # Grid ID and x y z coordinates cartesian coordinate system
    def __set_coordinate_points__(self)->None:

        self.__coordinate_grid:np.ndarray = np.zeros((self.__grid_point_number_total,4))

        for ii in np.arange(0,self.__grid_point_number_total,1):
            
            x:float = np.fmod(ii,self.__grid_point_number_X)*self.__element_length
            y:float = math.floor((ii)/self.__grid_point_number_X)*self.__element_height

            if x < 0:
                raise Exception("Error")
            
            z:float = 0

            # The matrix of the grid data
            self.__coordinate_grid[ii,0] = ii
            self.__coordinate_grid[ii,1] = x
            self.__coordinate_grid[ii,2] = y
            self.__coordinate_grid[ii,3] = z
    
    # Set Element Data Matrix
    #==========================================================================
    #
    #             Node and Element Numbering
    #
    #         10______11______12______13______14
    #         |       |       |       |       |
    #         |   4   |   5   |   6   |   7   |
    #         |       |       |       |       |
    #         5-------6-------7-------8-------9
    #         |       |       |       |       |
    #         |   0   |   1   |   2   |   3   |
    #         |       |       |       |       |
    #         0-------1-------2-------3-------4
    #
    #==========================================================================

    # Generation of element data matrix
    # Element ID, IDs of the corner grids and orientation angle
    def __set_element_data_matrix__(self)->None:

        self.__E:np.ndarray = np.zeros((self.__nel_total,6),dtype=np.uint64)
        for ii in np.arange(0,self.__nel_total,1):
            # IDs of the corner grids
            n1:int = ii+np.fix((ii)/self.__nelx)
            n2:int = n1+1
            n3:int = n2+self.__grid_point_number_X
            n4:int = n1+self.__grid_point_number_X
            
            # The matrix of the element data
            self.__E[ii,0] = ii
            self.__E[ii,1] = n1
            self.__E[ii,2] = n2
            self.__E[ii,3] = n3
            self.__E[ii,4] = n4
            self.__E[ii,5] = 0

    # This function returns the midpoint of each element in an array
    # The structure will follow this order
    # Column 1 (0): The id of the element
    # Column 2 (1): The x-coordinate of the element
    # Column 3 (2): The y-coordinate of the element 
    # Column 4 (3): The z-coordinate of the element
    def compute_element_midpoints(self)-> np.ndarray:

        # This array will store the midpoints
        midpoints:np.ndarray = np.zeros((self.__E.shape[0],4),dtype=float)

        # Loop all over the elements
        for aa in range(self.__E.shape[0]):
            
            # Node array (with the positions per element) 
            node_array:np.ndarray = np.zeros((1,3,4))

            for bb,cc in enumerate(np.arange(1,5,1)):
                # Get the node positions
                curNode_id:int = self.__E[aa,cc]

                # Get the positions
                curPos:np.ndarray = self.__coordinate_grid[curNode_id,1:4]

                # Store the current positions
                node_array[0,:,bb] = curPos
        

            # Get the average position
            average_position:np.ndarray = np.ravel(np.mean(node_array,axis=2))

            # Store the average position
            midpoints[aa,:] = np.hstack((aa,average_position)).ravel()

        
        return midpoints
    

    def find_neighbouring_elements(self, elem_id:int)->np.ndarray:
        """
        This function returns a list of the neighbouring elements of an element given by an index indicator
        'elem_id'. The match is given by all the elements which share common nodes. The list returned does not include the 
        sought element given as a parameter. 

        ** Note: this formulation just works for quadrilaterals

        -----------
        Inputs:
        - elem_id: Integer with the reference to the element

        -----------
        Outputs:
        - elem_list: list with the elements which are neighbours
        
        """

        # perform input checks before
        if elem_id > self.nel_total-1:
            raise ValueError(f"The value of elem_id({elem_id}) exceeeds the value of total number of elements - 1 ({self.nel_total-1})")
        elif elem_id < 0:
            raise ValueError(f"The value of elem_id is negative.")

        # Get the nodes of the element received as a parameter

        nodes_elem:np.ndarray = np.ravel(self.E[np.where(self.E[:,0] == elem_id),1:5])

        # Start as an empty list
        elem_list:list = list()

        # Loop all over the element list
        for idx in range(self.E.shape[0]):

            # Check if the list is already of 8 elements
            if len(elem_list) == 8:
                break
            # Exclude the input element
            if idx == elem_id:
                continue
            else:
                # Extract the element nodes
                elem_nodes_id:np.ndarray = self.E[idx,1:5]
                
                # loop all over the nodes of the lookup node
                for node_id in elem_nodes_id.ravel():
                    if np.any(np.isin(nodes_elem,node_id)):

                        # If this condition suffices, then store the element and break the loop.
                        elem_list.append(idx)
                        break

        return np.array(elem_list,dtype=int)

    def find_neighbouring_elements_quad(self, elem_id:int)->np.ndarray:
        """
        This function returns a list of the neighbouring elements of an element given by an index indicator
        'elem_id'. The match is given by all the elements which share common nodes. The list returned does not include the 
        sought element given as a parameter. 

        ** Note: this formulation just works for regular grids

        -----------
        Inputs:
        - elem_id: Integer with the reference to the element

        -----------
        Outputs:
        - elem_list: list with the elements which are neighbours
        
        """

        # Check if the element is a corner
     
        if elem_id == 0: # Lower left corner
            elem_list = [1,self.nelx,self.nelx+1]
        elif elem_id == self.nelx-1: # Lower right corner
            elem_list = [self.nelx-2,2*self.nelx-2,2*self.nelx-1]
        elif elem_id == self.nel_total-1: # upper right corner
            elem_list = [self.nel_total-2,self.nel_total-1-self.nelx,self.nel_total-2-self.nelx]
        elif elem_id == self.nel_total-self.nelx: # upper left corner
            elem_list = [self.nel_total-self.nelx+1,self.nel_total-2*self.nelx+1,self.nel_total-2*self.nelx]
        elif elem_id > 0 and elem_id < self.nelx-1: # Check the element is in the lower side
            elem_list = [elem_id-1,elem_id+1,elem_id+self.nelx,elem_id+self.nelx+1,elem_id+self.nelx-1]
        elif elem_id > self.nel_total-self.nelx and elem_id < self.nel_total-1: # Check the element is in the upper side
            elem_list = [elem_id-1,elem_id+1,elem_id-self.nelx,elem_id-self.nelx-1,elem_id-self.nelx+1]
        elif np.remainder(elem_id,self.nelx)==0: # Check the element is in the left side
            elem_list = [elem_id+1,elem_id-self.nelx,elem_id-self.nelx+1,elem_id+self.nelx,elem_id+self.nelx+1]
        elif np.remainder(elem_id+1,self.nelx)==0: # Check the element is in the right side
            elem_list = [elem_id-1,elem_id-self.nelx,elem_id-self.nelx-1,elem_id+self.nelx,elem_id+self.nelx-1]
        else: # Everything within the padding
            elem_list = [elem_id-1,elem_id+1,elem_id-self.nelx,elem_id+self.nelx,elem_id-self.nelx-1,elem_id-self.nelx+1,elem_id+self.nelx-1,elem_id+self.nelx+1]
    

        return np.array(elem_list,dtype=int)

    def is_inside(self,spatial_pos:np.ndarray)->bool:
        """
        This function is to detect if a point given by parameter `spatial_pos'
        lies inside the mesh or not.
        """

        if spatial_pos.size != 3:
            raise Exception("The given parameter does not correspond to a point")
        else:
            spatial_pos = spatial_pos.ravel()
        

        # Loop all over the element list
        for idx in range(self.__E.shape[0]):

            # Node array (with the positions per element) 
            node_array:np.ndarray = np.zeros((1,3,4))

            for bb,cc in enumerate(np.arange(1,5,1)):
                # Get the node positions
                curNode_id:int = self.__E[idx,cc]

                # Get the positions
                curPos:np.ndarray = self.__coordinate_grid[curNode_id,1:4]

                # Store the current positions
                node_array[0,:,bb] = curPos

            # Get the extremal values
            maximum_values:np.ndarray = np.ravel(np.max(node_array,axis=2))
            minimum_values:np.ndarray = np.ravel(np.min(node_array,axis=2))

            # Compare the values if the value is inside the domain
            comp_1:np.ndarray = spatial_pos >= minimum_values
            comp_2:np.ndarray = spatial_pos <= maximum_values

            # Compare bit-wise if the arrays are true
            result_bitwise:np.ndarray = np.bitwise_and(comp_1,comp_2)


    
    # -------------------- Member Functions -------------------------------------------------------
    
    # Return the number of elements in x-direction
    @property
    def nelx(self)->int:
        return self.__nelx
    
    # Return the number of elements in y-direction
    @property
    def nely(self)->int:
        return self.__nely
    
    # Return the total number of elements of the mesh
    @property
    def nel_total(self)->int:
        return self.__nel_total
    
    @property
    def E(self)->np.ndarray:
        return self.__E.copy()
    
    # Return the coordinate grid
    @property
    def coordinate_grid(self)->np.ndarray:
        return self.__coordinate_grid.copy()
    
    @property
    def grid_point_number_X(self)->int:
        return self.__grid_point_number_X
    
    @property
    def grid_point_number_Y(self)->int:
        return self.__grid_point_number_Y
    
    @property
    def grid_point_number_total(self)->int:
        return self.__grid_point_number_total
    
    @property
    def element_height(self)->float:
        return self.__element_height
    
    @property
    def element_length(self)->float:
        return self.__element_length
    
    @property
    def total_length(self)->float:
        return self.__length
    
    @property
    def total_height(self)->float:
        return self.__height
