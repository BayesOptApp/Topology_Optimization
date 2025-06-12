from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from meshers.MeshGrid2D import MeshGrid2D

class AbstractBoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions in Finite Element Method.
    """

    @abstractmethod
    def __init__(self, location: Union[Tuple[float, float], Tuple[Tuple[float, float]]]):
        """
        Initialize the boundary condition.

        Args:
            location (Union[Tuple[float, float], Tuple[Tuple[float, float]]]): The location of the boundary condition.
            value (Union[float, List[float]]): The value(s) of the boundary condition.
        """
        self.location = location

    @property
    def location(self)-> Union[Tuple[float, float],Tuple[Tuple[float,float]]]:
        r"""
        Return the location of the boundary condition.

        Returns:
            1)  A tuple representing a point in 2D space (x, y).
            2)  A tuple of tuples representing multiple points in 2D space ((x1, y1), (x2, y2),
                This assumes that there is a distribution of points where the boundary condition is applied and 
                the tuple are defining endpoints of a line segment.
            """
        return self._location

    @location.setter
    def location(self, value: Union[Tuple[float, float], Tuple[Tuple[float, float]]]):
        """
        Set the location of the boundary condition.

        Args:
            value (Union[Tuple[float, float], Tuple[Tuple[float, float]]]): The new location.
        """
        if not isinstance(value, (tuple, list)):
            raise TypeError("Location must be a tuple or list of tuples.")
        
        if isinstance(value, tuple) and len(value) == 2:
            if not all(isinstance(coord, float) for coord in value):
                raise ValueError("Location must be a tuple of two numeric values (x, y).")
        
        elif isinstance(value, list):
            if not all(isinstance(coord, tuple) and len(coord) == 2 and all(isinstance(c, float) for c in coord) for coord in value):
                raise ValueError("Location must be a list of tuples, each containing two numeric values (x, y).")

            elif isinstance(value, tuple) and all(isinstance(coord, tuple) and len(coord) == 2 for coord in value):
                if not all(isinstance(c, float) for coord in value for c in coord):
                    raise ValueError("Each coordinate in the location must be a tuple of two numeric values (x, y).")
        
        # Assign the location
        self._location = value
    
    @abstractmethod
    def get_affected_nodes(self, meshgrid:MeshGrid2D) -> List[int]:
        """
        Get the indices of the nodes affected by this boundary condition.

        Args:
            meshgrid (MeshGrid2D): The mesh grid containing the nodes.

        Returns:
            list: List of node indices affected by this boundary condition.
        """

        # This method should be implemented in subclasses to return the correct node indices
        pass


    @property
    def affected_nodes_list(self) -> List[int]:
        """
        Get the list of affected nodes by this boundary condition.

        Returns:
            list: List of node indices affected by this boundary condition.
        """
        return self._affected_nodes_list
    
    @affected_nodes_list.setter
    def affected_nodes_list(self, value: List[int]):
        """
        Set the list of affected nodes by this boundary condition.

        Args:
            value (List[int]): The new list of affected node indices.
        """
        if not isinstance(value, list):
            raise TypeError("Affected nodes must be a list of integers.")
        if not all(isinstance(node, int) for node in value):
            raise ValueError("All elements in the affected nodes list must be integers.")
        self._affected_nodes_list = value
    


    # @abstractmethod
    # def apply(self, system_matrix, rhs_vector):
    #     """
    #     Apply the boundary condition to the system matrix and right-hand side vector.

    #     Args:
    #         system_matrix: The global stiffness (or system) matrix to modify.
    #         rhs_vector: The global right-hand side vector to modify.
    #     """
    #     pass

    # @abstractmethod
    # def get_dofs(self):
    #     """
    #     Return the degrees of freedom (DOFs) affected by this boundary condition.

    #     Returns:
    #         list: List of DOF indices.
    #     """
    #     pass