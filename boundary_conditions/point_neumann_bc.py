from boundary_conditions.abstract_bc import AbstractBoundaryCondition
from typing import Union, Tuple, List
from meshers.MeshGrid2D import MeshGrid2D

class PointNeumannBC(AbstractBoundaryCondition):
    r"""
    Boundary condition to handle point forces.
    This class represents a point force boundary condition in a finite element system.
    It allows for the specification of a point in 2D space and the point force applied at that point in the x and y directions.
    """

    def __init__(self, location:Union[Tuple[float, float], Tuple[Tuple[float, float]]], 
                 force_vector: Tuple[float, float]):
        """
        Initialize the point displacement boundary condition.

        Args:
            location (`tuple` or `list`): Coordinates of the point in (x,y) format.
            force_vector (`tuple`): The force vector applied at the point in (fx, fy) format.
        """
        super().__init__(location)
        self.force_vector= force_vector
    
    @property
    def force_vector(self) -> Union[int, Tuple[float, ...]]:
        r"""
        Get the blocked degrees of freedom at the point.

        Returns:
            - `Union[int, Tuple[float, ...]]`: The blocked degree(s) of freedom.
        """
        return self._force_vector
    
    @force_vector.setter
    def force_vector(self, value: Tuple[float, ...]):
        r"""
        Set the force vector at the point.

        Args:
            - value `Tuple[float,...]`: The new force vector. Must be of size 2, representing (fx, fy).
        Raises:
            - `ValueError`: If the value is not a tuple of size 2 or does not contain valid force components.
            
        """
        
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("Blocked degrees of freedom must be a tuple of size 2 (fx, fy).")
        if not all(isinstance(v, (int, float)) for v in value):
            raise ValueError("Blocked degrees of freedom must contain numeric values (fx, fy).")
        self._force_vector = value
        
    # def apply(self, system):
    #     """
    #     Apply the point displacement boundary condition to the system.

    #     Args:
    #         system: The system (e.g., mesh, solver) to which the BC is applied.
    #     """
    #     # Example: system should have a method to set displacement at a point
    #     system.set_point_displacement(self.point, self.displacement)

    def __repr__(self):
        return f"PointForceBC(point={self.location}, force_vector={self.force_vector})"
    
    def get_affected_nodes(self, 
                           meshgrid:MeshGrid2D) ->  List[int]:
        r"""
        Get the indices of the nodes affected by this boundary condition.
        This method already assumes that the mesh nodes are indexed in a way that corresponds to the finite element mesh,
        where the node indices are calculated based on the number of elements in the x and y directions.
        Args:
            MeshGrid2D: `MeshGrid2D`: The mesh grid containing the nodes.
        Returns:
            list: `List[int]` of node indices affected by this boundary condition.
        """

        # Find the node(s) in the meshgrid closest to the specified location
        if isinstance(self.location[0], (float, int)):
            locations = [self.location].copy()
        else:
            locations = self.location.copy()

        # Transform the locations by scaling by the number of elements in the meshgrid
        for i, loc in enumerate(locations):
            locations[i] = (loc[0] * meshgrid.nelx, loc[1] * meshgrid.nely)

        affected_nodes = []
        for loc in locations:
            min_dist = float('inf')
            closest_node = None
            for idx, node_coords in enumerate(meshgrid.coordinate_grid):
                dist = ((node_coords[1] - loc[0]) ** 2 + (node_coords[2] - loc[1]) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_node = idx
            if closest_node is not None:
                affected_nodes.append(closest_node)
        
        self.affected_nodes_list = affected_nodes.copy()
        return affected_nodes
        