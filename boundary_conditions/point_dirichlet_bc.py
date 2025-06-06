from boundary_conditions.abstract_bc import AbstractBoundaryCondition
from typing import Union, Tuple, List
from meshers.MeshGrid2D import MeshGrid2D

class PointDirichletBC(AbstractBoundaryCondition):
    r"""
    Boundary condition to handle point displacement.
    This class represents a point displacement boundary condition in a finite element system.
    It allows for the specification of a point in 2D space and the blocked degrees of freedom (dof) at that point.
    This assumes that the displacement is 0 at the specified point, effectively clamping it in place.
    """

    def __init__(self, location:Union[Tuple[float, float], Tuple[Tuple[float, float]]], 
                 blocked_dof: Union[int, Tuple[int, ...]]):
        """
        Initialize the point displacement boundary condition.

        Args:
            location (`tuple` or `list`): Coordinates of the point in (x,y) format.
            blocked_dof (`int` or `tuple`): The degree(s) of freedom to block at the point.
        """
        super().__init__(location)
        self.blocked_dof= blocked_dof
    
    @property
    def blocked_dof(self) -> Union[int, Tuple[int, ...]]:
        r"""
        Get the blocked degrees of freedom at the point.

        Returns:
            - `Union[int, Tuple[int, ...]]`: The blocked degree(s) of freedom.
        """
        return self._blocked_dof
    
    @blocked_dof.setter
    def blocked_dof(self, value: Union[int, Tuple[int, ...]]):
        """
        Set the blocked degrees of freedom at the point.

        Args:
            value (int or tuple of int): Degree(s) of freedom to block.
                                        Use 1 for x-direction, 2 for y-direction.
                                        Accepts a single int or a tuple with up to two values.
        """
        allowed_dofs = (1, 2)

        if isinstance(value, int):
            value = (value,)
        elif isinstance(value, tuple):
            if not all(isinstance(dof, int) for dof in value):
                raise TypeError("All elements in the tuple must be integers.")
        else:
            raise TypeError("Blocked dof must be an int or a tuple of ints.")

        if not set(value).issubset(allowed_dofs):
            raise ValueError("Allowed values are 1 (x-direction) and 2 (y-direction) only.")

        if len(value) > 2:
            raise ValueError("Blocked dof tuple can contain at most two elements.")

        self._blocked_dof = tuple(sorted(set(value)))
        
    # def apply(self, system):
    #     """
    #     Apply the point displacement boundary condition to the system.

    #     Args:
    #         system: The system (e.g., mesh, solver) to which the BC is applied.
    #     """
    #     # Example: system should have a method to set displacement at a point
    #     system.set_point_displacement(self.point, self.displacement)

    def __repr__(self):
        return f"PointDisplacementBC(point={self.location}, blocked_dofs={self.blocked_dof})"
    
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
        




