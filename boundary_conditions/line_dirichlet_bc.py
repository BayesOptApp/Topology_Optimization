import numpy as np
from typing import Tuple, List
from boundary_conditions.point_dirichlet_bc import PointDirichletBC
from meshers.MeshGrid2D import MeshGrid2D

class LineDirichletBC(PointDirichletBC):
    """
    Represents a Dirichlet boundary condition applied along a line segment.
    """

    def __init__(self, 
                 start_point:Tuple[float,float], 
                 end_point:Tuple[float,float], 
                 blocked_dof:Tuple[int, int] = (1, 2))->None:
        """
        Initialize the line Dirichlet boundary condition.

        Args:
            start_point (tuple): Coordinates (x, y) of the start of the line.
            end_point (tuple): Coordinates (x, y) of the end of the line.
            value (float or list): Value(s) to enforce along the line.
            direction (int or list, optional): Direction(s) of the constraint (e.g., 1 for x, 2 for y).
        """

        if not isinstance(start_point, tuple) or len(start_point) != 2:
            raise ValueError("Start point must be a tuple of two numeric values (x, y).")
        if not isinstance(end_point, tuple) or len(end_point) != 2:
            raise ValueError("End point must be a tuple of two numeric values (x, y).")
        
        self._start_point = start_point
        self._end_point = end_point
    
        super().__init__(location=[start_point, end_point], blocked_dof=blocked_dof)
    
    @property
    def start_point(self) -> Tuple[float, float]:
        """
        Get the start point of the line segment.

        Returns:
            tuple: Coordinates (x, y) of the start point.
        """
        return self._start_point
    
    @start_point.setter
    def start_point(self, value: Tuple[float, float]) -> None:
        """
        Set the start point of the line segment.

        Args:
            value (tuple): Coordinates (x, y) of the start point.
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("Start point must be a tuple of two numeric values (x, y).")
        self._start_point = value

        # Update the location to reflect the new start point
        self.location = (self._start_point, self._end_point)
    
    @property
    def end_point(self) -> Tuple[float, float]:
        """
        Get the end point of the line segment.

        Returns:
            tuple: Coordinates (x, y) of the end point.
        """
        return self._end_point
    
    @end_point.setter
    def end_point(self, value: Tuple[float, float]) -> None:
        """
        Set the end point of the line segment.

        Args:
            value (tuple): Coordinates (x, y) of the end point.
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("End point must be a tuple of two numeric values (x, y).")
        self._end_point = value

        # Update the location to reflect the new start point
        self.location = (self._start_point, self._end_point)
        

    def is_on_line(self, point, tol=1e-8):
        """
        Check if a given point lies on the boundary line (within a tolerance).

        Args:
            point (tuple): Coordinates (x, y) of the point to check.
            tol (float): Tolerance for point-line distance.

        Returns:
            bool: True if the point is on the line, False otherwise.
        """
        p1 = np.array(self.start_point)
        p2 = np.array(self.end_point)
        p = np.array(point)
        line_vec = p2 - p1
        point_vec = p - p1
        cross = np.cross(line_vec, point_vec)
        dist = np.linalg.norm(cross) / (np.linalg.norm(line_vec) + tol)
        # Check if projection falls within the segment
        dot = np.dot(point_vec, line_vec)
        if dist < tol and 0 <= dot <= np.dot(line_vec, line_vec):
            return True
        return False
    
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

        affected_nodes = []

        for ii, point in enumerate(meshgrid.coordinate_grid):
            point_normalized = (point[1] / meshgrid.nelx, point[2] / meshgrid.nely)
            if self.is_on_line(point=point_normalized):
                affected_nodes.append(ii)

        # Store the affected nodes in the instance variable
        self.affected_nodes_list = affected_nodes.copy()
        return affected_nodes

    def apply(self, nodes):
        """
        Apply the boundary condition to a list of nodes.

        Args:
            nodes (list): List of node coordinates [(x1, y1), (x2, y2), ...].

        Returns:
            list: Indices of nodes where the BC is applied.
        """
        indices = []
        for i, node in enumerate(nodes):
            if self.is_on_line(node):
                indices.append(i)
        return indices