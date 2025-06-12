from boundary_conditions.abstract_bc import AbstractBoundaryCondition

class BoundaryConditionList(list):
    """
    A list to hold boundary condition objects.
    Extends the built-in list.
    """

    def add(self, boundary_condition:AbstractBoundaryCondition):
        """Add a boundary condition to the list."""
        if not isinstance(boundary_condition, AbstractBoundaryCondition) or not hasattr(boundary_condition, 'location'):
            raise TypeError("boundary_condition must be an instance of AbstractBoundaryCondition.")
        self.append(boundary_condition)

    def remove(self, boundary_condition:AbstractBoundaryCondition):
        """Remove a boundary condition from the list."""
        if not isinstance(boundary_condition, AbstractBoundaryCondition) or not hasattr(boundary_condition, 'location'):
            raise TypeError("boundary_condition must be an instance of AbstractBoundaryCondition.")
        return super().remove(boundary_condition)
    
    def insert(self, index, object):
        """Insert a boundary condition at a specific index."""
        if not isinstance(object, AbstractBoundaryCondition) or not hasattr(object, 'location'):
            raise TypeError("object must be an instance of AbstractBoundaryCondition.")
        return super().insert(index, object)
    
    def append(self, object):
        """Append a boundary condition to the end of the list."""
        if not isinstance(object, AbstractBoundaryCondition) or not hasattr(object, 'location'):
            raise TypeError("object must be an instance of AbstractBoundaryCondition.")
        return super().append(object)
    
    def __getitem__(self, index):
        """Get a boundary condition by index."""
        if isinstance(index, slice):
            return BoundaryConditionList(super().__getitem__(index))
        elif isinstance(index, int):
            return super().__getitem__(index)
        else:
            raise TypeError("Index must be an integer or a slice.")

    def get_all(self):
        """Return all boundary conditions."""
        return list(self)