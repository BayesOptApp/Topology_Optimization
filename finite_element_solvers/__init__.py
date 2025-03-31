from .four_point_quadrature_plane_stress import Mesh
from .four_point_quadrature_plane_stress_composite import CompositeMaterialMesh


def select_2D_quadrature_solver(mode: str,
                                length:float,
                                height:float,
                                element_length:float,
                                element_height:float,
                                sparse_matrices:bool=True) -> Mesh:
    r"""
    Selects the appropriate solver based on the provided parameters.

    Parameters:
    mode (str): The mode of operation, either "TO" or "TO+LP".
    length (float): The length of the mesh.
    height (float): The height of the mesh.
    element_length (float): The length of each element in the mesh.
    element_height (float): The height of each element in the mesh.
    sparse_matrices (bool): Flag to use sparse matrices or not: Default is True.

    Returns:
    Mesh: An instance of either Mesh or MeshComposite based on the input parameters.
    """
    
    if mode == "TO":
        return Mesh(length=length,
                    height=height,
                    element_length=element_length,
                    element_height=element_height,
                    sparse_matrices=sparse_matrices)
    
    elif mode == "TO+LP":
        return CompositeMaterialMesh(length=length,
                    height=height,
                    element_length=element_length,
                    element_height=element_height,
                    sparse_matrices=sparse_matrices)