
from Design_Examples.Raw_Design.Design import Design
from Design_Examples.Raw_Design.Design_LP import Design_LP
from Design_Examples.IOH_Wrappers.IOH_Wrapper import Design_IOH_Wrapper
from Design_Examples.IOH_Wrappers.IOH_Wrapper_LP import Design_LP_IOH_Wrapper

from typing import Union, Dict, Optional


def return_material_properties(material_definition: str,)->dict:
    """
    Return the material properties based on the provided name.
    
    Args:
        name (`str`): The name of the case.
        
    Returns:
        `dict`: The material properties.
    """
    
     # Set the material definition dictionary based on the selected material definition
    if  material_definition == "orthotropic":
        material_definition_dict = {
            "E11": 25,  # Young's modulus in x-direction
            "E22": 1,   # Young's modulus in y-direction
            "nu12": 0.25,  # Poisson's ratio
            "G12": 0.5,  # Shear modulus
        }
    
    elif material_definition == "quasi-isotropic":
        material_definition_dict = {
            "E11": 13,  # Young's modulus in x-direction
            "E22": 13,  # Young's modulus in y-direction
            "nu12": 0.25,  # Poisson's ratio
            "G12": 13/((1+0.25)),   # Shear modulus
        }
    
    else:
        material_definition_dict = {
            "E11": 13,  # Young's modulus in x-direction
            "E22": 13,  # Young's modulus in y-direction
            "nu12": 0.25,  # Poisson's ratio
            "G12": 13/(2*(1+0.25)),   # Shear modulus
        }
    
    return material_definition_dict


def set_case_per_name(name: str, 
                      material_definition:str ) -> Union[Design, Design_LP]:
    """
    Set the case name based on the provided name.
    
    Args:
        name (`str`): The name of the case.
        dim (`int`): The dimension of the problem.
        
    Returns:
        str: The case name.
    """

    # Get the material properties based on the provided material definition
    material_properties_dict = return_material_properties(material_definition=material_definition)

    
    if "Topology_Optimization_With_Lamination_Parameters" in name.strip():

        # Instantiate the Design_LP class
        design_lp = Design_LP(nmmcsx=3,
                              nmmcsy=2,
                              nelx=100,
                              nely=50,
                              mode="TO+LP",
                              symmetry_condition=True,
                              continuity_check_mode="discrete",
                              scalation_mode="unitary",
                              material_properties_dict=material_properties_dict)
        
        return design_lp
        
    elif "Topology_Optimization" in name.strip():
        # Instantiate the Design_LP class
        design_lp = Design_LP(nmmcsx=3,
                              nmmcsy=2,
                              nelx=100,
                              nely=50,
                              mode="TO",
                              symmetry_condition=True,
                              continuity_check_mode="discrete",
                              scalation_mode="unitary",
                              material_properties_dict=material_properties_dict)
        
        return design_lp
    elif "Lamination_Parameters_Optimization" in name.strip():
        # Instantiate the Design_LP class
        design_lp = Design_LP(nmmcsx=3,
                              nmmcsy=2,
                              nelx=100,
                              nely=50,
                              mode="LP",
                              symmetry_condition=True,
                              continuity_check_mode="discrete",
                              scalation_mode="unitary",
                              material_properties_dict=material_properties_dict)
        return design_lp
    elif "Topology_Optimization_MMC" in name.strip():
        design_st = Design(nmmcsx=3,
                              nmmcsy=2,
                              nelx=100,
                              nely=50,
                              symmetry_condition=True,
                              continuity_check_mode="discrete",
                              scalation_mode="unitary",
                              material_properties_dict=material_properties_dict)
        return design_st
    else:
        raise ValueError(f"Unknown case for {name}")

def get_list_of_strings_of_variables(dim:int)->list:
    """
    Get a list of strings of the variables based on the dimension.
    
    Args:
        dim (`int`): The dimension of the problem.
        
    Returns:
        `list`: A list of strings of the variables.
    """
    
    initial_list = []

    for i in range(dim):
        initial_list.append(f"x{i}")
    

    return initial_list