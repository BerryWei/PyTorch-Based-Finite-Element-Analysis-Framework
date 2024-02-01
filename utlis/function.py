import numpy as np
from scipy.interpolate import Rbf
from tqdm import tqdm
from multiprocessing import Pool
import torch
from typing import Union
from torch import Tensor

def interpolate_to_nodes(input_coor, input_attribute, target_coor):
    """
    Interpolate values from Gauss points to nodes using Radial Basis Function (RBF) interpolation.
    This function is dimensionality-agnostic and can handle both 2D and 3D cases.
    
    :param input_coor: A 2D array of Gauss points coordinates.
    :param input_attribute: A 2D array of attributes (strains or stresses) at the Gauss points.
    :param target_coor: A 2D array of node coordinates where the values are to be interpolated.
    :return: A 2D array of interpolated attribute values at the node coordinates.
    """
    flattened_input_coor = input_coor.reshape(-1, input_coor.shape[-1])
    flattened_input_attribute = input_attribute.reshape(-1, input_attribute.shape[-1])
    # Determine if we are working with 2D or 3D
    dimensionality = input_coor.shape[-1]
    

    # Initialize node attributes
    num_nodes = target_coor.shape[0]
    num_attr_components = flattened_input_attribute.shape[1]
    node_attributes = np.zeros((num_nodes, num_attr_components))
    
    if dimensionality == 2:
        for d in range(num_attr_components):
        #for d in tqdm(range(num_attr_components), desc="Interpolating"):
            rbf = Rbf(flattened_input_coor[:, 0], flattened_input_coor[:, 1],
                    flattened_input_attribute[:, d], function='linear', smooth=1e-9)
            node_attributes[:, d] = rbf(target_coor[:, 0], target_coor[:, 1])
    elif dimensionality == 3:
        for d in range(num_attr_components):
        #for d in tqdm(range(num_attr_components), desc="Interpolating"):
            rbf = Rbf(flattened_input_coor[:, 0], flattened_input_coor[:, 1], flattened_input_coor[:, 2],
                    flattened_input_attribute[:, d], function='multiquadric', smooth=1e-9)
            node_attributes[:, d] = rbf(target_coor[:, 0], target_coor[:, 1], target_coor[:, 2])
    else:
        raise ValueError("The input coordinate dimension is neither 2D nor 3D.")

    return node_attributes


def interpolate_to_nodes_for_single_element(input_coor, input_attribute, target_coor):
    """
    Interpolate values from Gauss points to nodes using Radial Basis Function (RBF) interpolation.
    This function is dimensionality-agnostic and can handle both 2D and 3D cases.
    
    :param input_coor: A 2D array of Gauss points coordinates.
    :param input_attribute: A 2D array of attributes (strains or stresses) at the Gauss points.
    :param target_coor: A 2D array of node coordinates where the values are to be interpolated.
    :return: A 2D array of interpolated attribute values at the node coordinates.
    """
    flattened_input_coor = input_coor.reshape(-1, input_coor.shape[-1])
    flattened_input_attribute = input_attribute.reshape(-1, input_attribute.shape[-1])
    # Determine if we are working with 2D or 3D
    dimensionality = input_coor.shape[-1]
    

    # Initialize node attributes
    num_nodes = target_coor.shape[0]
    num_attr_components = flattened_input_attribute.shape[1]
    node_attributes = np.zeros((num_nodes, num_attr_components))
    
    if dimensionality == 2:
        for d in range(num_attr_components):
            _average = np.average(flattened_input_attribute[:, d])
            node_attributes[:, d] = _average
    elif dimensionality == 3:
        for d in range(num_attr_components):
            
            _average = np.average(flattened_input_attribute[:, d])
            node_attributes[:, d] = _average
            
    else:
        raise ValueError("The input coordinate dimension is neither 2D nor 3D.")

    return node_attributes


def write_to_vtk_manual(node_coords, cell_array, cell_types, point_data, filename):
    
    with open(filename, 'w') as file:
        # 寫入標頭
        file.write("# vtk DataFile Version 2.0\n")
        file.write("Generated Volume Mesh\n")
        file.write("ASCII\n")
        file.write("DATASET UNSTRUCTURED_GRID\n")

        # 寫入點座標
        file.write(f"POINTS {len(node_coords)} float\n")
        for coord in node_coords:
            file.write(f"{' '.join(map(str, coord))}\n")

        # 預備 CELLS 信息
        cells_flat = []
        for sublist in cell_array:
            cells_flat.append(len(sublist))  # Append the size of the cell
            cells_flat.extend(sublist)       # Extend the flat list with the cell's content
        cells_size = len(cells_flat)

        # 寫入單元格信息
        file.write(f"CELLS {len(cell_array)} {cells_size}\n")
        for cell in cell_array:
            file.write(f"{len(cell)} {' '.join(map(str, cell))}\n")

        # 寫入 CELL_TYPES 信息
        file.write(f"CELL_TYPES {len(cell_types)}\n")
        for cell_type in cell_types:
            file.write(f"{cell_type}\n")

        # 寫入 POINT_DATA 信息
        num_points = len(node_coords)
        file.write(f"POINT_DATA {num_points}\n")
        for name, data in point_data.items():
            file.write(f"SCALARS {name} float\n")
            file.write("LOOKUP_TABLE default\n")
            for value in data:
                file.write(f"{value}\n")


def get_vtk_cell_type(num_nodes_per_element, num_dimensions):
    """
    Map the number of nodes per element and number of dimensions to VTK cell type.
    Note: This mapping is not exhaustive and will need to be expanded based on the specific elements used.
    
    Parameters:
    - num_nodes_per_element: int, number of nodes per element
    - num_dimensions: int, number of spatial dimensions
    
    Returns:
    - vtk_cell_type: int, VTK cell type
    """
    if num_dimensions == 2:
        # For 2D elements
        if num_nodes_per_element == 3:
            return 5  # VTK_TRIANGLE
        elif num_nodes_per_element == 4:
            return 9  # VTK_QUAD
        elif num_nodes_per_element == 8:
            return 23  # VTK_QUADRATIC_QUAD
        # Add more cases for different types of 2D elements
    elif num_dimensions == 3:
        # For 3D elements
        if num_nodes_per_element == 4:
            return 10  # VTK_TETRA
        elif num_nodes_per_element == 8:
            return 12  # VTK_HEXAHEDRON
        # Add more cases for different types of 3D elements

    raise ValueError("Unsupported element configuration")


def add_zero_z_coordinate(node_coords):
    """
    For a 2D array of node coordinates (with shape [n_nodes, 2]),
    add a zero z-coordinate to make it compatible with VTK 3D format.

    Parameters:
    - node_coords: numpy.ndarray, 2D array of node coordinates

    Returns:
    - new_node_coords: numpy.ndarray, 3D array of node coordinates with added zero z-coordinate
    """
    n_nodes = node_coords.shape[0]
    new_node_coords = np.zeros((n_nodes, 3))
    new_node_coords[:, :2] = node_coords
    # The z-coordinate is assumed to be zero and hence not modified (it remains zero from the initialization)
    return new_node_coords




def Vec2Mat_Vogit(stressVector, dtype=torch.float64, device='cpu'):
    """
    Convert a stress vector to a stress matrix in Voigt notation.

    Parameters:
    stressVector (Tensor): A stress vector (3 elements for 2D, 6 elements for 3D).
    dtype (torch.dtype): Data type of the output tensor.
    device (string): The device on which the tensor will be allocated.

    Returns:
    Tensor: A 2D or 3D stress matrix.
    """
    dim = len(stressVector)
    
    if dim == 3:
        output = torch.zeros((2, 2), dtype=dtype, device=device)
        output[0, 0] = stressVector[0]
        output[1, 1] = stressVector[1]
        output[0, 1] = output[1, 0] = stressVector[2]
    elif dim == 6:
        output = torch.zeros((3, 3), dtype=dtype, device=device)
        output[0, 0], output[1, 1], output[2, 2] = stressVector[0:3]
        output[0, 1] = output[1, 0] = stressVector[5]
        output[1, 2] = output[2, 1] = stressVector[3]
        output[0, 2] = output[2, 0] = stressVector[4]
        
    else:
        raise ValueError("Stress vector must have 3 (2D) or 6 (3D) elements.")
    
    return output

def Mat2Vec_Vogit(stressMatrix: torch.Tensor, dtype: torch.dtype = torch.float64, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Convert a stress matrix to a stress vector in Voigt notation.

    Parameters:
    stressMatrix (torch.Tensor): A stress matrix (2x2 for 2D, 3x3 for 3D).
    dtype (torch.dtype): Data type of the output tensor.
    device (Union[str, torch.device]): The device on which the tensor will be allocated.

    Returns:
    torch.Tensor: A stress vector (3 elements for 2D, 6 elements for 3D).
    """
    rows, cols = stressMatrix.shape

    if rows == cols == 2:  # 2D case
        vector_length = 3
        output = torch.zeros(vector_length, dtype=dtype, device=device)
        output[0] = stressMatrix[0, 0]
        output[1] = stressMatrix[1, 1]
        output[2] = stressMatrix[0, 1]  # Assuming a symmetric matrix, so 0,1 and 1,0 are the same
    elif rows == cols == 3:  # 3D case
        vector_length = 6
        output = torch.zeros(vector_length, dtype=dtype, device=device)
        output[0], output[1], output[2] = stressMatrix[0, 0], stressMatrix[1, 1], stressMatrix[2, 2] 
        output[5], output[3], output[4] = stressMatrix[0, 1], stressMatrix[1, 2], stressMatrix[0, 2]  
    else:
        raise ValueError("Stress matrix must be 2x2 (2D) or 3x3 (3D).")

    return output


def sym3333_to_m66(m3333:Tensor, notationMap:Tensor, symmetrise:bool) -> Tensor:
	"""
	Reduce symmetric 3x3x3x3 tensor to 6x6 tensor following `notationMap(default=IMP)` notation.

	Args:
		m3333: The 3x3x3x3 tensor. m3333.shape == (3, 3, 3, 3).
		notationMap: The tensor notation map.

	Returns:
		The 6x6 tensor.
	"""
	m66 = m3333.new_zeros((6,6))
	for i in range(6):
		for j in range(6):
			m66[i,j] = m3333[notationMap[0,i], notationMap[1,i], notationMap[0,j], notationMap[1,j]]

	if symmetrise:
		m66 = (m66 + m66.t()) / 2
		
	return m66
