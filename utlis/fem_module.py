import numpy as np 
import torch
import yaml
from .element import *
from typing import Union, Dict, Any
from .material import * 
from pathlib import Path

class FiniteElementModel:
    def __init__(self) -> None:
        """
        Initialize the Finite Element Model.
        """
        # Tensor to store all node coordinates (shape: num-node x num-dim)
        self.node_coords: torch.Tensor = None  
        
        # Tensor to store node indices for each element (shape: num-node x num-elem-node)
        self.element_node_indices: torch.Tensor = None  
        
        self.parameters: Dict[str, Any] = {}



    def read_geom_from_yaml(self, file_path: Path) -> None:
        """
        Read data from a YAML file and populate the model's attributes.

        Parameters:
        - file_path (Path): Path to the input YAML file.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Populate node coordinates tensor --> self.node_coords.shape = (num-node, num-dim)
        self.node_coords = torch.tensor(data["NODE"]["nodal-coord"])
        
        # Populate element node indices tensor --> self.node_coords.shape = (num-node, num-elem-node)
        self.element_node_indices = torch.tensor(data["Element"]["elem-conn"])
        
        # Populate parameters
        self.parameters["num_dimensions"] = data["PARAMETER"]["num-dim"]


    def read_material_from_yaml(self, file_path: Path):
        """
        Read material properties from a YAML file, calculate the elasticity stiffness matrix, 
        and instantiate the Elasticity class to populate the model's material attribute.

        Parameters:
        - file_path (Path): Path to the material YAML file.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Whitelist of allowed material models
        allowed_models = ["Elasticity_2D", "Elasticity"]

        model_name = data['MODEL']
        if model_name not in allowed_models:
            raise ValueError(f"{model_name} is not an allowed material model.")
        
        MaterialClass = eval(model_name)

        # Calculate the 3D elasticity stiffness matrix using the Material class
        self.material = MaterialClass(**data["MATPROP"])

    def read_loading_from_yaml(self, file_path: Path):
        """
        Read loading data from a YAML file and populate the relevant attributes in the model.

        Parameters:
        - file_path (Path): Path to the loading YAML file.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Extracting boundary conditions and converting to PyTorch tensors
        self.node_dof_disp = torch.tensor(data["Boundary"].get("node_dof_disp", []))
        self.elem_face_trac = torch.tensor(data["Boundary"].get("elem_face_trac", []))


    # To move data to GPU
    def to_device(self, device: Union[str, torch.device]) -> None:
        """
        Move the model data to a specified device.

        Parameters:
        - device (Union[str, torch.device]): The target device, e.g., "cuda" or "cpu".
        """
        self.node_coords = self.node_coords.to(device)
        self.element_node_indices = self.element_node_indices.to(device)
        if hasattr(self, 'material'):
            self.material.to_device(device)
        if hasattr(self, 'node_dof_disp'):
            self.node_dof_disp = self.node_dof_disp.to(device)
        if hasattr(self, 'elem_face_trac'):
            self.elem_face_trac = self.elem_face_trac.to(device)

    def is_data_on_cuda(self) -> bool:
        """
        Check if all data tensors are on CUDA.

        Returns:
        - bool: True if all tensors are on CUDA, False otherwise.
        """
        tensors_on_cuda = [
            self.node_coords.device.type == 'cuda',
            self.element_node_indices.device.type == 'cuda'
        ]
        
        if hasattr(self, 'node_dof_disp'):
            tensors_on_cuda.append(self.node_dof_disp.device.type == 'cuda')
        if hasattr(self, 'elem_face_trac'):
            tensors_on_cuda.append(self.elem_face_trac.device.type == 'cuda')
        
        # If material data is a tensor, check if it's on CUDA
        # NOTE: This assumes that the material attribute, if it's a tensor, has a device attribute.
        # If the material data is more complex, additional checks may be needed.
        if hasattr(self, 'material') and isinstance(self.material, torch.Tensor):
            tensors_on_cuda.append(self.material.device.type == 'cuda')
        
        return all(tensors_on_cuda)