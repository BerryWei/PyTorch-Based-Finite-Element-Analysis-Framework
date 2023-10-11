import numpy as np 
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import yaml
from .element import *
from typing import Union, Dict, Any, Type, List, Callable
from .material import * 
from tqdm import tqdm
from pathlib import Path


class FiniteElementModel:
    def __init__(self, device: str='cuda') -> None:
        """
        Initialize the finite element model.
        
        Parameters:
        - device (str): Device for torch tensors (default: 'cuda').
        """
        # Fundamental parameters
        self.parameters = {
            'num_dimensions': None
        }
        # Mesh attributes
        self.node_coords = None
        self.element_node_indices = None
        self.num_dofs = None
        self.num_element = None
        
        # Stiffness and Load attributes
        self.global_stiffness = None
        self.global_load = None

        # Boundary conditions
        self.node_dof_disp = None
        self.elem_face_trac = None
        self.node_dof_forces = None
        
        # Derived attributes
        self.num_dofs = None
        

        # Other attributes (can be initialized later)
        self.element_stiffnesses = None


    def compute_element_stiffness(self, material: Material, ElementClass: Type[BaseElement]) -> None:
        
        dof_per_element = ElementClass.element_dof
        self.element_stiffnesses = torch.zeros(self.num_element, dof_per_element, dof_per_element, device=self.device)

        for i in range(self.num_element):
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes]
            area = 0.5 * abs(node_coords[0, 0] * (node_coords[1, 1] - node_coords[2, 1]) +
                            node_coords[1, 0] * (node_coords[2, 1] - node_coords[0, 1]) +
                            node_coords[2, 0] * (node_coords[0, 1] - node_coords[1, 1]))
            
            B_matrix = ElementClass.compute_B_matrix(node_coords, device=self.device)
            D_matrix = material.consistent_tangent(element_index=i)
            K = area * torch.einsum('ji,jk,kl->il', B_matrix, D_matrix, B_matrix)
            self.element_stiffnesses[i] = K

    def compute_element_stiffness_vectorize(self, material: Material, ElementClass: Type[BaseElement]) -> None:
            
        dof_per_element = ElementClass.element_dof
        self.element_stiffnesses = torch.zeros(self.num_element, dof_per_element, dof_per_element, device=self.device)

        # Get all node coordinates for all elements at once
        all_node_coords = self.node_coords[self.element_node_indices]

        # Compute areas for all elements
        areas = 0.5 * torch.abs(
            all_node_coords[:, 0, 0] * (all_node_coords[:, 1, 1] - all_node_coords[:, 2, 1]) +
            all_node_coords[:, 1, 0] * (all_node_coords[:, 2, 1] - all_node_coords[:, 0, 1]) +
            all_node_coords[:, 2, 0] * (all_node_coords[:, 0, 1] - all_node_coords[:, 1, 1])
        )

        # Compute B_matrix for all elements
        B_matrices = ElementClass.compute_B_matrix_vectorized(all_node_coords, device=self.device) # (num_element, 3, dof_per_element)

        # Compute D_matrix for all elements
        D_matrices = material.consistent_tangent_vectorized(num_elements=self.num_element) # (num_element, 3, 3)

        # Compute K for all elements
        K_matrices = areas[:, None, None] * torch.einsum('bji,bjk,bkl->bil', B_matrices, D_matrices, B_matrices)

        self.element_stiffnesses = K_matrices




    def assemble_global_stiffness(self):
        """
        Assemble the global stiffness matrix from the element stiffness matrices.
        """               
        # Initialize the global stiffness matrix to zero
        self.global_stiffness = torch.zeros(self.num_dofs, self.num_dofs, device=self.device)
        
        # Determine the global degree of freedom indices for all elements
        # Calculate the DOF indices for each dimension of each node in the elements
        dof_per_dimension = torch.arange(self.parameters['num_dimensions']).to(self.device).unsqueeze(0)  # [0,1], or[0,1,2]
        elemental_dof_indices = self.element_node_indices.unsqueeze(-1) * self.parameters['num_dimensions'] + dof_per_dimension  # Broadcasting


        # Use advanced indexing to assemble all element stiffness matrices into the global stiffness matrix simultaneously
        n_dim = self.parameters['num_dimensions']
        for i, elem_nodes in enumerate(self.element_node_indices):
            # Determine the global degree of freedom indices for this element
            elem_dof_indices = torch.cat([(n_dim*elem_nodes).unsqueeze(-1), (n_dim*elem_nodes+1).unsqueeze(-1)], dim=-1).view(-1).long()
            
            # Assemble the element stiffness matrix into the global stiffness matrix
            self.global_stiffness[elem_dof_indices[:, None], elem_dof_indices[None, :]] += self.element_stiffnesses[i]

        #self.global_stiffness[global_dof_indices[:, :, None], global_dof_indices[:, None, :]] += self.element_stiffnesses


    def read_geom_from_yaml(self, file_path: Path) -> None:
        """
        Read data from a YAML file and populate the model's attributes.

        Parameters:
        - file_path (Path): Path to the input YAML file.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Update the fundamental parameters
        self.parameters['num_dimensions'] = data['PARAMETER']['num-dim']
        self.element_type = data['Element']['type']

        # Update the mesh attributes
        self.num_element = data['Element']['num-elem']
        self.num_node = data['NODE']['num-node']
        self.node_coords = torch.tensor(data["NODE"]['nodal-coord'])
        self.element_node_indices = torch.tensor(data["Element"]['elem-conn'], dtype=torch.long)
        
        # Update the derived attributes
        self.num_dofs = self.node_coords.shape[0] * self.parameters['num_dimensions']

    def assemble_global_load_vector(self, ElementClass: Type[BaseElement]) -> None:
        """
        Assemble the global load vector R from the prescribed displacements and tractions.
        """
        # Initialize the global load vector to zero
        self.global_load = torch.zeros(self.num_dofs, device=self.device)

        if self.node_dof_disp.numel() > 0:
            # Incorporate prescribed displacements into the global load vector using vectorized operations
            node_indices = self.node_dof_disp[:, 0].long()
            dof_indices = self.node_dof_disp[:, 1].long()
            values = self.node_dof_disp[:, 2]

            global_dof_indices = node_indices * self.parameters['num_dimensions'] + dof_indices
            self.global_load.index_add_(dim=0, index=global_dof_indices, source=values)

        if self.node_dof_forces.numel() > 0:
            # Incorporate nodal forces into the global load vector using vectorized operations
            force_node_indices = self.node_dof_forces[:, 0].long()
            force_dof_indices = self.node_dof_forces[:, 1].long()
            force_values = self.node_dof_forces[:, 2]
            force_global_dof_indices = force_node_indices * self.parameters['num_dimensions'] + force_dof_indices
            self.global_load.index_add_(0, force_global_dof_indices, force_values)


        # Incorporate tractions into the global load vector
        # NOTE: TO DO


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
        self.node_dof_forces = torch.tensor(data["Boundary"].get("node_dof_forces", []))


    # To move data to GPU
    def to_device(self, device: Union[str, torch.device]) -> None:
        """
        Move the model data to a specified device.

        Parameters:
        - device (Union[str, torch.device]): The target device, e.g., "cuda" or "cpu".
        """
        self.device = device
        self.node_coords = self.node_coords.to(device)
        self.element_node_indices = self.element_node_indices.to(device)
        if hasattr(self, 'material'):
            self.material.to_device(device)
        if hasattr(self, 'node_dof_disp'):
            self.node_dof_disp = self.node_dof_disp.to(device)
        if hasattr(self, 'elem_face_trac'):
            self.elem_face_trac = self.elem_face_trac.to(device)
        if hasattr(self, 'node_dof_forces'):
            self.node_dof_forces = self.node_dof_forces.to(device)


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
    
    def solve_system(self):
        """
        Solve the global system of equations to get the nodal displacements.
        """

        # Identify known DOFs from self.node_dof_disp
        if self.node_dof_disp.numel() > 0:
            known_dof_indices = (self.node_dof_disp[:, 0] * self.parameters['num_dimensions'] + self.node_dof_disp[:, 1]).long()
        else:
            known_dof_indices = torch.tensor([], dtype=torch.long, device=self.device)
        # Identify unknown DOFs
        all_dofs = torch.arange(self.num_dofs, device=self.device)
        unknown_dof_indices = torch.tensor([idx for idx in all_dofs if idx not in known_dof_indices], device=self.device)

        # Extract submatrix and subvector
        K_sub = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        R_sub = self.global_load[unknown_dof_indices]

        # Solve the system
        u_sub = torch.linalg.solve(K_sub, R_sub.unsqueeze(1))
        residual = R_sub - K_sub @ u_sub.squeeze()
        relative_residual_error = torch.norm(residual) / torch.norm(R_sub)
        print(f"Relative Residual Error: {relative_residual_error.item():.8f}")



        # Create a global displacement vector
        self.global_displacements = torch.zeros(self.num_dofs, device=self.device)
        self.global_displacements[unknown_dof_indices] = u_sub.squeeze()

        # Fill known displacements, if any
        if self.node_dof_disp.numel() > 0:
            self.global_displacements[known_dof_indices] = self.node_dof_disp[:, 2]

        print('test')





    def compute_elemental_strains_stresses(self, ElementClass: Type[BaseElement]) -> None:
            """Compute strains and stresses for each element."""
            if self.parameters['num_dimensions'] == 2:
                dim = 3
            elif self.parameters['num_dimensions'] == 3:
                dim = 6
            else:
                raise ValueError(f"self.parameters['num_dimensions'] = {self.parameters['num_dimensions']} is not correct.")
            self.elemental_strains = torch.zeros(self.num_element, dim, device=self.device)  # Assuming plane strain condition
            self.elemental_stresses = torch.zeros(self.num_element, dim, device=self.device)  # Assuming plane strain condition

            for i, elem_nodes in enumerate(self.element_node_indices):
                node_coords = self.node_coords[elem_nodes]
                B_matrix = ElementClass.compute_B_matrix(node_coords, device=self.device)

                # elem_nodes = [n1, n2, n3]--> elem_dof_indices = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1]
                elem_dof_indices = torch.cat([(self.parameters['num_dimensions']*elem_nodes).unsqueeze(-1), (self.parameters['num_dimensions']*elem_nodes+1).unsqueeze(-1)], dim=-1).view(-1).long()
                elem_displacements = self.global_displacements[elem_dof_indices]
                strains = B_matrix @ elem_displacements
                stresses = self.material.consistent_tangent(element_index = i) @ strains
                
                self.elemental_strains[i] = strains
                self.elemental_stresses[i] = stresses

    def save_results_to_file(self, file_path: Path) -> None:
        """Save elemental strains and stresses to a file."""
        with open(file_path, 'w') as file:
            file.write("*NODE\n")
            if self.parameters['num_dimensions'] == 2:
                file.write("node#-u1-u2:\n")
                for i, node_coord in enumerate(self.node_coords):
                    disp = self.global_displacements[i*2:i*2+2]
                    file.write(f"{i} {disp[0]:.12f} {disp[1]:.12f}\n")
                
                file.write("*ELEMENT\n")
                file.write("elem#-e11-e22-e12-s11-s22-s12\n")
                for i, (strain, stress) in enumerate(zip(self.elemental_strains, self.elemental_stresses)):
                    file.write(f"{i} {strain[0]:.12f} {strain[1]:.12f} {strain[2]:.12f} {stress[0]:.12f} {stress[1]:.12f} {stress[2]:.12f}\n")
            
            elif self.parameters['num_dimensions'] == 3:
                file.write("node#-u1-u2-u3:\n")
                for i, node_coord in enumerate(self.node_coords):
                    disp = self.global_displacements[i*3:i*3+3]
                    file.write(f"{i} {disp[0]:.12f} {disp[1]:.12f} {disp[2]:.12f}\n")
                
                file.write("*ELEMENT\n")
                file.write("elem#-e11-e22-e33-e23-e13-e12-s11-s22-s33-s23-s13-s12\n")
                for i, (strain, stress) in enumerate(zip(self.elemental_strains, self.elemental_stresses)):
                    file.write(f"{i} {' '.join([f'{val:.12f}' for val in strain])} {' '.join([f'{val:.12f}' for val in stress])}\n")
            
            else:
                raise ValueError(f"Unsupported number of dimensions: {self.parameters['num_dimensions']}")

    def plot(self, funcs: List[Callable[[], torch.Tensor]]) -> None:
        """
        Plot the distribution of specified attributes in the domain.

        Parameters:
        - funcs (List[Callable[[], torch.Tensor]]): A list of functions that return attributes for plotting.
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        x = self.node_coords[:, 0].cpu().numpy()
        y = self.node_coords[:, 1].cpu().numpy()
        triangles = self.element_node_indices.cpu().numpy()

        for func in funcs:
            values = func().cpu().numpy()

            fig, ax = plt.subplots()
            tripcolor_plot = ax.tripcolor(x, y, triangles, facecolors=values, shading='flat', edgecolors='k')
            plt.colorbar(tripcolor_plot)
            ax.set_aspect('equal', 'box')
            ax.set_title(func.__name__)
            plt.show()
    @staticmethod
    def node_indices_in_x_range(node_coords, x_range):
        lower_bound, upper_bound = x_range
        selected_indices = [idx for idx, coord in enumerate(node_coords) if lower_bound <= coord[0] <= upper_bound]
        return selected_indices
    
    @staticmethod
    def element_indices_in_x_range(element_node_indices, node_coords, x_range):
        lower_bound, upper_bound = x_range

        # Check if any node of an element is in the given x_range
        def is_element_in_range(element):
            for node_index in element:
                # Ensure that node_index is valid before using it
                if node_index is not None:
                    
                    if lower_bound <= node_coords[node_index][0] <= upper_bound:
                        return True
            return False

        selected_element_indices = [idx for idx, element in enumerate(element_node_indices) if is_element_in_range(element)]
        return selected_element_indices
