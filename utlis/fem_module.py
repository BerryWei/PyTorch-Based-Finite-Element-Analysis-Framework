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
from .gaussQuadrature import GaussQuadrature
from scipy.linalg import sqrtm
from .function import Vec2Mat_Vogit
from multiprocessing import Pool

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
        self.device = device

    def generate_material_dict(self) -> None:

        n_dim = self.parameters['num_dimensions']
        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, n_dim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()
        self.MaterialClass_args['device'] = self.device
        self.material_dict = {}
        total_iterations = self.num_element * len(gauss_points)
        with tqdm(total=total_iterations, desc='Initializing Materials') as pbar:
            for i in range(self.num_element):
                for j in range(len(gauss_points)):
                    self.material_dict[(i, j)] = eval(self.MaterialClass_name)(**self.MaterialClass_args)
                    pbar.update(1)  # Update the progress bar by one iteration
        

    def init_element_class(self):
        self.elementClass = eval(self.element_type)




    def compute_element_stiffness(self) -> None:
        
        dof_per_element = self.elementClass.element_dof
        n_dim = self.parameters['num_dimensions']
        self.element_stiffnesses = torch.zeros(self.num_element, dof_per_element, dof_per_element, device=self.device)

        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, n_dim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()

        for i in tqdm(range(self.num_element), desc='Computing Stiffness Matrix'):
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes].type(torch.float64)

            K_elem = torch.zeros(dof_per_element, dof_per_element, dtype=torch.float64, device=self.device)
            for j, (gauss_point, weight) in enumerate(zip(gauss_points, weights)):

                dN_dxi = self.elementClass.shape_function_derivatives(gauss_point, device=self.device)
                J = self.elementClass.jacobian(node_coords, dN_dxi, device=self.device)
                detJ = torch.det(J)
                
                B_matrix = self.elementClass.compute_B_matrix(dN_dxi, J, device=self.device)
                D_matrix = self.material_dict[(i, j)].consistent_tangent(F=None, strain=None).type(torch.float64)

                K_elem += weight * detJ * torch.einsum('ji,jk,kl->il', B_matrix, D_matrix, B_matrix)
            self.element_stiffnesses[i] = K_elem




    def assemble_global_stiffness(self):
        """
        Assemble the global stiffness matrix from the element stiffness matrices.
        """               
        self.global_stiffness = torch.zeros(self.num_dofs, self.num_dofs, device=self.device).to(dtype=torch.float64)
        
        dof_per_dimension = torch.arange(self.parameters['num_dimensions']).to(self.device).unsqueeze(0)  # [[0,1]], or[[0,1,2]]
        n_dim = self.parameters['num_dimensions']

        for i in range(self.num_element):
            elem_nodes = self.element_node_indices[i]
            elem_dof_indices = (elem_nodes.unsqueeze(-1) * n_dim + dof_per_dimension).view(-1).long()
            
            # Assemble the element stiffness matrix into the global stiffness matrix
            self.global_stiffness[elem_dof_indices[:, None], elem_dof_indices[None, :]] += self.element_stiffnesses[i]


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
        self.node_coords = torch.tensor(data["NODE"]['nodal-coord'], dtype=torch.float64)
        self.element_node_indices = torch.tensor(data["Element"]['elem-conn'], dtype=torch.long)
        
        # Update the derived attributes
        self.num_dofs = self.node_coords.shape[0] * self.parameters['num_dimensions']
    
    def _get_dof_indices(self):
        known_dof_indices = (self.node_dof_disp[:, 0] * self.parameters['num_dimensions'] + self.node_dof_disp[:, 1]).long() if self.node_dof_disp.numel() > 0 else torch.tensor([], dtype=torch.long, device=self.device)
        all_dofs = torch.arange(self.num_dofs, device=self.device)
        unknown_dof_indices = torch.tensor(list(set(all_dofs.tolist()) - set(known_dof_indices.tolist())), device=self.device)
        return known_dof_indices, unknown_dof_indices

    def assemble_global_load_vector(self, ) -> None:
        """
        Assemble the global load vector R from the prescribed displacements and tractions.
        """
        # Initialize the global load vector to zero
        self.global_load = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        known_dof_indices, unknown_dof_indices = self._get_dof_indices()

        # Extract submatrix and subvector
        # K_unknown_unknown = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        K_known_known = self.global_stiffness[known_dof_indices, :][:, known_dof_indices]
        K_unknown_known = self.global_stiffness[unknown_dof_indices, :][:, known_dof_indices]

        if self.node_dof_disp.numel() > 0:
            # Incorporate prescribed displacements into the global load vector using vectorized operations
            values = self.node_dof_disp[:, 2].to(dtype=torch.float64)
            self.global_load.index_add_(dim=0, index=known_dof_indices, source= - K_known_known @ values)
            self.global_load.index_add_(dim=0, index=unknown_dof_indices, source= - K_unknown_known @ values)

        if self.node_dof_forces.numel() > 0:
            # Incorporate nodal forces into the global load vector using vectorized operations
            force_node_indices = self.node_dof_forces[:, 0].long()
            force_dof_indices = self.node_dof_forces[:, 1].long()
            force_values = self.node_dof_forces[:, 2].to(dtype=torch.float64)
            force_global_dof_indices = force_node_indices * self.parameters['num_dimensions'] + force_dof_indices
            self.global_load.index_add_(0, force_global_dof_indices, force_values)



        # Incorporate tractions into the global load vector
        if self.elem_face_trac.numel() > 0:
            for traction_data in self.elem_face_trac:
                elem_idx, face_idx = traction_data[:2]
                traction_values = traction_data[2:]
                elem_idx = elem_idx.long()  

                # Get the element's node indices for the specified face
                face_nodes = self.elementClass.boundary_nodes(face_idx)
                elem_nodes = self.element_node_indices[elem_idx]
                nodes = [elem_nodes[fn] for fn in face_nodes]

                # Get the coordinates of the nodes
                node_coords = self.node_coords[nodes,:]


                # Get Gauss points and weights for the face
                gauss_points, weights = self.elementClass.face_gauss_points_and_weights(face_idx, node_coords)

                for gp, w in zip(gauss_points, weights):
                    for i, node in enumerate(nodes):
                        global_dof_index = node * self.parameters['num_dimensions']
                        for dim in range(self.parameters['num_dimensions']):
                            self.global_load[global_dof_index + dim] += traction_values[dim] * w 

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
        allowed_models = ["Elasticity_2D", "Elasticity_3D", "Hypoelastic", "Mooney", 'neoHookean', "neoHookean_matadi"]

        model_name = data['MODEL']
        if model_name not in allowed_models:
            raise ValueError(f"{model_name} is not an allowed material model.")
        
        self.MaterialClass_name = model_name
        self.MaterialClass_args = data["MATPROP"]
        



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

        # # Identify known DOFs from self.node_dof_disp
        # if self.node_dof_disp.numel() > 0:
        #     known_dof_indices = (self.node_dof_disp[:, 0] * self.parameters['num_dimensions'] + self.node_dof_disp[:, 1]).long()
        # else:
        #     known_dof_indices = torch.tensor([], dtype=torch.long, device=self.device)
        # # Identify unknown DOFs
        # all_dofs = torch.arange(self.num_dofs, device=self.device)
        # unknown_dof_indices = torch.tensor([idx for idx in all_dofs if idx not in known_dof_indices], device=self.device)
        known_dof_indices, unknown_dof_indices = self._get_dof_indices()

        # Extract submatrix and subvector
        K_sub = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        R_sub = self.global_load[unknown_dof_indices]

        # Solve the system
        u_sub = torch.linalg.solve(K_sub, R_sub.unsqueeze(1))


        # residual = R_sub - K_sub @ u_sub.squeeze()
        # relative_residual_error = torch.norm(residual) / torch.norm(R_sub)
        # print(f"Relative Residual Error: {relative_residual_error.item():.8f}")



        # Create a global displacement vector
        self.global_displacements = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        self.global_displacements[unknown_dof_indices] = u_sub.squeeze()

        # Fill known displacements, if any
        if self.node_dof_disp.numel() > 0:
            self.global_displacements[known_dof_indices] = self.node_dof_disp[:, 2].to(dtype=torch.float64)






    def compute_GP_strains_stresses(self) -> None:
        """Compute strains and stresses for each element."""
        if self.parameters['num_dimensions'] == 2:
            dim = 3
        elif self.parameters['num_dimensions'] == 3:
            dim = 6
        else:
            raise ValueError(f"self.parameters['num_dimensions'] = {self.parameters['num_dimensions']} is not correct.")
        self.elemental_strains = torch.zeros(self.num_element, dim, device=self.device)  
        self.elemental_stresses = torch.zeros(self.num_element, dim, device=self.device)  
        ndim = self.parameters['num_dimensions']
        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, ndim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()

        # Additional storage for Gauss point coordinates, strains, and stresses
        self.gauss_point_coordinates = torch.zeros((self.num_element, len(gauss_points), ndim), device=self.device)
        self.gauss_point_strains = torch.zeros((self.num_element, len(gauss_points), dim), device=self.device)  
        self.gauss_point_stresses = torch.zeros((self.num_element, len(gauss_points), dim), device=self.device)  

        for i in tqdm(range(self.num_element), desc='Computing stress/strain'):
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes]
            elem_dof_indices = torch.cat([(elem_nodes*ndim + i).unsqueeze(0) for i in range(ndim)], dim=0)
            elem_dof_indices = elem_dof_indices.t().contiguous().view(-1)

            elem_displacements = self.global_displacements[elem_dof_indices]

            for j, gauss_point in enumerate(gauss_points):
                N = self.elementClass.shape_functions(gauss_point, device=self.device)
                physical_coords = torch.mm(N.unsqueeze(0), node_coords).squeeze(0)
                self.gauss_point_coordinates[i, j] = physical_coords

                # Compute strains and stresses at each Gauss point
                dN_dxi = self.elementClass.shape_function_derivatives(gauss_point, device=self.device)
                dx_dxi = J = self.elementClass.jacobian(node_coords, dN_dxi, device=self.device)
                B_matrix = self.elementClass.compute_B_matrix(dN_dxi, J, device=self.device)

                # material law : eval stain @ gp (Vogit notation)
                strainVec = B_matrix @ elem_displacements

                strain = Vec2Mat_Vogit(strainVec)
                dN_dx = torch.einsum('im, mj -> ij',dN_dxi, dx_dxi.inverse())
                F = self.computeF(dN_dx, elem_displacements) # F.shape = (3,3)

                stressVector = self.material_dict[(i, j)].update_states(F=F, strain=strain).type(torch.float64)
                
                self.gauss_point_strains[i, j] = strainVec  # Store strains at Gauss points
                self.gauss_point_stresses[i, j] = stressVector.squeeze()  # Store stresses at Gauss points

    def compute_GP_strains_stresses_finiteStrain(self) -> None:
        """Compute strains and stresses for each element."""
        if self.parameters['num_dimensions'] == 2:
            dim = 3
        elif self.parameters['num_dimensions'] == 3:
            dim = 6
        else:
            raise ValueError(f"self.parameters['num_dimensions'] = {self.parameters['num_dimensions']} is not correct.")
        ndim = self.parameters['num_dimensions']
        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, ndim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()

        # Additional storage for Gauss point coordinates, strains, and stresses
        self.gauss_point_coordinates = torch.zeros((self.num_element, len(gauss_points), ndim), device=self.device)
        self.gauss_point_E = torch.zeros((self.num_element, len(gauss_points), dim), device=self.device)
        self.gauss_point_F = torch.zeros((self.num_element, len(gauss_points), 9), device=self.device)
        self.gauss_point_P = torch.zeros((self.num_element, len(gauss_points), 9), device=self.device)
        self.gauss_point_S = torch.zeros((self.num_element, len(gauss_points), dim), device=self.device)
        self.gauss_point_Cauchy = torch.zeros((self.num_element, len(gauss_points), dim), device=self.device)

        for i in range(self.num_element):
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes]
            elem_dof_indices = torch.cat([(elem_nodes*ndim + i).unsqueeze(0) for i in range(ndim)], dim=0)
            elem_dof_indices = elem_dof_indices.t().contiguous().view(-1)

            elem_displacements = self.global_displacements[elem_dof_indices]

            for j, gauss_point in enumerate(gauss_points):
                N = self.elementClass.shape_functions(gauss_point, device=self.device)
                physical_coords = torch.mm(N.unsqueeze(0), node_coords).squeeze(0)
                self.gauss_point_coordinates[i, j] = physical_coords

                # Compute strains and stresses at each Gauss point
                dN_dxi = self.elementClass.shape_function_derivatives(gauss_point, device=self.device)
                dx_dxi = J = self.elementClass.jacobian(node_coords, dN_dxi, device=self.device)
                

                dN_dx = torch.einsum('im, mj -> ij',dN_dxi, dx_dxi.inverse())
                F = self.computeF(dN_dx, elem_displacements) # F.shape = (3,3)

                self.gauss_point_F[i, j] = F.reshape(1,-1).squeeze(0)

                Svec = self.material_dict[(i, j)].update_states(F=F, strain=None).type(torch.float64)
                Smat = torch.tensor([[Svec[0], Svec[3], Svec[5]],
                                     [Svec[3], Svec[1], Svec[4]],
                                     [Svec[5], Svec[4], Svec[2]]])
                # compute Lagrangian strain
                E = 1 /2 * (F @ F.T - torch.eye(3))
                EVec = Mat2Vec_Vogit(E)

                # [S11, S22, S33, S23, S13, S12] <-- [S11, S22, S33, S12, S23, S13]
                S_vogit = torch.tensor([Svec[0], Svec[1], Svec[2], Svec[4], Svec[5], Svec[3]], device=self.device)

                J = torch.det(F)
                CauchyMat = (1/J) * F @ Smat @ F.T
                Pmat = F @ Smat
                self.gauss_point_P[i, j] = Pmat.reshape(1,-1).squeeze(0)

                CauchyVec = Mat2Vec_Vogit(CauchyMat)
                
                # Store Lagrangian strain at Gauss points
                if self.ndim == 3:
                    self.gauss_point_S[i, j] = S_vogit.squeeze()  # Store PK2 at Gauss points
                    self.gauss_point_E[i, j] = EVec.squeeze()
                    self.gauss_point_Cauchy[i, j] = CauchyVec.squeeze()
                else:
                    self.gauss_point_S[i, j] = S_vogit.squeeze()[torch.tensor([0, 1, 5])]  # Store PK2 at Gauss points
                    self.gauss_point_E[i, j] = EVec.squeeze()[torch.tensor([0, 1, 5])]
                    self.gauss_point_Cauchy[i, j] = CauchyVec.squeeze()[torch.tensor([0, 1, 5])]




                
            


    def compute_element_stiffness_with_shear_locking(self) -> None:
        
        dof_per_element = self.elementClass.element_dof
        n_dim = self.parameters['num_dimensions']
        self.element_stiffnesses = torch.zeros(self.num_element, dof_per_element, dof_per_element, dtype=torch.float64, device=self.device)

        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, n_dim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()

        

        #for i in tqdm(range(self.num_element), desc='Computing Stiffness Matrix'):
        for i in range(self.num_element):  
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes].type(torch.float64)

            # Define B_matrix_revised with enough size from the start.
            if n_dim==2:
                B_matrix_revised = torch.zeros((3, dof_per_element + n_dim * n_dim), dtype=torch.float64, device=self.device)
            elif n_dim == 3:
                B_matrix_revised = torch.zeros((6, dof_per_element + n_dim * n_dim), dtype=torch.float64, device=self.device)
            else:
                raise ValueError(f"self.parameters['num_dimensions'] = {n_dim} is not correct.")

            K_whole = torch.zeros((dof_per_element + n_dim * n_dim, dof_per_element + n_dim * n_dim), dtype=torch.float64, device=self.device)
            K_uu = torch.zeros(dof_per_element, dof_per_element, dtype=torch.float64, device=self.device)
            K_au = torch.zeros(n_dim*n_dim, dof_per_element, dtype=torch.float64, device=self.device)
            K_ua = torch.zeros(dof_per_element, n_dim*n_dim, dtype=torch.float64, device=self.device)
            K_aa = torch.zeros(n_dim*n_dim, n_dim*n_dim, dtype=torch.float64, device=self.device)

            # compute (xi_0, xi_1, xi_2) = (0, 0, 0) jacobian
            if n_dim == 2:
                gauss_point0 = torch.tensor([0,0], device=self.device).to(dtype=torch.float64)
            elif n_dim == 3:
                gauss_point0 = torch.tensor([0,0,0], device=self.device).to(dtype=torch.float64)
            else:
                raise ValueError(f"{n_dim} is not an allowed value.")
            
            dN_dxi0 = self.elementClass.shape_function_derivatives(gauss_point0, device=self.device)
            J0 = self.elementClass.jacobian(node_coords, dN_dxi0, device=self.device)
            detJ0 = torch.linalg.det(J0).to(dtype=torch.float64)

            
            
            for j, (gauss_point, weight) in enumerate(zip(gauss_points, weights)):

                dN_dxi = self.elementClass.shape_function_derivatives(gauss_point, device=self.device)
                dx_dxi = J = self.elementClass.jacobian(node_coords, dN_dxi, device=self.device)
                detJ = torch.linalg.det(J)
                invJ = torch.linalg.inv(J)
             
                

                B_matrix = self.elementClass.compute_B_matrix(dN_dxi, J, device=self.device)
                if n_dim ==2: # append_matrix.shape = (3, 4)
                    append_matrix = torch.tensor([
                        [detJ0/detJ*gauss_point[0]*invJ[0,0], 0, detJ0/detJ*gauss_point[1]*invJ[1,0], 0],
                        [0, detJ0/detJ*gauss_point[0]*invJ[0,1], 0, detJ0/detJ*gauss_point[1]*invJ[1,1]], 
                        [detJ0/detJ*gauss_point[0]*invJ[0,1], detJ0/detJ*gauss_point[0]*invJ[0,0], detJ0/detJ*gauss_point[1]*invJ[1,1], detJ0/detJ*gauss_point[1]*invJ[1,0]]
                    ],device=self.device).to(dtype=torch.float64)

                elif n_dim ==3: # append_matrix.shape = (6, 9)
                    append_matrix = torch.tensor([
                        [detJ0/detJ*gauss_point[0]*invJ[0,0], 0, 0, detJ0/detJ*gauss_point[1]*invJ[1,0], 0, 0, detJ0/detJ*gauss_point[2]*invJ[2,0], 0, 0],
                        [0, detJ0/detJ*gauss_point[0]*invJ[0,1], 0, 0, detJ0/detJ*gauss_point[1]*invJ[1,1], 0, 0, detJ0/detJ*gauss_point[2]*invJ[2,1], 0],
                        [0, 0, detJ0/detJ*gauss_point[0]*invJ[0,2], 0, 0, detJ0/detJ*gauss_point[1]*invJ[1,2], 0, 0, detJ0/detJ*gauss_point[2]*invJ[2,2]],
                        [0, detJ0/detJ*gauss_point[0]*invJ[0,2], detJ0/detJ*gauss_point[0]*invJ[0,1], 0, detJ0/detJ*gauss_point[1]*invJ[1,2], detJ0/detJ*gauss_point[1]*invJ[1,1], 0, detJ0/detJ*gauss_point[2]*invJ[2,2], detJ0/detJ*gauss_point[2]*invJ[2,1]],
                        [detJ0/detJ*gauss_point[0]*invJ[0,2], 0, detJ0/detJ*gauss_point[0]*invJ[0,0], detJ0/detJ*gauss_point[1]*invJ[1,2], 0, detJ0/detJ*gauss_point[1]*invJ[1,0], detJ0/detJ*gauss_point[2]*invJ[2,2], 0, detJ0/detJ*gauss_point[2]*invJ[2,0],],
                        [detJ0/detJ*gauss_point[0]*invJ[0,1], detJ0/detJ*gauss_point[0]*invJ[0,0], 0, detJ0/detJ*gauss_point[1]*invJ[1,1], detJ0/detJ*gauss_point[1]*invJ[1,0], 0, detJ0/detJ*gauss_point[2]*invJ[2,1], detJ0/detJ*gauss_point[2]*invJ[2,0], 0],
                        
                    ],device=self.device).to(dtype=torch.float64)

                else:
                    raise ValueError(f"Problem dimension {n_dim} is not an allowed value.")
                B_matrix_revised = torch.cat((B_matrix, append_matrix), dim=1)


                D_matrix = self.material_dict[(i, j)].consistent_tangent(strain=None, F=None).type(torch.float64)

                K_whole += weight * detJ * B_matrix_revised.T @ D_matrix @ B_matrix_revised


            K_uu = K_whole[0:dof_per_element, 0:dof_per_element]
            K_au = K_whole[dof_per_element:(dof_per_element+n_dim*n_dim), 0:dof_per_element]
            K_ua = K_whole[0:dof_per_element, dof_per_element:(dof_per_element+n_dim*n_dim)]
            K_aa = K_whole[dof_per_element:(dof_per_element+n_dim*n_dim), dof_per_element:(dof_per_element+n_dim*n_dim)]
            

            self.element_stiffnesses[i] =   K_uu - K_ua @ torch.inverse(K_aa) @ K_au

    def computeF(self, dN_dx, elem_displacements):
        '''F_ij = I_ij + dN_a/dx_j u^a_i'''
        # reshape elem_displacements --> (-1, dim)
        n_dim = self.parameters['num_dimensions']
        elem_displacements = elem_displacements.reshape(-1, n_dim)
        output = torch.eye(3, dtype=torch.float64, device=self.device)
        if n_dim == 2:
            output[:2,:2] += torch.einsum('aj, ai -> ij',dN_dx, elem_displacements)
        if n_dim == 3:
            output += torch.einsum('aj, ai -> ij',dN_dx, elem_displacements)
        return output






    #################
    ##    modal
    #################

    def compute_mass_matrix(self) -> None:
            
        dof_per_element = self.elementClass.element_dof
        n_node_per_element  = self.elementClass.node_per_element
        n_dim = self.parameters['num_dimensions']
        self.element_massMatrix = torch.zeros(self.num_element, dof_per_element, dof_per_element, device=self.device)

        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, n_dim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()


        for idx in tqdm(range(self.num_element), desc='Computing Mass Matrix'):
            elem_nodes = self.element_node_indices[idx]
            node_coords = self.node_coords[elem_nodes].type(torch.float64)


            M_elem = torch.zeros(n_node_per_element, n_node_per_element, dtype=torch.float64, device=self.device)
            
            for j, (gauss_point, weight) in enumerate(zip(gauss_points, weights)):
                N = self.elementClass.shape_functions(gauss_point, device=self.device)
                dN_dxi = self.elementClass.shape_function_derivatives(gauss_point, device=self.device)
                J = self.elementClass.jacobian(node_coords, dN_dxi, device=self.device)
                detJ = torch.det(J)
                
                rho = self.material_dict[(idx, j)].rho

                M_elem += weight * detJ * rho * torch.einsum('i,j -> ij', N, N)


            expanded_mat = torch.zeros(dof_per_element, dof_per_element, dtype=torch.float64, device=self.device)
            if n_dim == 2:
                for i in range(M_elem.shape[0]):
                    for j in range(M_elem.shape[1]):
                        expanded_mat[i*2, j*2] = M_elem[i, j]      
                        expanded_mat[i*2+1, j*2+1] = M_elem[i, j] 
            elif n_dim ==3:
                for i in range(M_elem.shape[0]):
                    for j in range(M_elem.shape[1]):
                        expanded_mat[i*3, j*3] = M_elem[i, j]      
                        expanded_mat[i*3+1, j*3+1] = M_elem[i, j]
                        expanded_mat[i*3+2, j*3+2] = M_elem[i, j]


            self.element_massMatrix[idx] = expanded_mat

    def assemble_global_mass_matrix(self):
        """
        Assemble the global stiffness matrix from the element stiffness matrices.
        """               
        # Initialize the global stiffness matrix to zero
        self.global_mass_matrix = torch.zeros(self.num_dofs, self.num_dofs, device=self.device).to(dtype=torch.float64)
        
        # Determine the global degree of freedom indices for all elements
        # Calculate the DOF indices for each dimension of each node in the elements
        dof_per_dimension = torch.arange(self.parameters['num_dimensions']).to(self.device).unsqueeze(0)  # [0,1], or[0,1,2]

        # Use advanced indexing to assemble all element stiffness matrices into the global stiffness matrix simultaneously
        n_dim = self.parameters['num_dimensions']

        for i in tqdm(range(self.num_element), desc="Assembling Mass Matrix"):
            elem_nodes = self.element_node_indices[i]
            elem_dof_indices = (elem_nodes.unsqueeze(-1) * n_dim + dof_per_dimension).view(-1).long()
            
            # Assemble the element stiffness matrix into the global stiffness matrix
            self.global_mass_matrix[elem_dof_indices[:, None], elem_dof_indices[None, :]] += self.element_massMatrix[i]

    def solve_system_modal(self):
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
        M_sub = self.global_mass_matrix[unknown_dof_indices, :][:, unknown_dof_indices]
        R_sub = self.global_load[unknown_dof_indices]

        # q=M**(1/2); H=Q @ A @ Q.T
        M_sub_np  = M_sub.cpu().numpy()
        M_sub_sqrt_np = sqrtm(M_sub_np)
        M_sub_sqrt = torch.from_numpy(M_sub_sqrt_np)
        M_sub_sqrt = M_sub_sqrt.to(dtype=torch.float64, device=self.device)

        M_sub_sqrt_inv = torch.linalg.pinv(M_sub_sqrt)
        H_sub = M_sub_sqrt_inv @ K_sub @ M_sub_sqrt_inv

        Q, S, _ = torch.linalg.svd(H_sub, full_matrices=True)
        
        S_sorted, sorted_indices = torch.sort(S, descending=False)

        Q_sorted = Q[:, sorted_indices]

        nmodrbm = 3 if self.parameters['num_dimensions'] == 2 else 6

        Nmode=3
        self.global_displacements_mod = torch.zeros(self.num_dofs, Nmode,  device=self.device).to(dtype=torch.float64)

        self.global_displacements_mod[unknown_dof_indices] = M_sub_sqrt_inv @ Q_sorted[:,:Nmode]



        self.global_displacements = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        self.global_displacements = torch.sum(self.global_displacements_mod, dim=1)
        # Fill known displacements, if any
        if self.node_dof_disp.numel() > 0:
            self.global_displacements[known_dof_indices] = self.node_dof_disp[:, 2].to(dtype=torch.float64)


class FiniteElementModel_dynamic_Newmark(FiniteElementModel):
    def __init__(self, device: str='cuda') -> None:
        super().__init__(self)

    def compute_acc_t0(self, u0_func, externalForcefunc) -> None:
        """ Compute the self.global_acc_t0 """

        u0 = u0_func(self.node_coords, self.device)
        
        self.u_prev = u0
        self.v_prev = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        self.mkpres = self.global_mass_matrix + 0.5 *self.beta2*self.dt*self.dt * self.global_stiffness

        #self.a_prev = self.global_mass_matrix.inverse() @ (-self.global_stiffness @ u0 + self.global_load)
        
        # sub matrix
        #############
        # Identify known DOFs from self.node_dof_disp
        if self.node_dof_disp.numel() > 0:
            known_dof_indices = (self.node_dof_disp[:, 0] * self.parameters['num_dimensions'] + self.node_dof_disp[:, 1]).long()
        else:
            known_dof_indices = torch.tensor([], dtype=torch.long, device=self.device)
        # Identify unknown DOFs
        all_dofs = torch.arange(self.num_dofs, device=self.device)
        unknown_dof_indices = torch.tensor([idx for idx in all_dofs if idx not in known_dof_indices], device=self.device)

        # Extract submatrix and subvector
        mkpres_sub = self.mkpres[unknown_dof_indices, :][:, unknown_dof_indices]
        stiffness_sub = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        
        u0_sub = u0[unknown_dof_indices]
        global_load_sub = self.global_load[unknown_dof_indices]

        A = mkpres_sub
        B = (-1*stiffness_sub @u0_sub + global_load_sub)
        x_sub = torch.linalg.lstsq(A, B).solution

        self.a_prev = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        self.a_prev[unknown_dof_indices] = x_sub
        

        # consider prescribed acc. for boundary condition
        if self.node_dof_disp.numel() > 0: # FIXED only
            # Incorporate prescribed displacements into the global load vector using vectorized operations
            node_indices = self.node_dof_disp[:, 0].long()
            dof_indices = self.node_dof_disp[:, 1].long()
            values = self.node_dof_disp[:, 2].to(dtype=torch.float64)

            global_dof_indices = node_indices * self.parameters['num_dimensions'] + dof_indices
            self.a_prev.index_add_(dim=0, index=global_dof_indices, source=values)

        self.global_displacements = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        self.global_displacements = u0



    def assemble_global_load_vector_dynamic(self, externalForcefunc, t) -> None:
        """
        Assemble the global load vector R from the prescribed displacements and tractions.
        """
        # Initialize the global load vector to zero
        self.global_load = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)

        # Identify known DOFs from self.node_dof_disp
        if self.node_dof_disp.numel() > 0:
            known_dof_indices = (self.node_dof_disp[:, 0] * self.parameters['num_dimensions'] + self.node_dof_disp[:, 1]).long()
        else:
            known_dof_indices = torch.tensor([], dtype=torch.long, device=self.device)
        # Identify unknown DOFs
        all_dofs = torch.arange(self.num_dofs, device=self.device)
        unknown_dof_indices = torch.tensor([idx for idx in all_dofs if idx not in known_dof_indices], device=self.device)

        # Extract submatrix and subvector
        # K_unknown_unknown = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        K_known_known = self.global_stiffness[known_dof_indices, :][:, known_dof_indices]
        K_unknown_known = self.global_stiffness[unknown_dof_indices, :][:, known_dof_indices]

        if self.node_dof_disp.numel() > 0:
            # Incorporate prescribed displacements into the global load vector using vectorized operations
            node_indices = self.node_dof_disp[:, 0].long()
            dof_indices = self.node_dof_disp[:, 1].long()
            values = self.node_dof_disp[:, 2].to(dtype=torch.float64)

            global_dof_indices = node_indices * self.parameters['num_dimensions'] + dof_indices
            self.global_load.index_add_(dim=0, index=global_dof_indices, source= - K_known_known @ values)

        forces = externalForcefunc(self.node_coords, t=t, device=self.device)
        self.global_load[::2] += forces[:, 0]
        self.global_load[1::2] += forces[:, 1]
            

        if self.node_dof_forces.numel() > 0:
            # Incorporate nodal forces into the global load vector using vectorized operations
            force_node_indices = self.node_dof_forces[:, 0].long()
            force_dof_indices = self.node_dof_forces[:, 1].long()
            force_values = self.node_dof_forces[:, 2].to(dtype=torch.float64)
            force_global_dof_indices = force_node_indices * self.parameters['num_dimensions'] + force_dof_indices
            self.global_load.index_add_(0, force_global_dof_indices, force_values)

        self.global_load[unknown_dof_indices] += -K_unknown_known @ values

    def solve_system_dynamic(self):
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
        mkpres_sub = self.mkpres[unknown_dof_indices, :][:, unknown_dof_indices]
        R_sub = self.global_load[unknown_dof_indices]

        a_prev_sub = self.a_prev[unknown_dof_indices]
        v_prev_sub = self.v_prev[unknown_dof_indices]
        u_prev_sub = self.u_prev[unknown_dof_indices]

        
        # a_current_sub = mkpres_sub.inverse() @ (R_sub - K_sub @ (u_prev_sub + 
        #                                               self.dt*v_prev_sub+
        #                                               self.dt*self.dt/2*(1-self.beta2) * a_prev_sub))
        A = mkpres_sub
        B = (R_sub - K_sub @ (u_prev_sub + 
                            self.dt*v_prev_sub+
                            self.dt*self.dt/2*(1-self.beta2) * a_prev_sub))
        a_current_sub = torch.linalg.lstsq(A, B).solution
        
        
        v_current_sub = v_prev_sub + self.dt*((1.-self.beta1)*a_prev_sub + self.beta1*a_current_sub)

        u_current_sub = u_prev_sub + self.dt*v_prev_sub + self.dt*self.dt/2*((1.-self.beta2)*a_prev_sub + self.beta2*a_current_sub)


        # Create a global displacement vector
        self.global_displacements = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        self.global_displacements[unknown_dof_indices] = u_current_sub.squeeze()

        # Fill known displacements, if any
        if self.node_dof_disp.numel() > 0:
            self.global_displacements[known_dof_indices] = self.node_dof_disp[:, 2].to(dtype=torch.float64)

        # update the self.a_prev...

        self.a_prev[unknown_dof_indices] = a_current_sub.squeeze()
        self.v_prev[unknown_dof_indices] = v_current_sub.squeeze()
        self.u_prev[unknown_dof_indices] = u_current_sub.squeeze()



class FiniteElementModel_nonlinear(FiniteElementModel):
    def __init__(self, device: str='cuda') -> None:
        super().__init__(self)
        
    def compute_element_stiffness_nonlinear(self) -> None:
        
        dof_per_element = self.elementClass.element_dof
        n_dim = self.parameters['num_dimensions']
        self.element_stiffnesses = torch.zeros(self.num_element, dof_per_element, dof_per_element, device=self.device)

        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, n_dim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()

        for i in range(self.num_element):
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes].type(torch.float64)

            K_elem = torch.zeros(dof_per_element, dof_per_element, dtype=torch.float64, device=self.device)
            
            ndim = self.parameters['num_dimensions']
            elem_dof_indices = torch.cat([(elem_nodes*ndim + i).unsqueeze(0) for i in range(ndim)], dim=0)
            elem_dof_indices = elem_dof_indices.t().contiguous().view(-1)
            elem_displacements = self.global_displacements_temp[elem_dof_indices]

            for j, (gauss_point, weight) in enumerate(zip(gauss_points, weights)):

                dN_dxi = self.elementClass.shape_function_derivatives(gauss_point, device=self.device)
                dx_dxi = J = self.elementClass.jacobian(node_coords, dN_dxi, device=self.device)
                detJ = torch.det(J)
                
                B_matrix = self.elementClass.compute_B_matrix(dN_dxi, J, device=self.device)

                # material law : eval stain @ gp (Vogit notation)
                strainVec = B_matrix @ elem_displacements
                strain = Vec2Mat_Vogit(strainVec)

                dN_dx = torch.einsum('im, mj -> ij',dN_dxi, dx_dxi.inverse())
                F = self.computeF(dN_dx, elem_displacements) # F.shape = (3,3)
                
                C_matrix = self.material_dict[(i, j)].consistent_tangent(strain=strain, F=F).type(torch.float64)

                K_elem += weight * detJ * torch.einsum('ji,jk,kl->il', B_matrix, C_matrix, B_matrix)
            self.element_stiffnesses[i] = K_elem



    def compute_element_residual(self):
        dof_per_element = self.elementClass.element_dof
        node_per_element = self.elementClass.node_per_element
        n_dim = self.parameters['num_dimensions']
        self.element_residual = torch.zeros((self.num_element, dof_per_element), dtype=torch.float64, device=self.device)

        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, n_dim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()

        #for i in tqdm(range(self.num_element), desc='Computing residual stress'):
        for i in range(self.num_element):
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes].type(torch.float64)
        
            ndim = self.parameters['num_dimensions']
            elem_dof_indices = torch.cat([(elem_nodes*ndim + i).unsqueeze(0) for i in range(ndim)], dim=0)
            elem_dof_indices = elem_dof_indices.t().contiguous().view(-1)
            elem_displacements = self.global_displacements_temp[elem_dof_indices]
            residual = torch.zeros((node_per_element, ndim), dtype=torch.float64, device=self.device)

            for j, (gauss_point, weight) in enumerate(zip(gauss_points, weights)):

                dN_dxi = self.elementClass.shape_function_derivatives(gauss_point, device=self.device)
                dx_dxi = J = self.elementClass.jacobian(node_coords, dN_dxi, device=self.device)
                detJ = torch.det(J)
                
                B_matrix = self.elementClass.compute_B_matrix(dN_dxi, J, device=self.device)

                # material law : eval stain @ gp (Vogit notation)
                strainVec = B_matrix @ elem_displacements
                strain = Vec2Mat_Vogit(strainVec) # strain.shape = (3) or (6)

                dN_dx = torch.einsum('im, mj -> ij',dN_dxi, dx_dxi.inverse())
                F = self.computeF(dN_dx, elem_displacements) # F.shape = (3,3)

                stressVector = self.material_dict[(i, j)].update_states(strain=strain, F=F)

                stress = Vec2Mat_Vogit(stressVector, device=self.device)

                residual += torch.einsum('ij, aj ->ai', stress, dN_dx) *  weight * detJ
            
            self.element_residual[i] = residual.reshape(-1)



    def assemble_global_residual(self):
        """
        Assemble the global load vector R from the prescribed displacements and tractions.
        """
        # Initialize the global load vector to zero
        self.global_residual_stress = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)

        #for i in tqdm(range(self.num_element), desc='Computing stress/strain'):
        for i in range(self.num_element):
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes]
            ndim = self.parameters['num_dimensions']
            elem_dof_indices = torch.cat([(elem_nodes*ndim + i).unsqueeze(0) for i in range(ndim)], dim=0)
            elem_dof_indices = elem_dof_indices.t().contiguous().view(-1)

            self.global_residual_stress[elem_dof_indices] += self.element_residual[i]


        
        


    def assemble_global_load_vector_nonlinear(self, factor) -> None:
        """
        Assemble the global load vector R from the prescribed displacements and tractions.
        """
        # Initialize the global load vector to zero
        self.global_load = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        known_dof_indices, unknown_dof_indices = self._get_dof_indices()

        # Extract submatrix and subvector
        # K_unknown_unknown = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        K_known_known = self.global_stiffness[known_dof_indices, :][:, known_dof_indices]
        K_unknown_known = self.global_stiffness[unknown_dof_indices, :][:, known_dof_indices]
        

        if self.node_dof_disp.numel() > 0:
            # Incorporate prescribed displacements into the global load vector using vectorized operations
            values = self.node_dof_disp[:, 2].to(dtype=torch.float64) 

            self.current_known_disp = values * factor
            self.global_load.index_add_(dim=0, index=known_dof_indices, source= - K_known_known @ self.current_known_disp)
            #self.global_load.index_add_(dim=0, index=unknown_dof_indices, source= - K_unknown_known @ self.current_known_disp)
            # NOTE:  不能當成 prescribed displacement boundary condition

        if self.node_dof_forces.numel() > 0:
            # Incorporate nodal forces into the global load vector using vectorized operations
            force_node_indices = self.node_dof_forces[:, 0].long()
            force_dof_indices = self.node_dof_forces[:, 1].long()
            force_values = self.node_dof_forces[:, 2].to(dtype=torch.float64) * factor
            force_global_dof_indices = force_node_indices * self.parameters['num_dimensions'] + force_dof_indices
            self.global_load.index_add_(0, force_global_dof_indices, force_values)





    def solve_system_nonlinear(self):
        """
        Solve the global system of equations to get the nodal displacements.
        """

        # Identify known DOFs from self.node_dof_disp
        known_dof_indices, unknown_dof_indices = self._get_dof_indices()

        # Extract submatrix and subvector
        K_sub = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        R_sub = self.global_load[unknown_dof_indices]
        F_sub = self.global_residual_stress[unknown_dof_indices]
        B_sub = R_sub - F_sub


        # Solve the system
        self.u_incr_sub = torch.linalg.solve(K_sub, B_sub.unsqueeze(1))
        
        
        self.global_displacements_temp[unknown_dof_indices] += self.u_incr_sub.squeeze()

        if known_dof_indices.numel() > 0:
            self.global_displacements_temp[known_dof_indices] = self.current_known_disp

    def update_displacement(self):
        self.global_displacements = self.global_displacements_temp





    def init_global_displacements_temp(self):
        self.ndim = self.parameters['num_dimensions']
        self.global_displacements_temp = torch.zeros(self.num_dofs, device=self.device, dtype=torch.float64)

    def update_prescribed_global_displacements_temp(self, factor):
        known_dof_indices, unknown_dof_indices = self._get_dof_indices()
        if self.node_dof_disp.numel() > 0:
            # Incorporate prescribed displacements into the global load vector using vectorized operations
            values = self.node_dof_disp[:, 2].to(dtype=torch.float64) 
            # nonlinear added 
            self.current_known_disp = values * factor
            self.global_displacements_temp[known_dof_indices] = self.current_known_disp
            


    def recoverError(self):
        known_dof_indices, unknown_dof_indices = self._get_dof_indices()
        self.global_abs_u_incr = torch.zeros(self.num_dofs, device=self.device, dtype=torch.float64)
        self.global_abs_u_incr[unknown_dof_indices] = torch.abs(self.u_incr_sub.squeeze())
        # self.global_abs_u_incr[known_dof_indices]  assign --> 0


    def compute_element_stiffness_nonlinear_finiteStrain(self):
        dof_per_element = self.elementClass.element_dof
        self.element_stiffnesses = torch.zeros(self.num_element, dof_per_element, dof_per_element, device=self.device)
        self.element_residual = torch.zeros((self.num_element, dof_per_element), dtype=torch.float64, device=self.device)

        gauss_quadrature = GaussQuadrature(self.elementClass.node_per_element, self.ndim, device=self.device)
        gauss_points, weights = gauss_quadrature.get_points_and_weights()

        for i in range(self.num_element):
            elem_nodes = self.element_node_indices[i]
            node_coords = self.node_coords[elem_nodes].type(torch.float64)

            K_elem = torch.zeros(dof_per_element, dof_per_element, dtype=torch.float64, device=self.device)
            residual = torch.zeros(dof_per_element, dtype=torch.float64, device=self.device)
            
            ndim = self.parameters['num_dimensions']
            elem_dof_indices = torch.cat([(elem_nodes*ndim + i).unsqueeze(0) for i in range(ndim)], dim=0)
            elem_dof_indices = elem_dof_indices.t().contiguous().view(-1)
            elem_displacements = self.global_displacements_temp[elem_dof_indices]

            for j, (gauss_point, weight) in enumerate(zip(gauss_points, weights)):
                
                dN_dxi = self.elementClass.shape_function_derivatives(gauss_point, device=self.device)
                dx_dxi = J = self.elementClass.jacobian(node_coords, dN_dxi, device=self.device)
                detJ = torch.det(J)
                
                dN_dx = torch.einsum('im, mj -> ij',dN_dxi, dx_dxi.inverse())
                F = self.computeF(dN_dx, elem_displacements) # F.shape = (3,3)
                BN, BG = self.computeBG_BN(dN_dx=dN_dx, F=F)

                

                # Stress = 2nd PK stress [S11, S22, S33, S12, S23, S13]
                Svec = self.material_dict[(i, j)].update_states(F=F, strain=None).type(torch.float64)

                Smat = torch.tensor([[Svec[0], Svec[3], Svec[5]],
                                     [Svec[3], Svec[1], Svec[4]],
                                     [Svec[5], Svec[4], Svec[2]]])

                
                if self.ndim == 3:
                    # Build SHEAD matrix
                    SHEAD = torch.zeros(9, 9, dtype=Svec.dtype, device=Svec.device)
                    SHEAD[0:3, 0:3] = Smat
                    SHEAD[3:6, 3:6] = Smat
                    SHEAD[6:9, 6:9] = Smat
                    # Material stiffness [6x6]
                    dSdE = self.material_dict[(i, j)].consistent_tangent(F=F, strain=None).type(torch.float64)

                    K_elem += weight * detJ * (BN.T @ dSdE @ BN + BG.T @ SHEAD @ BG)
                    residual += weight * detJ * (BN.T @ Svec.squeeze())
                else:
                    # Build SHEAD matrix
                    SHEAD = torch.zeros(4, 4, dtype=Svec.dtype, device=Svec.device)
                    SHEAD[0:2, 0:2] = Smat[:2, :2]
                    SHEAD[2:4, 2:4] = Smat[:2, :2]

                    # Material stiffness [6x6] <-- [S11, S22, S33, S12, S23, S13]
                    dSdE = self.material_dict[(i, j)].consistent_tangent(F=F, strain=None).type(torch.float64)
                    rows = torch.tensor([0, 1, 3])
                    cols = torch.tensor([0, 1, 3])
                    dSdE = dSdE[rows][:, cols]

                    K_elem += weight * detJ * (BN.T @ dSdE @ BN + BG.T @ SHEAD @ BG)
                    residual += weight * detJ * (BN.T @ Svec.squeeze()[rows])
                          
            self.element_stiffnesses[i] = K_elem
            self.element_residual[i] = residual


    def computeBG_BN(self, dN_dx, F):
        num_node = self.elementClass.node_per_element
        n_dim = self.parameters['num_dimensions']
        if n_dim == 2:
            BN = torch.zeros(3, num_node * n_dim, dtype=torch.float64, device=self.device)
            BG = torch.zeros(4, num_node * n_dim, dtype=torch.float64, device=self.device)

            for I in range(num_node):
                COL = I * n_dim
                # 計算 BN 矩陣
                BN[:, COL:COL + n_dim] = torch.tensor([
                    [dN_dx[I, 0] * F[0, 0],  dN_dx[I, 0] * F[1, 0]],
                    [dN_dx[I, 1] * F[0, 1],  dN_dx[I, 1] * F[1, 1]],
                    [dN_dx[I, 0] * F[0, 1] + dN_dx[I, 1] * F[0, 0], \
                                dN_dx[I, 0] * F[1, 1] + dN_dx[I, 1] * F[1, 0]],
                ], dtype=torch.float64, device=self.device)

                # 計算 BG 矩陣
                BG[:, COL:COL + n_dim] = torch.tensor([
                    [dN_dx[I, 0], 0],
                    [dN_dx[I, 1], 0],
                    [0, dN_dx[I, 0]],
                    [0, dN_dx[I, 1]],
                ], dtype=torch.float64, device=self.device)
        else:
            BN = torch.zeros(6, num_node * n_dim, dtype=torch.float64, device=self.device)
            BG = torch.zeros(9, num_node * n_dim, dtype=torch.float64, device=self.device)

            for I in range(num_node):
                COL = I * n_dim
                # 計算 BN 矩陣
                BN[:, COL:COL + n_dim] = torch.tensor([
                    [dN_dx[I, 0] * F[0, 0],  dN_dx[I, 0] * F[1, 0], dN_dx[I, 0] * F[2, 0]],
                    [dN_dx[I, 1] * F[0, 1],  dN_dx[I, 1] * F[1, 1], dN_dx[I, 1] * F[2, 1]],
                    [dN_dx[I, 2] * F[0, 2],  dN_dx[I, 2] * F[1, 2], dN_dx[I, 2] * F[2, 2]],
                    [dN_dx[I, 0] * F[0, 1] + dN_dx[I, 1] * F[0, 0], dN_dx[I, 0] * F[1, 1] + dN_dx[I, 1] * F[1, 0], dN_dx[I, 0] * F[2, 1] + dN_dx[I, 1] * F[2, 0]],
                    [dN_dx[I, 1] * F[0, 2] + dN_dx[I, 2] * F[0, 1], dN_dx[I, 1] * F[1, 2] + dN_dx[I, 2] * F[1, 1], dN_dx[I, 1] * F[2, 2] + dN_dx[I, 2] * F[2, 1]],
                    [dN_dx[I, 0] * F[0, 2] + dN_dx[I, 2] * F[0, 0], dN_dx[I, 0] * F[1, 2] + dN_dx[I, 2] * F[1, 0], dN_dx[I, 0] * F[2, 2] + dN_dx[I, 2] * F[2, 0]]
                ], dtype=torch.float64, device=self.device)

                # 計算 BG 矩陣
                BG[:, COL:COL + n_dim] = torch.tensor([
                    [dN_dx[I, 0], 0, 0],
                    [dN_dx[I, 1], 0, 0],
                    [dN_dx[I, 2], 0, 0],
                    [0, dN_dx[I, 0], 0],
                    [0, dN_dx[I, 1], 0],
                    [0, dN_dx[I, 2], 0],
                    [0, 0, dN_dx[I, 0]],
                    [0, 0, dN_dx[I, 1]],
                    [0, 0, dN_dx[I, 2]]
                ], dtype=torch.float64, device=self.device)

        return BN, BG

    def solve_system_nonlinear_finiteStrain(self):
        """
        Solve the global system of equations to get the nodal displacements.
        """

        # Identify known DOFs from self.node_dof_disp
        known_dof_indices, unknown_dof_indices = self._get_dof_indices()

        # Extract submatrix and subvector
        K_sub = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        R_sub = self.global_load[unknown_dof_indices]
        F_sub = self.global_residual_stress[unknown_dof_indices]
        B_sub = R_sub - F_sub
        

        # Solve the system
        self.u_incr_sub = torch.linalg.solve(K_sub, B_sub.unsqueeze(1))

        self.global_displacements_temp[unknown_dof_indices] += self.u_incr_sub.squeeze()

        if known_dof_indices.numel() > 0:
            self.global_displacements_temp[known_dof_indices] = self.current_known_disp


    def assemble_global_load_vector_nonlinear_finiteStrain(self, factor) -> None:
        """
        Assemble the global load vector R from the prescribed displacements and tractions.
        """
        # Initialize the global load vector to zero
        self.global_load = torch.zeros(self.num_dofs, device=self.device).to(dtype=torch.float64)
        known_dof_indices, unknown_dof_indices = self._get_dof_indices()

        # Extract submatrix and subvector
        # K_unknown_unknown = self.global_stiffness[unknown_dof_indices, :][:, unknown_dof_indices]
        K_known_known = self.global_stiffness[known_dof_indices, :][:, known_dof_indices]
        K_unknown_known = self.global_stiffness[unknown_dof_indices, :][:, known_dof_indices]
        

        if self.node_dof_disp.numel() > 0:
            # Incorporate prescribed displacements into the global load vector using vectorized operations
            values = self.node_dof_disp[:, 2].to(dtype=torch.float64) 

            self.current_known_disp = values * factor
            self.global_load.index_add_(dim=0, index=known_dof_indices, source= - K_known_known @ self.current_known_disp)
            #self.global_load.index_add_(dim=0, index=unknown_dof_indices, source= - K_unknown_known @ self.current_known_disp)
            # NOTE:  不能當成 prescribed displacement boundary condition

        if self.node_dof_forces.numel() > 0:
            # Incorporate nodal forces into the global load vector using vectorized operations
            force_node_indices = self.node_dof_forces[:, 0].long()
            force_dof_indices = self.node_dof_forces[:, 1].long()
            force_values = self.node_dof_forces[:, 2].to(dtype=torch.float64) * factor
            force_global_dof_indices = force_node_indices * self.parameters['num_dimensions'] + force_dof_indices
            self.global_load.index_add_(0, force_global_dof_indices, force_values)

