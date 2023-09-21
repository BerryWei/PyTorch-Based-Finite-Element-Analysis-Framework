import torch
from abc import ABC, abstractmethod

class BaseElement(ABC):
    element_dof = None  # 添加 element_dof 属性

    @staticmethod
    @abstractmethod
    def shape_functions(natural_coords: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to compute the shape functions.
        
        Parameters:
        - natural_coords (torch.Tensor): Natural coordinates.
        
        Returns:
        - torch.Tensor: Shape functions.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def compute_B_matrix(node_coords: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to compute the B matrix.
        
        Parameters:
        - node_coords (torch.Tensor): Node coordinates for the element.
        
        Returns:
        - torch.Tensor: B matrix.
        """
        pass


class TriangularElement(BaseElement):
    element_dof = 6
    node_per_element = 3
    
    @staticmethod
    def shape_functions(natural_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        xi, eta = natural_coords
        N = torch.tensor([1 - xi - eta, xi, eta]).to(device)
        return N
    
    @staticmethod
    def compute_B_matrix(node_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        # Checking the input shape
        if len(node_coords.shape) != 2 or node_coords.shape[0] != 3 or node_coords.shape[1] != 2:
            raise ValueError("node_coords shape should be (3, 2) for a single triangle")

        dN_dxi = torch.tensor([-1, 1, 0], dtype=torch.float).to(device)
        dN_deta = torch.tensor([-1, 0, 1], dtype=torch.float).to(device)

        J11 = node_coords[:, 0] @ dN_dxi
        J12 = node_coords[:, 0] @ dN_deta
        J21 = node_coords[:, 1] @ dN_dxi
        J22 = node_coords[:, 1] @ dN_deta

        J = torch.tensor([[J11, J12], [J21, J22]]).to(device)

        inv_J = torch.inverse(J)

        dN_dx = inv_J[0, 0] * dN_dxi + inv_J[0, 1] * dN_deta
        dN_dy = inv_J[1, 0] * dN_dxi + inv_J[1, 1] * dN_deta

        # B-matrix
        B = torch.zeros((3, 6)).to(device)
        B[0, 0::2] = dN_dx
        B[1, 1::2] = dN_dy
        B[2, 0::2] = dN_dy
        B[2, 1::2] = dN_dx

        return B
    
    
# a = torch.tensor([1/3, 1/3])
# print(TriangularElement.shape_functions(a))
# node_coords = torch.tensor([
#     [0., 0],
#     [1, 0],
#     [0, 1]
# ])

# B_matrix = TriangularElement.compute_B_matrix(node_coords)
# print(B_matrix)
