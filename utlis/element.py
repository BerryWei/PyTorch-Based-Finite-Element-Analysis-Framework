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

        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]

        # Compute the area of the triangle
        A = 0.5 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

        # Check if area is close to zero (degenerate triangle)
        if abs(A) < 1e-10:
            raise ValueError("The provided node coordinates result in a degenerate triangle with almost zero area.")
        
        # Derivatives of shape functions
        b1 = (y2 - y3) / (2*A)
        b2 = (y3 - y1) / (2*A)
        b3 = (y1 - y2) / (2*A)

        c1 = (x3 - x2) / (2*A)
        c2 = (x1 - x3) / (2*A)
        c3 = (x2 - x1) / (2*A)

        # Construct the B-matrix
        B = torch.tensor([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ], dtype=torch.float).to(device)

        return B
    
# a = torch.tensor([1/3, 1/3])
# print(TriangularElement.shape_functions(a))
# node_coords = torch.tensor([
#     [0., 0],
#     [1, 0],
#     [0, 1]
# ]).to('cuda')

# B_matrix = TriangularElement.compute_B_matrix(node_coords)
# print(B_matrix)
