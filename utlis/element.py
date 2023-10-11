import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseElement(ABC):
    element_dof = None  

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
    def compute_B_matrix_legacy(node_coords: torch.Tensor, device='cuda') -> torch.Tensor:
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
    
    @staticmethod
    def compute_B_matrix(node_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        # Checking the input shape
        if len(node_coords.shape) != 2 or node_coords.shape[0] != 3 or node_coords.shape[1] != 2:
            raise ValueError("node_coords shape should be (3, 2) for a single triangle")


        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]

        # Compute the area of the triangle (2A for convenience in the formula)
        two_A = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        # Compute derivatives of shape functions
        dN1_dx = (y2 - y3) / ((y1-y2)*(x3-x2)-(x1-x2)*(y3-y2))
        dN2_dx = (y3 - y1) / ((y2-y3)*(x1-x3)-(x2-x3)*(y1-y3))
        dN3_dx = (y1 - y2) / ((y3-y1)*(x2-x1)-(x3-x1)*(y2-y1))

        dN1_dy = (x3 - x2) / ((y1-y2)*(x3-x2)-(x1-x2)*(y3-y2))
        dN2_dy = (x1 - x3) / ((y2-y3)*(x1-x3)-(x2-x3)*(y1-y3))
        dN3_dy = (x2 - x1) / ((y3-y1)*(x2-x1)-(x3-x1)*(y2-y1))

        # Construct the B matrix
        B = torch.tensor([[dN1_dx, 0, dN2_dx, 0, dN3_dx, 0],
                        [0, dN1_dy, 0, dN2_dy, 0, dN3_dy],
                        [dN1_dy, dN1_dx, dN2_dy, dN2_dx, dN3_dy, dN3_dx]], device=device, dtype=torch.float)

        return B



    @staticmethod
    def compute_B_matrix_vectorized(node_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        # Checking the input shape
        if len(node_coords.shape) != 3 or node_coords.shape[1] != 3 or node_coords.shape[2] != 2:
            raise ValueError("node_coords shape should be (num_elements, 3, 2) for multiple triangles")

        x1, y1 = node_coords[:, 0].t()
        x2, y2 = node_coords[:, 1].t()
        x3, y3 = node_coords[:, 2].t()

        # Compute the area of the triangles
        A = 0.5 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

        # Check if any area is close to zero (degenerate triangle)
        if torch.any(torch.abs(A) < 1e-10):
            raise ValueError("At least one set of node coordinates result in a degenerate triangle with almost zero area.")
        
        # Derivatives of shape functions
        b1 = (y2 - y3) / (2*A)
        b2 = (y3 - y1) / (2*A)
        b3 = (y1 - y2) / (2*A)

        c1 = (x3 - x2) / (2*A)
        c2 = (x1 - x3) / (2*A)
        c3 = (x2 - x1) / (2*A)

        # Construct the B-matrices
        B = torch.stack([
            torch.stack([b1, torch.zeros_like(b1), b2, torch.zeros_like(b2), b3, torch.zeros_like(b3)], dim=1),
            torch.stack([torch.zeros_like(c1), c1, torch.zeros_like(c2), c2, torch.zeros_like(c3), c3], dim=1),
            torch.stack([c1, b1, c2, b2, c3, b3], dim=1)
        ], dim=1).to(device)

        return B
    
    @staticmethod
    def boundary_nodes(face_idx: int) -> List[int]:
        """Return the nodes of a boundary (edge) of the element."""
        # Here, the boundaries are the edges for a triangle
        # This might be different for other elements (like quads or tetrahedrons)
        if face_idx == 0:
            return [0, 1]
        elif face_idx == 1:
            return [1, 2]
        elif face_idx == 2:
            return [2, 0]
        else:
            raise ValueError("Invalid face index for a triangle.")

    
    @staticmethod
    def face_gauss_points_and_weights(face_idx: int) -> Tuple[List[Tuple[float]], List[float]]:
        """Return the Gauss points and weights for a face (edge) of the element."""
        # For 2D edge integration, using 2 Gauss points as example
        return [(-1/np.sqrt(3),), (1/np.sqrt(3),)], [1.0, 1.0]
    
    @staticmethod
    def compute_face_jacobian(node_coords: torch.Tensor) -> float:
        """Compute the Jacobian for a face (edge) of the element."""
        # For the 2D triangular element, this is essentially the length of the edge
        node1, node2 = node_coords
        length = torch.norm(node2 - node1).item()
        return length
    
    @staticmethod
    def boundary_shape_functions(gp: float, device='cuda') -> torch.Tensor:
        """Return the shape functions for the boundary (1D) for the given Gauss point."""
        N = torch.tensor([0.5 * (1 - gp[0]), 0.5 * (1 + gp[0])]).to(device)

        return N




    
# a = torch.tensor([1/3, 1/3])
# print(TriangularElement.shape_functions(a))
# node_coords = torch.tensor([
#     [0., 0],
#     [1, 0],
#     [0, 1]
# ]).to('cuda')

# B_matrix = TriangularElement.compute_B_matrix(node_coords)
# print(B_matrix)
