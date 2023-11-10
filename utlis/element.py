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
        N = torch.tensor([xi, eta, 1 - xi - eta], dtype=torch.float64).to(device)
        return N
    
    @staticmethod
    def shape_function_derivatives(device='cuda') -> torch.Tensor:
        ''' 2*3 tensor
        [dN1/d_xi, dN2/d_xi, dN3/d_xi],
        [dN1/d_eta, dN2/d_eta, dN3/d_eta],
        
        '''
        # Derivatives of N with respect to xi and eta
        dN_dxi = torch.tensor([1, 0, -1], dtype=torch.float64, device=device)
        dN_deta = torch.tensor([0, 1, -1], dtype=torch.float64, device=device)
        
        # Combine them into a single tensor for convenience
        dN = torch.stack((dN_dxi, dN_deta), dim=0)
        
        return dN

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
        ], dtype=torch.float64).to(device)

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
                        [dN1_dy, dN1_dx, dN2_dy, dN2_dx, dN3_dy, dN3_dx]], device=device, dtype=torch.float64)

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
    def face_gauss_points_and_weights(face_idx: int, node_coords: torch.Tensor) -> Tuple[List[Tuple[float]], List[float]]:
        """Return the Gauss points and weights for a face (edge) of the element."""
        # Compute the half length of the edge
        node1, node2 = node_coords
        half_length = 0.5 * torch.norm(node2 - node1).item()
        # For 2D edge integration, using 2 Gauss points as example
        return [(-1/np.sqrt(3),), (1/np.sqrt(3),)], [half_length, half_length]

    
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





class T3Element(TriangularElement):
    element_dof = 6  # 2 DOFs (u, v) for each of the 3 nodes
    node_per_element = 3  # Number of nodes for T3 element
    dimension = 2  # 2D element

    @staticmethod
    def shape_function_derivatives(natural_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        xi, eta = natural_coords
        dN_dxi = torch.tensor([
            [-1, 1, 0],
            [-1, 0, 1]
        ], dtype=torch.float64, device=device)
        
        return dN_dxi
    
    @staticmethod
    def shape_functions(natural_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        xi, eta = natural_coords
        N = torch.tensor([
            1 - xi - eta,
            xi,
            eta
        ], dtype=torch.float64, device=device)
        return N


    @staticmethod
    def jacobian(node_coords, dN_dxi, device):
        J = torch.mm(dN_dxi, node_coords)
        detJ = torch.det(J)
        if detJ == 0:
            raise ValueError("The Jacobian is singular")
        return J

    @staticmethod
    def compute_B_matrix(dN_dxi, J, device):
        inv_J = torch.inverse(J)
        dN_dxy = torch.mm(inv_J, dN_dxi)
        
        B = torch.zeros(3, 6, dtype=torch.float64, device=device)  # 3x6 for plane stress or plane strain
        B[0, 0::2] = dN_dxy[0, :]
        B[1, 1::2] = dN_dxy[1, :]
        B[2, 0::2] = dN_dxy[1, :]
        B[2, 1::2] = dN_dxy[0, :]
        
        return B
    
class QuadElement(BaseElement):
    element_dof = 8  # 2 DOFs (u, v) for each of the 4 nodes
    node_per_element = 4  # Number of nodes for Q4 element
    dimension = 2  # 2D element

    @staticmethod
    def shape_functions(natural_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        xi, eta = natural_coords
        N = torch.tensor([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ], dtype=torch.float64).to(device)
        
        return N
    
    @staticmethod
    def shape_function_derivatives(natural_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        xi, eta = natural_coords
        dN_dxi = torch.tensor([
            [-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)],
            [-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)]
        ], dtype=torch.float64).to(device)
        
        return dN_dxi
    
    @staticmethod
    def jacobian(node_coords: torch.Tensor, dN_dxi: torch.Tensor, device='cuda') -> torch.Tensor:
        J = torch.mm(dN_dxi, node_coords)
        detJ = torch.det(J)
        if detJ == 0:
            raise ValueError("The Jacobian is singular")
        return J
    
    @staticmethod
    def compute_B_matrix(dN_dxi: torch.Tensor, J: torch.Tensor, device='cuda') -> torch.Tensor:
        inv_J = torch.inverse(J)
        dN_dxy = torch.mm(inv_J, dN_dxi)
        
        B = torch.zeros(3, 8, dtype=torch.float64, device=device)  # 3x8 for plane stress or plane strain
        B[0, 0::2] = dN_dxy[0, :]
        B[1, 1::2] = dN_dxy[1, :]
        B[2, 0::2] = dN_dxy[1, :]
        B[2, 1::2] = dN_dxy[0, :]
        
        return B


class Quad8Element(BaseElement):
    element_dof = 16  # 2 DOFs (u, v) for each of the 8 nodes
    node_per_element = 8  # Number of nodes for Q8 element
    dimension = 2  # 2D element

    @staticmethod
    def shape_functions(natural_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        xi, eta = natural_coords
        N = torch.tensor([
            0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1),
            0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1),
            0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1),
            0.25 * (1 - xi) * (1 + eta) * (eta - xi - 1),
            0.5 * (1 - xi ** 2) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta ** 2),
            0.5 * (1 - xi ** 2) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta ** 2)
        ], dtype=torch.float64).to(device)
        
        return N
    
    @staticmethod
    def shape_function_derivatives(natural_coor, device='cuda') -> torch.Tensor:
        xi, eta = natural_coor
        dN_dxi = torch.tensor([
            [-0.25*(eta - 1)*(eta + 2*xi), 0.25*(eta - 1)*(eta - 2*xi), 0.25*(eta + 1)*(eta + 2*xi),0.25*(-eta + 2*xi)*(eta + 1),1.0*xi*(eta - 1),0.5 - 0.5*eta**2,-1.0*xi*(eta + 1),0.5*eta**2 - 0.5],
            [-0.25*(2*eta + xi)*(xi - 1), 0.25*(2*eta - xi)*(xi + 1),0.25*(2*eta + xi)*(xi + 1),0.25*(-2*eta + xi)*(xi - 1),0.5*xi**2 - 0.5,-1.0*eta*(xi + 1),0.5 - 0.5*xi**2,eta*(xi - 1)]
        ], dtype=torch.float64).to(device)
        
        return dN_dxi

    @staticmethod
    def jacobian(node_coords: torch.Tensor, dN_dxi: torch.Tensor, device='cuda') -> torch.Tensor:
        J = torch.mm(dN_dxi, node_coords)
        detJ = torch.det(J)
        if detJ == 0:
            raise ValueError("The Jacobian is singular")
        return J

    @staticmethod
    def compute_B_matrix(dN_dxi: torch.Tensor, J: torch.Tensor, device='cuda') -> torch.Tensor:
        inv_J = torch.inverse(J)
        dN_dxy = torch.mm(inv_J, dN_dxi)
        
        B = torch.zeros(3, 16, dtype=torch.float64, device=device)  # 3x16 for plane stress or plane strain
        B[0, 0::2] = dN_dxy[0, :]
        B[1, 1::2] = dN_dxy[1, :]
        B[2, 0::2] = dN_dxy[1, :]
        B[2, 1::2] = dN_dxy[0, :]
        
        return B

class BrickElement(BaseElement):
    element_dof = 24  # 3 DOFs (u, v) for each of the 8 nodes
    node_per_element = 8  # Number of nodes for Q8 element
    dimension = 3  # 3D element

    @staticmethod
    def shape_functions(natural_coords: torch.Tensor, device='cuda') -> torch.Tensor:
        xi1, xi2, xi3 = natural_coords
        N = torch.tensor([
            1/8 * (1 - xi1) * (1 - xi2) * (1 - xi3),
            1/8 * (1 + xi1) * (1 - xi2) * (1 - xi3),
            1/8 * (1 + xi1) * (1 + xi2) * (1 - xi3),
            1/8 * (1 - xi1) * (1 + xi2) * (1 - xi3),
            1/8 * (1 - xi1) * (1 - xi2) * (1 + xi3),
            1/8 * (1 + xi1) * (1 - xi2) * (1 + xi3),
            1/8 * (1 + xi1) * (1 + xi2) * (1 + xi3),
            1/8 * (1 - xi1) * (1 + xi2) * (1 + xi3),
        ], dtype=torch.float64).to(device)
        
        return N
    
    @staticmethod
    def shape_function_derivatives(natural_coors, device='cuda') -> torch.Tensor:
        xi1, xi2, xi3 = natural_coors
        dN_dxi = torch.tensor([
            # Derivatives with respect to xi1 for each of the 8 shape functions
            [-0.125*(1 - xi2)*(1 - xi3), 0.125*(1 - xi2)*(1 - xi3), 0.125*(1 - xi3)*(xi2 + 1),-0.125*(1 - xi3)*(xi2 + 1),-0.125*(1 - xi2)*(xi3 + 1), 0.125*(1 - xi2)*(xi3 + 1), 0.125*(xi2 + 1)*(xi3 + 1),-0.125*(xi2 + 1)*(xi3 + 1)],
            [ (0.125 - 0.125*xi1)*(xi3 - 1), (0.125*xi1 + 0.125)*(xi3 - 1), (1 - xi3)*(0.125*xi1 + 0.125), (0.125 - 0.125*xi1)*(1 - xi3),(0.125 - 0.125*xi1)*(-xi3 - 1),(0.125*xi1 + 0.125)*(-xi3 - 1), (0.125*xi1 + 0.125)*(xi3 + 1), (0.125 - 0.125*xi1)*(xi3 + 1)],
            [ (0.125 - 0.125*xi1)*(xi2 - 1), (0.125*xi1 + 0.125)*(xi2 - 1),(0.125*xi1 + 0.125)*(-xi2 - 1),(0.125 - 0.125*xi1)*(-xi2 - 1), (0.125 - 0.125*xi1)*(1 - xi2), (1 - xi2)*(0.125*xi1 + 0.125), (0.125*xi1 + 0.125)*(xi2 + 1), (0.125 - 0.125*xi1)*(xi2 + 1),]
        ], dtype=torch.float64).to(device)
        return dN_dxi

    @staticmethod
    def jacobian(node_coords: torch.Tensor, dN_dxi: torch.Tensor, device='cuda') -> torch.Tensor:
        J = torch.mm(dN_dxi, node_coords)
        detJ = torch.det(J)
        if detJ == 0:
            raise ValueError("The Jacobian is singular")
        return J

    @staticmethod
    def compute_B_matrix(dN_dxi: torch.Tensor, J: torch.Tensor, device='cuda') -> torch.Tensor:
        inv_J = torch.inverse(J)
        dN_dxy = torch.mm(inv_J, dN_dxi)
        
        B = torch.zeros(6, 24, dtype=torch.float64, device=device)
        B[0, 0::3] = dN_dxy[0, :]
        B[1, 1::3] = dN_dxy[1, :]
        B[2, 2::3] = dN_dxy[2, :]

        # sigma_23
        B[3, 0::3] = 0
        B[3, 1::3] = dN_dxy[2, :]
        B[3, 2::3] = dN_dxy[1, :]

        # sigma_13
        B[4, 0::3] = dN_dxy[2, :]
        B[4, 1::3] = 0
        B[4, 2::3] = dN_dxy[0, :]

        # sigma_12
        B[5, 0::3] = dN_dxy[1, :]
        B[5, 1::3] = dN_dxy[0, :]
        B[5, 2::3] = 0
        return B