from abc import ABC, abstractmethod
import torch
from typing import Union, Optional, Tuple
from .function import Vec2Mat_Vogit, Mat2Vec_Vogit, sym3333_to_m66

class ConstitutiveLaw(ABC):
    """Base Class for material constitutive laws."""

    @abstractmethod
    def update_states(self):
        pass

    @abstractmethod
    def consistent_tangent(self):
        pass


    
class Elasticity_2D(ConstitutiveLaw):
    """
    Elastic Constitutive Model for 2D analysis, supporting plane strain and plane stress conditions.
    
    This model calculates the stiffness matrix based on the Young's modulus and Poisson's ratio 
    for either plane strain or plane stress conditions in two-dimensional analysis.

    Parameters:
    E (float or torch.Tensor): Young's Modulus.
    mu (float or torch.Tensor): Poisson's ratio.
    is_plane_strain (bool, optional): Flag for plane strain analysis (default is True).
    device (str or torch.device, optional): Computational device (default is 'cuda').
    rho (float, optional): Material density in kg/m^3 (default is 8960 kg/m^3 for Copper).

    Attributes:
    stiffMat2d (torch.Tensor): Stiffness matrix for the 2D analysis.
    """

    def __init__(self, E: Union[float, torch.Tensor], 
                 mu: Union[float, torch.Tensor], 
                 is_plane_strain: bool = True, 
                 device: Union[str, torch.device] = 'cuda', 
                 rho: float = 8.96*1000) -> None:
        super().__init__()
        self.E = E
        self.mu = mu
        self.is_plane_strain = is_plane_strain
        self.rho = torch.tensor(rho, dtype=torch.float64, device=device)
        self.stiffMat2d = self._compute_stiffness_matrix(device)
        self.device = device

    def _compute_stiffness_matrix(self, device: Union[str, torch.device]) -> torch.Tensor:
        """
        Computes the stiffness matrix based on the current analysis conditions.

        Parameters:
        device (Union[str, torch.device]): The device on which the computations are performed.

        Returns:
        torch.Tensor: The computed stiffness matrix.
        """
        if self.is_plane_strain:
            return self.E / (1 + self.mu) / (1 - 2 * self.mu) * torch.tensor([
                [1 - self.mu, self.mu, 0],
                [self.mu, 1 - self.mu, 0],
                [0, 0, (1 - 2 * self.mu) / 2]
            ], dtype=torch.float64, device=device)
        else:
            return self.E / (1 - self.mu * self.mu) * torch.tensor([
                [1, self.mu, 0],
                [self.mu, 1, 0],
                [0, 0, (1 - self.mu) / 2]
            ], dtype=torch.float64, device=device)

    def update_states(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        """
        Updates the stress states based on the provided strain tensor.

        Parameters:
        F (torch.Tensor): Deformation gradient tensor.
        strain (torch.Tensor): Strain tensor.

        Returns:
        torch.Tensor: Updated stress tensor.
        """
        strainVec = Mat2Vec_Vogit(strain, device=self.device)
        return self.stiffMat2d @ strainVec

    def consistent_tangent(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        """
        Returns the consistent tangent operator for the current material state.

        Parameters:
        F (torch.Tensor): Deformation gradient tensor.
        strain (torch.Tensor): Strain tensor.

        Returns:
        torch.Tensor: Consistent tangent stiffness matrix.
        """
        return self.stiffMat2d
    

class Elasticity_3D(ConstitutiveLaw):
    """
    Elastic Constitutive Model for 3D analysis based on isotropic material properties.

    This class computes the stiffness matrix for an isotropic material in three-dimensional space
    using Young's modulus and Poisson's ratio.

    Parameters:
    E (float or torch.Tensor): Young's Modulus.
    mu (float or torch.Tensor): Poisson's ratio.
    device (str or torch.device, optional): Computational device (default is 'cuda').

    Attributes:
    stiffMat3d (torch.Tensor): Stiffness matrix for the 3D analysis.
    """

    def __init__(self, E: Union[float, torch.Tensor], 
                 mu: Union[float, torch.Tensor], 
                 device: Union[str, torch.device] = 'cuda') -> None:
        super().__init__()

        # Ensure E and mu are tensors
        E = torch.tensor(E, dtype=torch.float64, device=device) if not isinstance(E, torch.Tensor) else E
        mu = torch.tensor(mu, dtype=torch.float64, device=device) if not isinstance(mu, torch.Tensor) else mu

        # Compute Lame parameters
        lambda_ = (E * mu) / ((1 + mu) * (1 - 2 * mu))
        mu = E / (2 * (1 + mu))

        # Stiffness matrix for 3D isotropic material
        self.stiffMat3d = torch.tensor([
            [lambda_ + 2 * mu, lambda_, lambda_, 0, 0, 0],
            [lambda_, lambda_ + 2 * mu, lambda_, 0, 0, 0],
            [lambda_, lambda_, lambda_ + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ], dtype=torch.float64, device=device)
        self.device = device


    def update_states(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        """
        Updates the stress states based on the provided strain tensor.

        Parameters:
        F (torch.Tensor): Deformation gradient tensor.
        strain (torch.Tensor): Strain tensor.

        Returns:
        torch.Tensor: Updated stress tensor.
        """
        strainVec = Mat2Vec_Vogit(strain, device=self.device)
        return self.stiffMat3d @ strainVec

    def consistent_tangent(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        """
        Returns the consistent tangent operator for the current material state.

        Parameters:
        F (torch.Tensor): Deformation gradient tensor.
        strain (torch.Tensor): Strain tensor.

        Returns:
        torch.Tensor: Consistent tangent stiffness matrix.
        """
        return self.stiffMat3d
    


class Hypoelastic(ConstitutiveLaw):
    """
    Hypoelastic material model for 2D and 3D stress analysis.

    Parameters:
    - sigma_0 (float): Yield stress.
    - epsilon_0 (float): Reference strain.
    - n (float): Hardening coefficient.
    - mu (float): Poisson's ratio.
    - device (str, optional): Computation device (e.g., 'cuda'). Default is 'cuda'.

    Attributes:
    - E (float): Calculated Young's modulus based on the input parameters.
    """

    def __init__(self, sigma_0, epsilon_0, n, mu, device='cuda'):
        """
        Initializes the Hypoelastic model with given material properties.
        """
        self.E = n * sigma_0 / epsilon_0
        self.sigma_0 = sigma_0
        self.epsilon_0 = epsilon_0
        self.n = n
        self.mu = mu
        self.device = device
    
    def _extend_to_3d(self, tensor_2d):
        """
        Extends a 2D tensor to a 3D tensor for compatibility with 3D operations.
        """
        output = torch.zeros((3, 3), dtype=torch.float64, device=self.device)
        output[:2, :2] = tensor_2d
        return output


    def update_states(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        """
        Updates the stress states based on the provided strain tensor.

        Args:
        - F (torch.Tensor): Deformation gradient tensor.
        - strain (torch.Tensor): Strain tensor.

        Returns:
        torch.Tensor: Updated stress tensor. Shape (3, 1) for 2D analysis, (6, 1) for 3D.

        Input shape: (3, 3) or (2, 2)
        Output shape: (3, 1) or (6, 1)
        """
        if strain.shape[0] == 2:
            strain = self._extend_to_3d(strain)
            is2d = True
        else:
            is2d = False

        # compute e_ij
        identity_matrix = torch.eye(3, device=self.device)
        e_ij = strain - 1/3 * torch.einsum('kk,ij -> ij', strain, identity_matrix)

        # compute strain_e
        strain_e = torch.sqrt(2/3 * torch.einsum('ij,ij->',e_ij, e_ij))

        # compute S_ij
        if strain_e > 0:
            S_ij = 2/3 * self.sigma_e(strain_e) * e_ij / strain_e
        else:
            S_ij = torch.zeros((3, 3), dtype=torch.float64, device=self.device)


        sigma_kk = self.E / (1 - 2 * self.mu) * 1/3 * torch.einsum('kk->', strain)
        output = S_ij + sigma_kk * identity_matrix / 3
        outputVec = Mat2Vec_Vogit(output)

        return outputVec[[0, 1, 3]] if is2d else outputVec


    def consistent_tangent(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        """
        Returns the consistent tangent operator for the current material state.

        Args:
        - F (torch.Tensor): Deformation gradient tensor.
        - strain (torch.Tensor): Strain tensor.

        Returns:
        torch.Tensor: Consistent tangent stiffness matrix.

        Input shape: (3, 3) or (2, 2)
        Output shape: (6, 6) for 3D analysis, (3, 3) for 2D.
        """
        if strain.shape[0] == 2:
            strain = self._extend_to_3d(strain)
            is2d = True
        else:
            is2d = False

        # compute e_ij
        identity_matrix = torch.eye(3, device=self.device)
        e_ij = strain - 1/3 * torch.einsum('kk,ij -> ij', strain, identity_matrix)

        # compute strain_e
        strain_e = torch.sqrt(2/3 * torch.einsum('ij,ij->',e_ij, e_ij))

        # compute E_T
        E_T = self.E_tangent(strain_e=strain_e)
        E_S = self.sigma_e(strain_e=strain_e) / strain_e
        if strain_e > 0:
            output = 4/9 * (E_T - E_S) * torch.einsum('ij,kl->ijkl', e_ij, e_ij) + \
                     2/3 * E_S * (torch.einsum('ik,jl->ijkl', identity_matrix, identity_matrix) - \
                                  1/3 * torch.einsum('ij,kl->ijkl', identity_matrix, identity_matrix)) + \
                     self.E / (9 * (1 - 2 * self.mu)) * torch.einsum('ij,kl->ijkl', identity_matrix, identity_matrix)
        else:
            output = 2/3 * E_T * (torch.einsum('ik,jl->ijkl', identity_matrix, identity_matrix) - \
                                  1/3 * torch.einsum('ij,kl->ijkl', identity_matrix, identity_matrix)) + \
                     self.E / (9 * (1 - 2 * self.mu)) * torch.einsum('ij,kl->ijkl', identity_matrix, identity_matrix)



        m3333toIMP66 = torch.tensor([[0,1,2,0,0,1], 
				 				 	 [0,1,2,1,2,2]])
        output_m66 = sym3333_to_m66(output, notationMap=m3333toIMP66, symmetrise=True)

        
        return output_m66[[0, 1, 3], :][:, [0, 1, 3]] if is2d else output_m66


    def E_tangent(self, strain_e):
        if strain_e <= self.epsilon_0:
            numerator = self.sigma_0 * (self.n / (self.n - 1) - strain_e / self.epsilon_0)
            denominator = self.epsilon_0 * torch.sqrt((1 + self.n ** 2) / (self.n - 1) ** 2 - (self.n / (self.n - 1) - strain_e / self.epsilon_0) ** 2)
            return numerator / denominator
        else:
            return self.sigma_0 * (strain_e / self.epsilon_0) ** (1 / self.n)
        
    def sigma_e(self, strain_e):
        if strain_e <= self.epsilon_0:
            A = (1 + self.n ** 2) / (self.n - 1) ** 2
            B = (self.n / (self.n - 1) - strain_e / self.epsilon_0) ** 2
            return self.sigma_0 * (torch.sqrt(A - B) - 1 / (self.n - 1))
        else:
            return self.sigma_0 / (self.n * strain_e) * (strain_e / self.epsilon_0) ** (1 / self.n)
        
class neoHookean(ConstitutiveLaw):

    def __init__(self, C10, C01, D1, NSHR, NDI, device='cuda'):
        """
        Initializes the Hypoelastic model with given material properties.
        """
        self.C10 = C10
        self.C01 = C01
        self.D1 = D1
        self.device = device
        self.NSHR = NSHR
        self.NDI = NDI

    def update_states(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        """
        Updates the stress states given the deformation gradient tensor F.
        """
        DET = torch.det(F)
        scale = DET**(-1/3)

        F_bar = scale*F

        # CALCULATE LEFT CAUCHY-GREEN TENSOR
        B_bar = torch.matmul(F_bar, F_bar.T) 

        # Initialize BBAR tensor
        BBAR = torch.zeros(6)
        BBAR[0] = F_bar[0, 0]**2 + F_bar[0, 1]**2 + F_bar[0, 2]**2
        BBAR[1] = F_bar[1, 0]**2 + F_bar[1, 1]**2 + F_bar[1, 2]**2
        BBAR[2] = F_bar[2, 2]**2 + F_bar[2, 0]**2 + F_bar[2, 1]**2
        BBAR[3] = F_bar[0, 0]*F_bar[1, 0] + F_bar[0, 1]*F_bar[1, 1] + F_bar[0, 2]*F_bar[1, 2]
        
                  
        if self.NSHR == 3:
            BBAR[4] = F_bar[0, 0]*F_bar[2, 0] + F_bar[0, 1]*F_bar[2, 1] + F_bar[0, 2]*F_bar[2, 2]
            BBAR[5] = F_bar[1, 0]*F_bar[2, 0] + F_bar[1, 1]*F_bar[2, 1] + F_bar[1, 2]*F_bar[2, 2]

        # Calculate stress tensor
        TRBBAR = (BBAR[0] + BBAR[1] + BBAR[2]) / 3
        EG = 2 * self.C10 / DET
        EK = 2 / self.D1 * (2 * DET - 1)
        PR = 2 / self.D1 * (DET - 1)

        STRESS = torch.zeros(self.NDI + self.NSHR)
        for K1 in range(self.NDI):
            STRESS[K1] = EG * (BBAR[K1] - TRBBAR) + PR
        for K1 in range(self.NDI, self.NDI + self.NSHR):
            STRESS[K1] = EG * BBAR[K1]

        # [S11, S22, S33, S12, S23, S13] <-- [S11, S22, S33, S23, S13, S12]
        if self.NDI + self.NSHR == 6:
            STRESS_output = torch.tensor([STRESS[0], STRESS[1], STRESS[2], STRESS[5], STRESS[3], STRESS[4]], device=self.device)
        else:
            STRESS_output = torch.tensor([STRESS[0], STRESS[1], STRESS[2], 0, 0,])
        return STRESS_output

    def consistent_tangent(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        DET = torch.det(F)
        scale = DET**(-1/3)

        F_bar = scale*F

        # CALCULATE LEFT CAUCHY-GREEN TENSOR
        B_bar = torch.matmul(F_bar, F_bar.T) 

        # Initialize BBAR tensor
        BBAR = torch.zeros(6)
        BBAR[0] = F_bar[0, 0]**2 + F_bar[0, 1]**2 + F_bar[0, 2]**2
        BBAR[1] = F_bar[1, 0]**2 + F_bar[1, 1]**2 + F_bar[1, 2]**2
        BBAR[2] = F_bar[2, 2]**2 + F_bar[2, 0]**2 + F_bar[2, 1]**2
        BBAR[3] = F_bar[0, 0]*F_bar[1, 0] + F_bar[0, 1]*F_bar[1, 1] + F_bar[0, 2]*F_bar[1, 2]
        
                  
        if self.NSHR == 3:
            BBAR[4] = F_bar[0, 0]*F_bar[2, 0] + F_bar[0, 1]*F_bar[2, 1] + F_bar[0, 2]*F_bar[2, 2]
            BBAR[5] = F_bar[1, 0]*F_bar[2, 0] + F_bar[1, 1]*F_bar[2, 1] + F_bar[1, 2]*F_bar[2, 2]

        # Calculate stress tensor
        TRBBAR = (BBAR[0] + BBAR[1] + BBAR[2]) / 3
        EG = 2 * self.C10 / DET
        EK = 2 / self.D1 * (2 * DET - 1)
        PR = 2 / self.D1 * (DET - 1)


        # Calculate stiffness matrix
        EG23 = EG * 2 / 3
        DDSDDE = torch.zeros((self.NDI + self.NSHR, self.NDI + self.NSHR))

        DDSDDE[0, 0] = EG23 * (BBAR[0] + TRBBAR) + EK
        DDSDDE[1, 1] = EG23 * (BBAR[1] + TRBBAR) + EK
        DDSDDE[2, 2] = EG23 * (BBAR[2] + TRBBAR) + EK
        DDSDDE[0, 1] = -EG23 * (BBAR[0] + BBAR[1] - TRBBAR) + EK
        DDSDDE[0, 2] = -EG23 * (BBAR[0] + BBAR[2] - TRBBAR) + EK
        DDSDDE[1, 2] = -EG23 * (BBAR[1] + BBAR[2] - TRBBAR) + EK
        DDSDDE[0, 3] = EG23 * BBAR[3] / 2
        DDSDDE[1, 3] = EG23 * BBAR[3] / 2
        DDSDDE[2, 3] = -EG23 * BBAR[3]
        DDSDDE[3, 3] = EG * (BBAR[0] + BBAR[1]) / 2

        if self.NSHR == 3:
            DDSDDE[0, 4] = EG23 * BBAR[4] / 2
            DDSDDE[1, 4] = -EG23 * BBAR[4]
            DDSDDE[2, 4] = EG23 * BBAR[4] / 2
            DDSDDE[0, 5] = -EG23 * BBAR[5]
            DDSDDE[1, 5] = EG23 * BBAR[5] / 2
            DDSDDE[2, 5] = EG23 * BBAR[5] / 2
            DDSDDE[4, 4] = EG * (BBAR[0] + BBAR[2]) / 2
            DDSDDE[5, 5] = EG * (BBAR[1] + BBAR[2]) / 2
            DDSDDE[3, 4] = EG * BBAR[5] / 2
            DDSDDE[3, 5] = EG * BBAR[4] / 2
            DDSDDE[4, 5] = EG * BBAR[3] / 2

        # Ensuring symmetry
        for K1 in range(self.NDI + self.NSHR):
            for K2 in range(K1):
                DDSDDE[K1, K2] = DDSDDE[K2, K1]

        # [S11, S22, S33, S12, S23, S13] <-- [S11, S22, S33, S23, S13, S12]
        if self.NDI + self.NSHR == 6:
            DDSDDEoutput = torch.tensor([
                [DDSDDE[0,0], DDSDDE[0,1], DDSDDE[0,2], DDSDDE[0,5], DDSDDE[0,3], DDSDDE[0,4]],
                [DDSDDE[1,0], DDSDDE[1,1], DDSDDE[1,2], DDSDDE[1,5], DDSDDE[1,3], DDSDDE[1,4]],
                [DDSDDE[2,0], DDSDDE[2,1], DDSDDE[2,2], DDSDDE[2,5], DDSDDE[2,3], DDSDDE[2,4]],
                [DDSDDE[5,0], DDSDDE[5,1], DDSDDE[5,2], DDSDDE[5,5], DDSDDE[5,3], DDSDDE[5,4]],
                [DDSDDE[3,0], DDSDDE[3,1], DDSDDE[3,2], DDSDDE[3,5], DDSDDE[3,3], DDSDDE[3,4]],
                [DDSDDE[4,0], DDSDDE[4,1], DDSDDE[4,2], DDSDDE[4,5], DDSDDE[4,3], DDSDDE[4,4]],
            ])

        return DDSDDEoutput


class Mooney:
    def __init__(self, A10, A01, K, device='cuda'):
        """
        初始化 Mooney-Rivlin 材料模型。
        :param A10, A01, K: 材料常數。
        :param device: 運算設備（例如 'cuda' 或 'cpu'）。
        """
        self.A10 = A10
        self.A01 = A01
        self.K = K
        self.device = device

    def update_states(self, F: torch.Tensor, strain: torch.Tensor):
        '''Stress = 2nd PK stress [S11, S22, S33, S12, S23, S13]'''
        C = torch.mm(F.t(), F)
        I1 = C.trace()
        I2 = C[0, 0] * C[1, 1] + C[0, 0] * C[2, 2] + C[1, 1] * C[2, 2] - C[0, 1]**2 - C[1, 2]**2 - C[0, 2]**2
        I3 = torch.det(C)
        J = torch.sqrt(I3)
        J3M1 = J - 1

        W1 = I3**(-1/3)
        W2 = (1/3) * I1 * I3**(-4/3)
        W3 = I3**(-2/3)
        W4 = (2/3) * I2 * I3**(-5/3)
        W5 = (1/2) * I3**(-1/2)

        I1E = self._I1E(C)
        I2E = self._I2E(C)
        I3E = self._I3E(C)

        J1E = W1 * I1E - W2 * I3E
        J2E = W3 * I2E - W4 * I3E
        J3E = W5 * I3E

        Stress = self.A10 * J1E + self.A01 * J2E + self.K * J3M1 * J3E
        return Stress.view(6, 1)


    def consistent_tangent(self, F: torch.Tensor, strain: torch.Tensor):
        C = torch.mm(F.t(), F)
        I1 = C.trace()
        I2 = C[0, 0] * C[1, 1] + C[0, 0] * C[2, 2] + C[1, 1] * C[2, 2] - C[0, 1]**2 - C[1, 2]**2 - C[0, 2]**2
        I3 = torch.det(C)
        J = torch.sqrt(I3)
        J3M1 = J - 1

        I1E = self._I1E(C)
        I2E = self._I2E(C)
        I3E = self._I3E(C)

        X12 = 1/2; X13 = 1/3; X23 = 2/3; X43 = 4/3; X53 = 5/3; X89 = 8/9

        W1 = I3**(-1/3)
        W2 = (1/3) * I1 * I3**(-4/3)
        W3 = I3**(-2/3)
        W4 = (2/3) * I2 * I3**(-5/3)
        W5 = (1/2) * I3**(-1/2)


        J1E = W1 * I1E - W2 * I3E
        J2E = W3 * I2E - W4 * I3E
        J3E = W5 * I3E

        
        I2EE = self._I2EE()
        I3EE = self._I3EE(C)

        W1 = X23*I3**(-X12)
        W2 = X89*I1*I3**(-X43)
        W3 = X13*I1*I3**(-X43)
        W4 = X43*I3**(-X12)
        W5 = X89*I2*I3**(-X53)
        W6 = I3**(-X23)
        W7 = X23 * I2 * I3**(-X53)
        W8 = I3**(-X12)
        W9 = X12 * I3**(-X12)

        J1EE = -W1 * (torch.ger(J1E, J3E) + torch.ger(J3E, J1E)) + W2 * torch.ger(J3E, J3E) - W3 * I3EE
        J2EE = -W4 * (torch.ger(J2E, J3E) + torch.ger(J3E, J2E)) + W5 * torch.ger(J3E, J3E) + W6 * I2EE - W7 * I3EE
        J3EE = -W8 * torch.ger(J3E, J3E) + W9 * I3EE

        D = self.A10 * J1EE + self.A01 * J2EE + self.K * (torch.ger(J3E, J3E) + J3M1 * J3EE)
        return D

    # 輔助方法計算 I1E, I2E, 和 I3E
    def _I1E(self, C):
        return 2 * torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float64, device=self.device)

    def _I2E(self, C):
        C1, C2, C3, C4, C5, C6 = C[0, 0], C[1, 1], C[2, 2], C[0, 1], C[1, 2], C[0, 2]
        return 2 * torch.tensor([C2 + C3, C3 + C1, C1 + C2, -C4, -C5, -C6], dtype=torch.float64, device=self.device)

    def _I3E(self, C):
        C1, C2, C3, C4, C5, C6 = C[0, 0], C[1, 1], C[2, 2], C[0, 1], C[1, 2], C[0, 2]
        return 2 * torch.tensor([C2*C3 - C5**2, C3*C1 - C6**2, C1*C2 - C4**2, C5*C6 - C3*C4, C6*C4 - C1*C5, C4*C5 - C2*C6], dtype=torch.float64, device=self.device)
    
    def _I2EE(self):
        return torch.tensor([
            [0, 4, 4, 0, 0, 0],
            [4, 0, 4, 0, 0, 0],
            [4, 4, 0, 0, 0, 0],
            [0, 0, 0, -2, 0, 0],
            [0, 0, 0, 0, -2, 0],
            [0, 0, 0, 0, 0, -2]
        ], dtype=torch.float64, device=self.device)
    
    def _I3EE(self, C):
        C1, C2, C3, C4, C5, C6 = C[0, 0], C[1, 1], C[2, 2], C[0, 1], C[1, 2], C[0, 2]
        return torch.tensor([
            [0,     4 * C3,  4 * C2,  0,    -4 * C5,  0],
            [4 * C3,  0,     4 * C1,  0,     0,    -4 * C6],
            [4 * C2,  4 * C1,  0,    -4 * C4,  0,     0],
            [0,     0,    -4 * C4, -2 * C3,  2 * C6,  2 * C5],
            [-4 * C5,  0,     0,     2 * C6, -2 * C1,  2 * C4],
            [0,    -4 * C6,  0,     2 * C5,  2 * C4, -2 * C2]
        ], dtype=torch.float64, device=self.device)

    
from matadi.math import det, dot, eigvals, eye, log, sqrt, sum1, sym, trace, transpose
from matadi import Variable, Material
from matadi import MaterialHyperelastic
import numpy as np
class neoHookean_matadi(ConstitutiveLaw):

    def __init__(self, C10, C01, K, NSHR, NDI, device='cuda'):
        """
        Initializes the Hypoelastic model with given material properties.
        """
        self.C10 = C10
        self.C01 = C01
        self.K = K
        self.device = device
        self.NSHR = NSHR
        self.NDI = NDI
        kwargs={'C10': C10, 'C01': C01, 'K':K}
        self.Mat = MaterialHyperelastic(fun=neoHookean_matadi.strain_energy,  **kwargs)

    @staticmethod
    def strain_energy(F, C10, C01, K):
        
        J = det(F)

        C = transpose(F) @ F
        B = F @ transpose(F)
        I1 = trace(B) / ( J**(2/3) )
        I2 = (I1 ** 2 - trace(B @ B)/( J**(4/3)) ) / 2
    
        
        return C10 * (I1 - 3) + C01 * (I2 - 3) + K/2 * (J - 1) ** 2


    def update_states(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        """
        Updates the stress states given the deformation gradient tensor F.
        """
        F = np.array(F)
        P = self.Mat.gradient([F])[0]
        S = np.linalg.inv(F) @ P

        # [S11, S22, S33, S12, S23, S13] <-- [S11, S22, S33, S23, S13, S12]
        output = torch.tensor([S[0,0],S[1,1],S[2,2],S[0,1],S[1,2],S[0,2]], dtype=torch.float64)

        return output

    def consistent_tangent(self, F: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        F = np.array(F)
        E = 1/2* (F @ F.T -np.eye(3))
        A = self.Mat.hessian([F])[0]

        # compute dSdE <-- A  (8.60) C_{pjlq} = A_{ijkl} - S_{lj} δ_{ik}
        delta = np.eye(F.shape[0])

        A = self.Mat.hessian([F])[0]
        P = self.Mat.gradient([F])[0]
        S = np.linalg.inv(F) @ P
        temp1 = A  - np.einsum('lj, ik->ijkl', S, delta)
        C = np.einsum('ijkl,pi,qk->pjlq',temp1, np.linalg.inv(F), np.linalg.inv(F))

        C66 = torch.tensor([
            [C[0,0,0,0], C[0,0,1,1], C[0,0,2,2], C[0,0,0,1], C[0,0,1,2], C[0,0,0,2]],
            [C[1,1,0,0], C[1,1,1,1], C[1,1,2,2], C[1,1,0,1], C[1,1,1,2], C[1,1,0,2]],
            [C[2,2,0,0], C[2,2,1,1], C[2,2,2,2], C[2,2,0,1], C[2,2,1,2], C[2,2,0,2]],
            [C[0,1,0,0], C[0,1,1,1], C[0,1,2,2], C[0,1,0,1], C[0,1,1,2], C[0,1,0,2]],
            [C[1,2,0,0], C[1,2,1,1], C[1,2,2,2], C[1,2,0,1], C[1,2,1,2], C[1,2,0,2]],
            [C[0,2,0,0], C[0,2,1,1], C[0,2,2,2], C[0,2,0,1], C[0,2,1,2], C[0,2,0,2]],
        ], dtype=torch.float64)
        

        # [S11, S22, S33, S12, S23, S13] <-- [S11, S22, S33, S23, S13, S12]

        return C66