from abc import ABC, abstractmethod
import torch
from typing import Union

class ConstitutiveLaw(ABC):
    """Base Class for material constitutive laws."""

    @abstractmethod
    def update_states(self):
        pass

    @abstractmethod
    def consistent_tangent():
        pass


class Material:
    r"""Returns tangent stiffness or compliance matrix.
    """

    def __init__(self, option='stiffness'):
        self.option = option

    def isotropic3D(self, E, v) -> torch.Tensor:
        r"""Elastic tangent stiffness matrix.
        E: elastic modulus
        v: poisson's ratio
        """
        lmbd = E*v/(1+v)/(1-2*v)
        mu = E/2/(1+v)

        # [11, 22, 33, 12, t3, 13]
        C = torch.tensor(
            [
                [lmbd+2*mu, lmbd, lmbd, 0, 0, 0],
                [lmbd, lmbd+2*mu, lmbd, 0, 0, 0],
                [lmbd, lmbd, lmbd+2*mu, 0, 0, 0],
                [0, 0, 0, 2*mu, 0, 0],
                [0, 0, 0, 0, 2*mu,0],
                [0, 0, 0, 0, 0, 2*mu]
            ]
        )

        if self.option == 'stiffness':
            return C
        elif self.option == 'compliance':
            return torch.linalg.inv(C)




class Elasticity(ConstitutiveLaw):
    r"""Elastic Constitutive Model.
    """
    
    def __init__(self, elasStiff):
        self.stiffMat3d = elasStiff # 3D elastic stiffness matrix

    def update_states(self, deps, stressN, alphaN, epN):
        r""" Update internal variables.
        deps: incremental strain
        stressN: stress from previous load step
                [s11, s22, s33, t12, t23, t13]
        alphaN: back-stress from previous load step
                [a11, a22, a33, a12, a23, a13]
        epN: effective plastic strain from previous load step
        """

        stress = stressN + self.stiffMat3d @ deps
        return (stress, alphaN, epN)

    def consistent_tangent(self, deps=None, stressN=None, alphaN=None, epN=None):
        r""" Return consistent tangent.
        """
        return self.stiffMat3d
    
class Elasticity_2D(ConstitutiveLaw):
    r"""
    Elastic Constitutive Model for 2D analysis with an option for plane strain.

    Parameters:
    - E (float or torch.Tensor): Young's Modulus.
    - mu (float or torch.Tensor): Poisson's ratio.
    - is_plane_strain (bool, optional): Flag to determine if the analysis is for plane strain. Defaults to True.

    Attributes:
    - stiffMat2d (torch.Tensor): Stiffness matrix for the 2D analysis.
    """

    def __init__(self, E, mu, is_plane_strain=True, device='cuda', rho=8.96*1000):
        self.is_plane_strain = is_plane_strain
        self.rho = torch.tensor(rho, dtype=torch.float64, device=device)
        # Depending on the condition, adjust the stiffness matrix accordingly
        if self.is_plane_strain:
            # Stiffness matrix for plane strain condition
            self.stiffMat2d = E/(1+mu)/(1-2*mu) * torch.tensor([
                [1-mu, mu, 0],
                [mu, 1-mu, 0],
                [0, 0, (1-2*mu)/2]
            ], dtype=torch.float64, device=device)
        else:
            # Stiffness matrix for plane stress condition
            self.stiffMat2d = E/(1-mu*mu) * torch.tensor([
                [1, mu, 0],
                [mu, 1, 0],
                [0, 0, (1-mu)/2]
            ], dtype=torch.float64, device=device)


    def update_states(self, deps, stressN, alphaN, epN):
        r"""
        Update internal variables based on the provided incremental strain.

        Parameters:
        - deps (torch.Tensor): Incremental strain.
        - stressN (torch.Tensor): Stress from the previous load step.
        - alphaN (torch.Tensor): Back-stress from the previous load step.
        - epN (float or torch.Tensor): Effective plastic strain from the previous load step.

        Returns:
        - tuple: Updated stress, back-stress, and effective plastic strain.
        """
        stress = stressN + self.stiffMat2d @ deps
        return (stress, alphaN, epN)

    def consistent_tangent(self, element_index = None, deps=None, stressN=None, alphaN=None, epN=None):
        r"""
        Return consistent tangent (stiffness matrix) for the current state.

        Parameters:
        - deps (torch.Tensor, optional): Incremental strain. Defaults to None.
        - stressN (torch.Tensor, optional): Stress from the previous load step. Defaults to None.
        - alphaN (torch.Tensor, optional): Back-stress from the previous load step. Defaults to None.
        - epN (float or torch.Tensor, optional): Effective plastic strain from the previous load step. Defaults to None.

        Returns:
        - torch.Tensor: Consistent tangent (stiffness matrix).
        """
        return self.stiffMat2d
    
    def to_device(self, device: Union[str, torch.device]) -> None:
        """
        Move the material data to a specified device.
        
        Parameters:
        - device (Union[str, torch.device]): The target device, e.g., "cuda" or "cpu".
        """
        self.stiffMat2d = self.stiffMat2d.to(device)

class Elasticity_3D(ConstitutiveLaw):
    r"""
    Elastic Constitutive Model for 2D analysis with an option for plane strain.

    Parameters:
    - E (float or torch.Tensor): Young's Modulus.
    - mu (float or torch.Tensor): Poisson's ratio.
    - is_plane_strain (bool, optional): Flag to determine if the analysis is for plane strain. Defaults to True.

    Attributes:
    - stiffMat2d (torch.Tensor): Stiffness matrix for the 2D analysis.
    """

    def __init__(self, E, mu, device='cuda'):

        # Convert E and mu to tensors if they are not already
        if not isinstance(E, torch.Tensor):
            E = torch.tensor(E, dtype=torch.float64, device=device)
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float64, device=device)

        # Compute Lame parameters
        lambda_ = (E * mu) / ((1 + mu) * (1 - 2 * mu))
        mu = E / (2 * (1 + mu))
                
        # Create stiffness matrix for 3D isotropic material
        self.stiffMat3d = torch.tensor([
            [lambda_ + 2 * mu, lambda_, lambda_, 0, 0, 0],
            [lambda_, lambda_ + 2 * mu, lambda_, 0, 0, 0],
            [lambda_, lambda_, lambda_ + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ], dtype=torch.float64, device=device)



    def update_states(self, deps, stressN, alphaN, epN):
        r"""
        Update internal variables based on the provided incremental strain.

        Parameters:
        - deps (torch.Tensor): Incremental strain.
        - stressN (torch.Tensor): Stress from the previous load step.
        - alphaN (torch.Tensor): Back-stress from the previous load step.
        - epN (float or torch.Tensor): Effective plastic strain from the previous load step.

        Returns:
        - tuple: Updated stress, back-stress, and effective plastic strain.
        """
        stress = stressN + self.stiffMat3d @ deps
        return (stress, alphaN, epN)

    def consistent_tangent(self, element_index = None, deps=None, stressN=None, alphaN=None, epN=None):
        r"""
        Return consistent tangent (stiffness matrix) for the current state.

        Parameters:
        - deps (torch.Tensor, optional): Incremental strain. Defaults to None.
        - stressN (torch.Tensor, optional): Stress from the previous load step. Defaults to None.
        - alphaN (torch.Tensor, optional): Back-stress from the previous load step. Defaults to None.
        - epN (float or torch.Tensor, optional): Effective plastic strain from the previous load step. Defaults to None.

        Returns:
        - torch.Tensor: Consistent tangent (stiffness matrix).
        """
        return self.stiffMat3d

