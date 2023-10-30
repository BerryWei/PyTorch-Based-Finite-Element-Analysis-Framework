import torch

class GaussQuadrature:
    def __init__(self, nelnd: int, dimension: int):
        self.nelnd = nelnd
        self.dimension = dimension

    def get_points_and_weights(self):
        if self.dimension == 1:
            gauss_points = torch.tensor([[0]])
            weights = torch.tensor([2])
            return gauss_points, weights
        
        
        elif self.dimension == 2:
            if self.nelnd == 3: # M=1
                gauss_points = torch.tensor([[1/3, 1/3]])
                weights = torch.tensor([1/2])
                return gauss_points, weights
            
            elif self.nelnd == 6: # M=4
                gauss_points = torch.tensor([
                    [1/3, 1/3],
                    [0.6, 0.2],
                    [0.2, 0.6],
                    [0.2, 0.2]
                ])
                weights = torch.tensor([-27/96, 25/96, 25/96, 25/96])  # Assuming equal weights
                return gauss_points, weights
            
            elif self.nelnd == 4: # M=4
                gauss_points = torch.tensor([
                    [-0.577350269, -0.577350269],
                    [+0.577350269, -0.577350269],
                    [-0.577350269, +0.577350269],
                    [+0.577350269, +0.577350269],
                ])
                weights = torch.tensor([1, 1, 1, 1])  # Assuming equal weights
                return gauss_points, weights
            
            elif self.nelnd == 8: # M=9
                gauss_points = torch.tensor([
                    [-0.7745966692, -0.7745966692],
                    [0, -0.7745966692],
                    [0.7745966692, -0.7745966692],
                    [-0.7745966692, 0],
                    [0, 0],
                    [0.7745966692, 0],
                    [-0.7745966692, 0.7745966692],
                    [0, 0.7745966692],
                    [0.7745966692, 0.7745966692]
                ])
                weights = torch.tensor([
                    0.308641974,
                    0.493827159,
                    0.308641974,
                    0.493827159,
                    0.790123455,
                    0.493827159,
                    0.308641974,
                    0.493827159,
                    0.308641974
                ])
                return gauss_points, weights
            else:
                raise ValueError("number of element in 2D is not equal to 3,4,6,8")
        elif self.dimension == 3:
            if self.nelnd == 4:  # M=1
                gauss_points = torch.tensor([[1/4, 1/4, 1/4]])
                weights = torch.tensor([1/6])
                return gauss_points, weights
            
            elif self.nelnd == 10:  # M=4
                alpha = 0.58541020
                beta = 0.13819660
                gauss_points = torch.tensor([
                    [alpha, beta, beta],
                    [beta, alpha, beta],
                    [beta, beta, alpha],
                    [beta, beta, beta]
                ])
                weights = torch.tensor([1/24, 1/24, 1/24, 1/24])
                return gauss_points, weights
            
            elif self.nelnd == 8:  # M=8
                gauss_points = torch.tensor([
                    [-0.5773502692, -0.5773502692, -0.5773502692],
                    [0.5773502692, -0.5773502692, -0.5773502692],
                    [-0.5773502692, 0.5773502692, -0.5773502692],
                    [0.5773502692, 0.5773502692, -0.5773502692],
                    [-0.5773502692, -0.5773502692, 0.5773502692],
                    [0.5773502692, -0.5773502692, 0.5773502692],
                    [-0.5773502692, 0.5773502692, 0.5773502692],
                    [0.5773502692, 0.5773502692, 0.5773502692]
                ])
                weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
                return gauss_points, weights
            
            elif self.nelnd == 20:  # M=27
                gauss_points = torch.tensor([
                    [-0.7745966692, -0.7745966692, -0.7745966692],
                    [0, -0.7745966692, -0.7745966692],
                    [0.7745966692, -0.7745966692, -0.7745966692],
                    [-0.7745966692, 0, -0.7745966692],
                    [0, 0, -0.7745966692],
                    [0.7745966692, 0, -0.7745966692],
                    [-0.7745966692, 0.7745966692, -0.7745966692],
                    [0, 0.7745966692, -0.7745966692],
                    [0.7745966692, 0.7745966692, -0.7745966692],
                    [-0.7745966692, -0.7745966692, 0],
                    [0, -0.7745966692, 0],
                    [0.7745966692, -0.7745966692, 0],
                    [-0.7745966692, 0, 0],
                    [0, 0, 0],
                    [0.7745966692, 0, 0],
                    [-0.7745966692, 0.7745966692, 0],
                    [0, 0.7745966692, 0],
                    [0.7745966692, 0.7745966692, 0],
                    [-0.7745966692, -0.7745966692, 0.7745966692],
                    [0, -0.7745966692, 0.7745966692],
                    [0.7745966692, -0.7745966692, 0.7745966692],
                    [-0.7745966692, 0, 0.7745966692],
                    [0, 0, 0.7745966692],
                    [0.7745966692, 0, 0.7745966692],
                    [-0.7745966692, 0.7745966692, 0.7745966692],
                    [0, 0.7745966692, 0.7745966692],
                    [0.7745966692, 0.7745966692, 0.7745966692]
                ])
                weights = torch.tensor([
                    0.171467763, 0.274348421, 0.171467763,
                    0.274348421, 0.438957474, 0.274348421,
                    0.171467763, 0.274348421, 0.171467763,
                    0.274348421, 0.438957474, 0.274348421,
                    0.438957474, 0.702331959, 0.438957474,
                    0.274348421, 0.438957474, 0.274348421,
                    0.171467763, 0.274348421, 0.171467763,
                    0.274348421, 0.438957474, 0.274348421,
                    0.171467763, 0.274348421, 0.171467763
                ])
                return gauss_points, weights
            else:
                raise ValueError("Unsupported number of nodes for 3D elements")
        else:
            raise ValueError("Unsupported dimension")

