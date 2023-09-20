import torch

class Element:
    def __init__(self, node_indices):
        # Store node indices as a torch tensor
        self.node_indices = torch.tensor(node_indices)



class TriangularElement(Element):
    def __init__(self, node_indices):
        super().__init__(node_indices)
    # Specific methods for triangular elements can be added here