# PyTorch-Based Finite Element Analysis Framework

Introducing a framework designed for Finite Element Analysis (FEA) with PyTorch at its core. Our solution is specifically engineered to exploit the computational power of contemporary GPUs, enabling high-speed FEA computations.

## Key Features

- **Configurability**: Define nodes, elements, boundary condition, and material properties seamlessly using YAML configuration files.
- **GPU Acceleration**: Integrated with PyTorch, our framework benefits from GPU acceleration, ensuring fast computations.
- **Extensibility**: Designed with modularity in mind, it's easy to expand upon the existing functionalities, be it adding new element types or material models.

## Modules Overview

- **`element.py`**: Centers around the `TriangularElement` class, specialized in triangular element computations.
- **`material.py`**: A hub for material properties, it provides the `Material` class for elasticity stiffness matrix calculations and the `Elasticity_2D` class for 2D elasticity modeling.
- **`fem_module.py`**: The framework's core, it integrates nodes, elements, and material properties via the `FiniteElementModel` class.
- **`main.py`**: The primary execution script. Users can customize runs via command-line arguments.

## Getting Started

1. **Setup**: Craft nodes, elements, and material properties using the provided YAML templates in `geometry.yaml` and `material.yaml`.
2. **Execution**:
```bash
   python main.py --device cuda --geometry_path geometry.yaml --material_path material.yaml --loading_path loading.yaml
```

## Prerequisites

- Python (version 3.8 or higher)
- Pytorch (version 2.0.1 or higher)
- CUDA Toolkit (version 11.8 or higher)

## Roadmap

- **Element Expansion**: Our upcoming releases aim to support an extended array of element types.
- **Material Models**: We are working towards introducing a diverse set of material models to cater to a broad spectrum of engineering challenges.

## Contact

Should you have any inquiries, suggestions, or feedback, please don't hesitate to reach out:
- **Email**: [berrya90239@gmail.com](mailto:berrya90239@gmail.com)
- **GitHub**: [BerryWei's GitHub](https://github.com/BerryWei)
