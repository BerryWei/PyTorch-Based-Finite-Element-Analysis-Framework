# Finite Element Method Mesh Generator

This repository provides tools for generating meshes for use in Finite Element Method simulations. The code allows for the generation of simple meshes as well as meshes with holes.

## Features

- Generation of simple polygonal meshes.
- Generation of polygonal meshes with holes.
- Visualization of the generated mesh using `matplotlib`.
- Exporting the mesh data to YAML and MATLAB formats.

## Usage

1. Define your geometry in a YAML file.
2. Run the `main.py` script.

## Directory Structure

- `utlis/`:
  - `Geometry.py`: Contains the `Geometry` class for loading boundary and hole data from a YAML file.
  - `MeshGenerator.py`: Contains the `MeshGenerator` class for generating the mesh.
  - `Visualizer.py`: Contains the `Visualizer` class for visualizing the generated mesh.
- `main.py`: The main script to run the mesh generation.

## Example

### Example 1: Simple Rectangle
```yaml
geometry:
  boundary:
    - [0.0, 0.0]
    - [1.0, 0.0]
    - [1.0, 1.0]
    - [0.0, 1.0]
  holes: []

mesh:
  area: 1

num-dim: 2
```
### Example 2: Rectangle with a Hole
```yaml
geometry:
  boundary:
    - [0.0, 0.0]
    - [2.0, 0.0]
    - [2.0, 2.0]
    - [0.0, 2.0]
  holes:
    - center: [1.0, 1.0]
      radius: 0.5

mesh:
  area: 2

num-dim: 2
```

After defining the geometry in a YAML file, simply run:
```bash
python main.py
```

