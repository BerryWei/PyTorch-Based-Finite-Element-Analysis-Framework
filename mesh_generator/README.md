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

### Example 1: Simple Polygon
```yaml
geometry:
  boundary:
    - [0.0, -2.0]
    - [0.0, -2.0]
    - [60.0, -3.0]
    - [60.0, 3.0]
    - [0.0, 2.0]
  holes:
    # if no holes--> leave empty
    # - center: [0.5, 0.5]
    #   radius: 0.3

mesh:
  area: 100

num-dim: 2
```
<div style="text-align: center">
    <img src="./Problem1/mesh_config.png" alt="Sample GIF" width="300" height="300">
</div>

### Example 3: Polygon with a Hole
```yaml
geometry:
  boundary:
    - [0.0, -30]
    - [200.0, -30]
    - [200.0, 30]
    - [0.0, 30.]
  holes:
    # if no holes--> leave empty
    - center: [100, 0]
      radius: 12

mesh:
  area: 50

num-dim: 2
```

<div style="text-align: center">
    <img src="./Problem3/mesh_config.png" alt="Sample GIF" width="300" height="300">
</div>

### Example 4: Irregular shape
```yaml
geometry:
  boundary:
  - [-100, -30.0]
  - [-100, 0.0]
  - [-100, 30.0]
  - [1.7763568394002505e-15, 30.0]
  - [-0.8660498191297741, 28.73930148776709]
  - [-1.3852060837090914, 27.300593803451388]
  - [-1.523732245478456, 25.77736908908176]
  - [-1.2726264010407817, 24.268611692132122]
  - [-0.648206266307632, 22.872365824349224]
  - [0.30895120444559065, 21.679364313812755]
  - [1.5366466477378982, 20.767132475692478]
  - [2.9551002180845165, 20.19495025233008]
  - [4.472135954999579, 20.0]
  - [100, 20.0]
  - [100, 0.0]
  - [100, -20.0]
  - [4.47213595499958, -20.0]
  - [2.955100218084515, -20.19495025233008]
  - [1.536646647737899, -20.767132475692478]
  - [0.3089512044455933, -21.679364313812755]
  - [-0.648206266307632, -22.87236582434922]
  - [-1.2726264010407817, -24.268611692132122]
  - [-1.523732245478456, -25.777369089081756]
  - [-1.3852060837090914, -27.300593803451388]
  - [-0.866049819129775, -28.739301487767086]
  - [0.0, -30.0]
  holes:
    # if no holes--> leave empty
    # - center: [100, 0]
    #   radius: 12

mesh:
  area: 500

num-dim: 2
```

<div style="text-align: center">
    <img src="./Problem4/mesh_config.png" alt="Sample GIF" width="300" height="300">
</div>

After defining the geometry in a YAML file, simply run:
```bash
python main.py
```

