# Finite Element Method Mesh Generator

This repository provides a set of Python tools for generating meshes suitable for Finite Element Method (FEM) simulations. The primary objective is to facilitate the creation of polygonal meshes, with or without holes, and offer visualization capabilities.

## Features
- **Mesh Generation**: Create simple polygonal meshes or meshes with holes.
- **Visualization**: Visualize the generated mesh using `matplotlib`.
- **Export Options**: Save the mesh data in YAML and MATLAB formats.

## Mesh Construction

The framework utilizes the [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) library for mesh generation. Triangle is a robust implementation of the Delaunay triangulation and constrained Delaunay triangulation algorithms. 

To construct the mesh, boundary vertices are first defined, and segments are created between consecutive vertices. The `triangle.triangulate()` function is then invoked with specific meshing options to produce the desired mesh. 

For ensuring particular points, such as `(0,0)`, are included in the mesh vertices, they can be explicitly added to the boundary vertices before triangulation.

## Usage
### Setup
1. Clone the repository to your local machine.
2. Ensure you have the required Python libraries installed.
3. Define your geometry in a YAML file.

### Execution
Run the `main.py` script:
```bash
python main.py
```


## Directory Structure

- `utlis/`:
  - `Geometry.py`: Contains the `Geometry` class for loading boundary and hole data from a YAML file.
  - `MeshGenerator.py`: Contains the `MeshGenerator` class for generating the mesh.
  - `Visualizer.py`: Contains the `Visualizer` class for visualizing the generated mesh.
- `main.py`: The main entry point for mesh generation.

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
    <img src="./Problem1/mesh_config.png" alt="Sample GIF" width="600*1.4" height="480*1.4">
</div>

### Example 2: Complex Polygon
```yaml
geometry:
  boundary:
    - [0.0, 21]
    - [200, 21]
    - [200, 0]
    - [200, 31.]
    - [450, 31]
    - [450, 21]
    - [650, 21]
    - [650, -21]
    - [450, -21]
    - [450, 0]
    - [450, -31]
    - [200, -31]
    - [200, -21]
    - [0, -21]
    
  holes:
    # if no holes--> leave empty
    # - center: [0.5, 0.5]
    #   radius: 0.3

mesh:
  area: 1000

num-dim: 2
```

<div style="text-align: center">
    <img src="./Problem2/mesh_config.png" alt="Sample GIF" width="600*1.5" height="480*1.5">
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
    <img src="./Problem3/mesh_config.png" alt="Sample GIF" width="600*1.1" height="480*1.1">
</div>

### Example 4: Irregular shape
```yaml
geometry:
  boundary:
  - [-60, -30.0]
  - [-60, 30.0]
  - [0, 30.0]
  - [0.0, 26.0]
  - [0.09115348192675121, 24.95811093399842] 
  - [0.36184427528454943, 23.947879140045988]
  - [0.8038475772933671, 23.0]
  - [1.4037333412861317, 22.143274341880765] 
  - [2.143274341880763, 21.403733341286134]  
  - [2.9999999999999973, 20.803847577293368] 
  - [3.9478791400459885, 20.36184427528455]  
  - [4.958110933998418, 20.091153481926753]  
  - [5.999999999999999, 20.0]
  - [60, 20.0]
  - [60, -20.0]
  - [6.0, -20.0]
  - [4.958110933998418, -20.091153481926753] 
  - [3.9478791400459876, -20.36184427528455] 
  - [3.0000000000000013, -20.803847577293368]
  - [2.1432743418807636, -21.40373334128613] 
  - [1.4037333412861326, -22.143274341880762]
  - [0.8038475772933689, -23.0]
  - [0.3618442752845503, -23.947879140045988]
  - [0.0911534819267521, -24.95811093399842] 
  - [0.0, -26.0]
  - [0, -30.0]
  holes:
    # if no holes--> leave empty
    # - center: [100, 0]
    #   radius: 12

mesh:
  area: 50

num-dim: 2
```

<div style="text-align: center">
    <img src="./Problem4/mesh_config.png" alt="Sample GIF" width="600*1.2" height="480*1.2">
</div>



