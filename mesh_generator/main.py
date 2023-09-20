# main.py
from pathlib import Path
from utlis import Geometry, MeshGenerator, Visualizer

if __name__ == '__main__':
    path = Path('D:/finite_element_method/mesh_generator/geometry_p1.yaml')
    geometry_obj = Geometry(path)
    mesh_gen = MeshGenerator(geometry_obj)
    if geometry_obj.holes:
        print("This geometry contains holes.")
    else:
        print("This geometry doesn't contain any holes.")

    if geometry_obj.holes:
        nodes, elements = mesh_gen.generate_mesh_with_hole()
    else:
        nodes, elements = mesh_gen.generate_simple_mesh()

    mesh_gen.write_output_file_yaml(filename='inp.yaml',nodes=nodes, elements=elements, ndim=geometry_obj.ndim)

    Visualizer.visualize_mesh(nodes, elements)
