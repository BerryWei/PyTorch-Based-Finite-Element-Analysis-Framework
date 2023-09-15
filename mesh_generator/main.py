# main.py

from utlis import Geometry, MeshGenerator, Visualizer

if __name__ == '__main__':
    geometry_obj = Geometry('geometry.yaml')
    mesh_gen = MeshGenerator(geometry_obj)
    if geometry_obj.holes:
        print("This geometry contains holes.")
    else:
        print("This geometry doesn't contain any holes.")

    if geometry_obj.holes:
        nodes, elements = mesh_gen.generate_mesh_with_hole()
    else:
        nodes, elements = mesh_gen.generate_simple_mesh()

    mesh_gen.write_output_file('inp',nodes, elements)
    Visualizer.plot_mesh(nodes, elements)
