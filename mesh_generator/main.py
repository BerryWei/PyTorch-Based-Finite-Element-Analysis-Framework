import argparse
from pathlib import Path
from utlis.Geometry import Geometry
from utlis.MeshGenerator import MeshGenerator
from utlis.Visualizer import Visualizer


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Mesh Generation from Geometry YAML.")
    parser.add_argument('--input', type=str, default='./geometry_p3.yaml', 
                        help='Path to the input YAML file containing geometry information.')
    parser.add_argument('--output', type=str, default='geometry.yaml', 
                        help='Name of the output YAML file to store generated mesh data.')
    
    return parser.parse_args()

def main(input_file, output_file):
    """Main function to generate and visualize mesh."""
    geometry_obj = Geometry(input_file)
    mesh_gen = MeshGenerator(geometry_obj)

    # Display if the geometry has holes or not
    if geometry_obj.holes:
        print("This geometry contains holes.")
        nodes, elements = mesh_gen.generate_mesh_with_hole()
    else:
        print("This geometry doesn't contain any holes.")
        nodes, elements = mesh_gen.generate_simple_mesh()
    
    # Write the generated mesh data to an output YAML file
    mesh_gen.write_output_file_yaml(filename=output_file, nodes=nodes, elements=elements, ndim=geometry_obj.ndim)
    
    # Visualize the generated mesh
    Visualizer.visualize_mesh(nodes, elements)

if __name__ == '__main__':
    args = parse_arguments()
    input_file = Path(args.input)
    output_file = args.output
    
    main(input_file, output_file)
