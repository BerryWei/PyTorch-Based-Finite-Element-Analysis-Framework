import numpy as np
import triangle

class MeshGenerator:
    def __init__(self, geometry):
        self.geometry = geometry

    def create_circle(self, center, radius, num_segments=20):
        """Create a polygonal representation of a circle."""
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        return np.column_stack((x, y))

    def generate_mesh_with_hole(self):
        """Generate mesh based on provided geometry."""
        
        # Unpack boundary and holes
        boundary = np.array(self.geometry.boundary)
        holes = self.geometry.holes
        
        # Create segments for outer boundary
        segments = [[i, i+1] for i in range(len(boundary)-1)] + [[len(boundary)-1, 0]]
        
        hole_centers = []
        for hole in holes:
            hole_points = self.create_circle(hole['center'], hole['radius'])
            hole_segments_start_idx = len(boundary)
            hole_segments = [[i + hole_segments_start_idx, i + hole_segments_start_idx + 1] for i in range(len(hole_points)-1)] + [[hole_segments_start_idx + len(hole_points) - 1, hole_segments_start_idx]]

            boundary = np.vstack((boundary, hole_points))
            segments += hole_segments
            hole_centers.append(hole['center'])

        poly = {'vertices': boundary, 'segments': segments, 'holes': hole_centers}
        
        meshing_options = 'pq30'

        meshing_options += f'a{self.geometry.area}'
        
        mesh = triangle.triangulate(poly, meshing_options)
        
        return mesh['vertices'], mesh['triangles']

    def generate_simple_mesh(self):
        """Generate mesh without considering any holes."""
        
        # Unpack boundary
        boundary = np.array(self.geometry.boundary)
        
        # Create segments for outer boundary
        segments = [[i, i+1] for i in range(len(boundary)-1)] + [[len(boundary)-1, 0]]
        
        poly = {'vertices': boundary, 'segments': segments}
        
        meshing_options = 'pq30'

        meshing_options += f'a{self.geometry.area}'
        
        mesh = triangle.triangulate(poly, meshing_options)
        
        return mesh['vertices'], mesh['triangles']

    def write_output_file(self, filename, nodes, elements):
        with open(filename, 'w') as f:
            f.write("*PARAMETER\n")
            f.write("num-dim: 2\n")
            f.write("*MATPROP\n")
            f.write("b-plane-strain: 1\n")
            f.write("young's-modulus: 100.0\n")
            f.write("poisson's-ratio: 0.3\n")
            f.write("*NODE\n")
            f.write(f"num-node: {len(nodes)}\n")
            f.write("nodal-coord:\n")
            for node in nodes:
                f.write(f"{node[0]} {node[1]}\n")
            f.write("*ELEMENT\n")
            f.write(f"num-elem: {len(elements)}\n")
            f.write("num-elem-node: 3\n")
            f.write("elem-conn:\n")
            for elem in elements:
                f.write(f"{elem[0]+1} {elem[1]+1} {elem[2]+1}\n")