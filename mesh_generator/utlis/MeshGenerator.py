import numpy as np
import triangle
import yaml
from typing import List, Tuple, Dict, Union

class MeshGenerator:
    def __init__(self, geometry):
        """
        Initializes the MeshGenerator with a provided geometry.

        Args:
            geometry (Geometry): An instance of the Geometry class containing boundary and holes data.
        """
        self.geometry = geometry

    def create_circle(self, center, radius, num_segments=20):
        """
        Create a polygonal representation of a circle.

        Args:
            center (Tuple[float, float]): Coordinates for the center of the circle.
            radius (float): Radius of the circle.
            num_segments (int): Number of segments to represent the circle. Default is 20.

        Returns:
            np.ndarray: An array of shape (num_segments, 2) representing the circle coordinates.
        """
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        return np.column_stack((x, y))

    def generate_mesh_with_hole(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a mesh considering the holes provided in the geometry.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing nodes (vertices) and elements (triangles).
        """
        
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

    def generate_simple_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a mesh without considering any holes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing nodes (vertices) and elements (triangles).
        """
        
        # Unpack boundary
        boundary = np.array(self.geometry.boundary)
        
        # Create segments for outer boundary
        segments = [[i, i+1] for i in range(len(boundary)-1)] + [[len(boundary)-1, 0]]
        
        poly = {'vertices': boundary, 'segments': segments}
        
        meshing_options = 'pq30'

        meshing_options += f'a{self.geometry.area:f}'
        
        mesh = triangle.triangulate(poly, meshing_options)
        
        return mesh['vertices'], mesh['triangles']

    @staticmethod
    def write_output_file_yaml(filename, nodes: np.ndarray, elements: np.ndarray, ndim: int = 2):
        """
        Write the generated mesh to a YAML file.

        Args:
            filename (str): The name/path of the output YAML file.
            nodes (np.ndarray): A 2D array of node coordinates.
            elements (np.ndarray): A 2D array of element connectivity.
        """

        data = {
            'PARAMETER': {
                'num-dim': ndim
            },
            # 'MATPROP': {
            #     'b-plane-strain': 1,
            #     "young's-modulus": 100.0,
            #     "poisson's-ratio": 0.3
            # },
            'NODE': {
                'num-node': len(nodes),
                'nodal-coord': [[float(n[0]), float(n[1])] for n in nodes]
            },
            'Element': {
                'type': 'TriangularElement',
                'num-elem': len(elements),
                'num-elem-node': 3,
                'elem-conn': [[int(e[0]), int(e[1]), int(e[2])] for e in elements]
            }
        }
        
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=None)
