# Geometry.py

import yaml

class Geometry:
    def __init__(self, filename):
        """Initialize a Geometry object from a YAML file."""
        self.boundary = []
        self.holes = []
        self._load_from_yaml(filename)
        
    def _load_from_yaml(self, filename):
        """Load geometry from a YAML file."""
        with open(filename, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            self.boundary = data['geometry']['boundary']
            self.holes = data['geometry'].get('holes', [])
            self.area = data['mesh']['area']
