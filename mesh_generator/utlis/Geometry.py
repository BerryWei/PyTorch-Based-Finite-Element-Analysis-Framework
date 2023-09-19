import yaml
from pathlib import Path
from typing import List, Dict, Optional

class Geometry:
    def __init__(self, filename: Path):
        """
        Initialize a Geometry object from a YAML file.

        Args:
            filename (Path): The path to the YAML file.
        """
        self.boundary = []
        self.holes = []
        self._load_from_yaml(filename)
        
    def _load_from_yaml(self, filename: Path) -> None:
        """
        Load geometry data from a YAML file.

        Args:
            filename (Path): The path to the YAML file.
        """
        with open(filename, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            self.boundary = data['geometry']['boundary']
            self.holes = data['geometry'].get('holes', [])
            self.area = data['mesh']['area']
            self.ndim = data['num-dim']
