import torch
from pathlib import Path
from utlis.fem_module import FiniteElementModel  
import argparse

def main(args):
    print("Initializing the Finite Element Model...")
    # Initialize the model
    model = FiniteElementModel()

    print(f"Reading geometry data from {args.geometry_path}...")
    model.read_geom_from_yaml(args.geometry_path)

    print(f"Reading material data from {args.material_path}...")
    model.read_material_from_yaml(args.material_path)

    print(f"Reading loading data from {args.loading_path}...")
    model.read_loading_from_yaml(args.loading_path)

    # Move the model data to GPU if available and desired
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        model.to_device(device)
    else:
        device = torch.device("cpu")
        model.to_device(device)

    if model.is_data_on_cuda():
        print("All data is on CUDA.")
    else:
        print("Some data is not on CUDA.")


    print("Finite Element Model execution completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finite Element Model Execution')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='Device to run the FEM: "cpu" or "cuda". Default is "cpu".')
    parser.add_argument('--geometry_path', type=Path, default='geometry.yaml',
                        help='Path to the geometry.yaml file. Default is "inp.yaml".')
    parser.add_argument('--material_path', type=Path, default='material.yaml',
                        help='Path to the material.yaml file. Default is "material.yaml".')
    parser.add_argument('--loading_path', type=Path, default='loading.yaml',
                        help='Path to the loading.yaml file. Default is "loading.yaml".')

    args = parser.parse_args()
    main(args)
