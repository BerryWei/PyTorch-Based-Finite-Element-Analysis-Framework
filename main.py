import torch
from pathlib import Path
from utlis.fem_module import FiniteElementModel 
from utlis.element import *
import argparse
import time

def main(args):
    print("Initializing the Finite Element Model...")
    start_time = time.time()  # 记录开始时间

    # 初始化模型
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


    print(f"Compute the element stiffness...")
    model.compute_element_stiffness(material=model.material, ElementClass=eval(model.element_type))
  
    print(f"Assemble the element stiffness...")
    model.assemble_global_stiffness()
    assemble_end_time = time.time()  

    model.assemble_global_load_vector(ElementClass=eval(model.element_type))
    model.solve_system()

    model.compute_elemental_strains_stresses(ElementClass=eval(model.element_type))

    model.save_results_to_file(file_path=Path('output.opt'))

    def elemental_strain_x():
        return model.elemental_strains[:, 0]
    
    def elemental_stress_x():
        return model.elemental_stresses[:, 0]
    


    model.plot([elemental_strain_x])

    model.plot([elemental_stress_x])




    total_time = assemble_end_time - start_time  
    print(f"Finite Element Model execution completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finite Element Model Execution')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='Device to run the FEM: "cpu" or "cuda". Default is "cpu".')
    parser.add_argument('--geometry_path', type=Path, default='geometry.yaml',
                        help='Path to the geometry.yaml file. Default is "geometry.yaml".')
    parser.add_argument('--material_path', type=Path, default='material.yaml',
                        help='Path to the material.yaml file. Default is "material.yaml".')
    parser.add_argument('--loading_path', type=Path, default='loading.yaml',
                        help='Path to the loading.yaml file. Default is "loading.yaml".')

    args = parser.parse_args()
    main(args)
