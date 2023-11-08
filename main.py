import torch
from pathlib import Path
from utlis.fem_module import FiniteElementModel
from utlis.element import *
import argparse
import time
import logging
from utlis.function import *

def get_loggings(ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name='FEM-Analysis')
    logger.setLevel(level=logging.INFO)
    # set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # file handler
    file_handler = logging.FileHandler(ckpt_dir / "record.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = get_loggings(Path('checkpoint'))

def initialize_model(args):
    """
    Initialize the Finite Element Model and load data.
    """
    logger.info("Initializing the Finite Element Model...")
    model = FiniteElementModel()

    logger.info(f"Reading geometry data from {args.geometry_path}...")
    model.read_geom_from_yaml(args.geometry_path)

    logger.info(f"Reading material data from {args.material_path}...")
    model.read_material_from_yaml(args.material_path)

    logger.info(f"Reading loading data from {args.loading_path}...")
    model.read_loading_from_yaml(args.loading_path)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to_device(device)
    
    return model

def run_analysis(model, ElementClass):
    """
    Run the FEM analysis.
    """
    start_time = time.time()
    logger.info("Computing the element stiffness...")

    model.generate_material_dict(ElementClass=ElementClass)

    if args.incompatible_mode_element == True:
        model.compute_element_stiffness_with_shear_locking(ElementClass=ElementClass)
    else:
        model.compute_element_stiffness(ElementClass=ElementClass)
    logger.info("Assembling the element stiffness...")
    model.assemble_global_stiffness()
    model.assemble_global_load_vector(ElementClass=ElementClass)
    model.solve_system()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Finite Element Model execution completed in {total_time:.2f} seconds.")


def post_processing(model, ElementClass):
    """
    Perform post-processing steps.
    """
    logger.info("Post-processing...")
    model.compute_GP_strains_stresses(ElementClass=ElementClass)



    node_strains = interpolate_to_nodes(
        input_coor=model.gauss_point_coordinates.cpu().numpy(),
        input_attribute=model.gauss_point_strains.cpu().numpy(),
        target_coor=model.node_coords.cpu().numpy()
    )

    node_stresses = interpolate_to_nodes(
        input_coor=model.gauss_point_coordinates.cpu().numpy(),
        input_attribute=model.gauss_point_stresses.cpu().numpy(),
        target_coor=model.node_coords.cpu().numpy()
    )

    model_cell_types = get_vtk_cell_type(
        ElementClass.node_per_element,
        model.parameters['num_dimensions']
    )

    if model.parameters['num_dimensions']==2:
        # Convert 2D node coordinates to 3D by adding a zero z-coordinate
        node_coords_3d = add_zero_z_coordinate(model.node_coords.cpu().numpy())
        disp_dict = {
            'disp_x' : model.global_displacements[0::2],
            'disp_y' : model.global_displacements[1::2],
        }

    else:
        node_coords_3d = model.node_coords.cpu().numpy()
        disp_dict = {
            'disp_x' : model.global_displacements[0::3],
            'disp_y' : model.global_displacements[1::3],
            'disp_z' : model.global_displacements[2::3],
        }


    strain_components = {'strain11': node_strains[:, 0], 'strain22': node_strains[:, 1], 'strain12': node_strains[:, 2]}
    stress_components = {'stress11': node_stresses[:, 0], 'stress22': node_stresses[:, 1], 'stress12': node_stresses[:, 2]}
    
    # compute von Mises stress in 2D
    stress11 = stress_components['stress11']
    stress22 = stress_components['stress22']
    stress12 = stress_components['stress12']
    von_mises_stress = np.sqrt(stress11**2 - stress11*stress22 + stress22**2 + 3*stress12**2)
    



    point_data = {
        **stress_components,
        **strain_components,
        **disp_dict,
        'von_mises_stress': von_mises_stress
    }

    parent_folder = args.geometry_path.parent
    write_to_vtk_manual(
        node_coords=node_coords_3d,
        cell_array=model.element_node_indices.cpu().numpy(),  # This should be the actual cells data from your model
        cell_types=np.full(model.element_node_indices.shape[0], model_cell_types),  # Create an array filled with the cell type
        point_data=point_data,
        filename=parent_folder / 'results.vtk'
    )




def main(args):
    model = initialize_model(args)
    ElementClass = eval(model.element_type)
    run_analysis(model, ElementClass)
    post_processing(model, ElementClass)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finite Element Model Execution')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to run the FEM.')
    parser.add_argument('--geometry_path', type=Path, default='.\example\hw4_Problem2_q4_c\geometry.yaml', help='Path to the geometry.yaml file.')
    parser.add_argument('--material_path', type=Path, default='.\example\hw4_Problem2_q4_c\material.yaml', help='Path to the material.yaml file.')
    parser.add_argument('--loading_path', type=Path,  default='.\example\hw4_Problem2_q4_c\loading.yaml', help='Path to the loading.yaml file.')
    parser.add_argument('--incompatible_mode_element', action='store_true', help='Flag to enable incompatible mode for the element.')
    args = parser.parse_args()
    main(args)
