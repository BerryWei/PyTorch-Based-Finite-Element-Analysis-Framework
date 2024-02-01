import torch
from pathlib import Path
from utlis.fem_module import FiniteElementModel_nonlinear
from utlis.element import *
import argparse
import time
import logging
from utlis.function import *
import matplotlib.pyplot as plt

def get_loggings(ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name='FEM-Analysis: static')
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
    model = FiniteElementModel_nonlinear()

    logger.info(f"Reading geometry data from {args.geometry_path}...")
    model.read_geom_from_yaml(args.geometry_path)

    logger.info(f"Reading material data from {args.material_path}...")
    model.read_material_from_yaml(args.material_path)

    logger.info(f"Reading loading data from {args.loading_path}...")
    model.read_loading_from_yaml(args.loading_path)
    
    device = torch.device(args.device)
    model.to_device(device)
    
    return model

def run_analysis(model):
    """
    Run the FEM analysis.
    """
    start_time = time.time()

    # init material class
    model.init_element_class()
    model.generate_material_dict()
    
    # init temporay 
    model.init_global_displacements_temp()

    for step in range(args.nLoad+1):
        factor = (step)/args.nLoad
        iter = 0
        error = float('inf')
        
        print(f'step = {step}/{args.nLoad}, factor = {factor}')
        # Newton-Rapshon start
        while (iter < 1.0e+2 and error > 1.0e-7):
            iter += 1

            model.update_prescribed_global_displacements_temp(factor=factor)

            # compute stiffness matrix
            if args.incompatible_mode_element == True:
                model.compute_element_stiffness_with_shear_locking()
            else:
                #model.compute_element_stiffness_nonlinear()
                model.compute_element_stiffness_nonlinear_multicore(num_cores=32)
            model.assemble_global_stiffness()

            model.compute_element_residual()
            model.assemble_global_residual()
            model.assemble_global_load_vector_nonlinear(factor)

            model.solve_system_nonlinear()
            error = torch.norm(model.u_incr_sub, float('inf'))
            print(f'\titer={iter} error = {error}')
            # Store iteration and error for plotting

            

        # self.global_displacements_temp <-- self.global_displacements for printing
        model.update_displacement()

        if (step % args.nPrint == 0):
            post_processing(model, step)






    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Finite Element Model execution completed in {total_time:.2f} seconds.")


def post_processing(model, step:int):
    """
    Perform post-processing steps.
    """
    logger.info("Post-processing...")
    model.compute_GP_strains_stresses()



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
        model.elementClass.node_per_element,
        model.parameters['num_dimensions']
    )

    model.recoverError()
    abs_u_incr = model.global_abs_u_incr.cpu()




    if model.parameters['num_dimensions']==2:
        # Convert 2D node coordinates to 3D by adding a zero z-coordinate
        node_coords_3d = add_zero_z_coordinate(model.node_coords.cpu().numpy())
        disp_dict = {
            'disp_x' : model.global_displacements[0::2],
            'disp_y' : model.global_displacements[1::2],
            'abs_disp_x': abs_u_incr[0::2],
            'abs_disp_y': abs_u_incr[1::2]
        }
        strain_components = {'strain11': node_strains[:, 0], 'strain22': node_strains[:, 1], 'strain12': node_strains[:, 2]}
        stress_components = {'stress11': node_stresses[:, 0], 'stress22': node_stresses[:, 1], 'stress12': node_stresses[:, 2]}
        
        # compute von Mises stress in 2D
        stress11 = stress_components['stress11']
        stress22 = stress_components['stress22']
        stress12 = stress_components['stress12']
        von_mises_stress = np.sqrt(stress11**2 - stress11*stress22 + stress22**2 + 3*stress12**2)

    else:
        node_coords_3d = model.node_coords.cpu().numpy()
        disp_dict = {
            'disp_x' : model.global_displacements[0::3],
            'disp_y' : model.global_displacements[1::3],
            'disp_z' : model.global_displacements[2::3],
            'abs_disp_x': abs_u_incr[0::3],
            'abs_disp_y': abs_u_incr[1::3],
            'abs_disp_z': abs_u_incr[2::3]
        }

        strain_components = {
            'strain11': node_strains[:, 0],
            'strain22': node_strains[:, 1],
            'strain33': node_strains[:, 2],
            'strain23': node_strains[:, 3],
            'strain13': node_strains[:, 4],
            'strain12': node_strains[:, 5]
        }
        stress_components = {
            'stress11': node_stresses[:, 0],
            'stress22': node_stresses[:, 1],
            'stress33': node_stresses[:, 2],
            'stress23': node_stresses[:, 3],
            'stress13': node_stresses[:, 4],
            'stress12': node_stresses[:, 5]
        }

        # Compute von Mises stress for 3D
        stress11 = stress_components['stress11']
        stress22 = stress_components['stress22']
        stress33 = stress_components['stress33']
        stress12 = stress_components['stress12']
        stress13 = stress_components['stress13']
        stress23 = stress_components['stress23']
        von_mises_stress = np.sqrt(
            0.5 * ((stress11 - stress22)**2 + (stress22 - stress33)**2 + (stress33 - stress11)**2) +
            3 * (stress12**2 + stress13**2 + stress23**2)
        )
    



    point_data = {
        **stress_components,
        **strain_components,
        **disp_dict,
        'von_mises_stress': von_mises_stress
    }

    parent_folder = args.geometry_path.parent
    logger.info("Writing VTK files...")
    write_to_vtk_manual(
        node_coords=node_coords_3d,
        cell_array=model.element_node_indices.cpu().numpy(),  # This should be the actual cells data from your model
        cell_types=np.full(model.element_node_indices.shape[0], model_cell_types),  # Create an array filled with the cell type
        point_data=point_data,
        filename=parent_folder / f'results_{step}.vtk'
    )




def main(args):
    model = initialize_model(args)
    
    run_analysis(model)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finite Element Model Execution')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to run the FEM.')
    parser.add_argument('--geometry_path', type=Path, default='.\example\ytest_2d\geometry.yaml', help='Path to the geometry.yaml file.')
    parser.add_argument('--material_path', type=Path, default='.\example\ytest_2d\material.yaml', help='Path to the material.yaml file.')
    parser.add_argument('--loading_path', type=Path,  default='.\example\ytest_2d\loading.yaml', help='Path to the loading.yaml file.')
    parser.add_argument('--nPrint', type=int, default=1, help='Number of steps after which the .vtk files are saved. Controls the frequency of output for visualization.')
    parser.add_argument('--nLoad', type=int, default=10, help='Number of loading steps.')
    parser.add_argument('--incompatible_mode_element', action='store_true', help='Flag to enable incompatible mode for the element.')
    args = parser.parse_args()
    main(args)
