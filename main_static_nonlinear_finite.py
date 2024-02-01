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
        while ( iter < 1.0e+3 and error > 1.0e-7):
            iter += 1


            model.update_prescribed_global_displacements_temp(factor=factor)
            model.compute_element_stiffness_nonlinear_finiteStrain()    
            model.assemble_global_stiffness()  
            model.assemble_global_residual()
            model.assemble_global_load_vector_nonlinear_finiteStrain(factor)
            model.solve_system_nonlinear_finiteStrain()
            error = torch.norm(model.u_incr_sub, float('inf'))

            # Compute error
            print(f'\titer={iter} error = {error}')

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
    model.compute_GP_strains_stresses_finiteStrain()

    node_E = interpolate_to_nodes(
        input_coor=model.gauss_point_coordinates.cpu().numpy(),
        input_attribute=model.gauss_point_E.cpu().numpy(),
        target_coor=model.node_coords.cpu().numpy()
    )

    node_S = interpolate_to_nodes(
        input_coor=model.gauss_point_coordinates.cpu().numpy(),
        input_attribute=model.gauss_point_S.cpu().numpy(),
        target_coor=model.node_coords.cpu().numpy()
    )

    node_F = interpolate_to_nodes(
        input_coor=model.gauss_point_coordinates.cpu().numpy(),
        input_attribute=model.gauss_point_F.cpu().numpy(),
        target_coor=model.node_coords.cpu().numpy()
    )

    node_P = interpolate_to_nodes(
        input_coor=model.gauss_point_coordinates.cpu().numpy(),
        input_attribute=model.gauss_point_P.cpu().numpy(),
        target_coor=model.node_coords.cpu().numpy()
    )

    node_Cauchy = interpolate_to_nodes(
        input_coor=model.gauss_point_coordinates.cpu().numpy(),
        input_attribute=model.gauss_point_Cauchy.cpu().numpy(),
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
        E_components = {'E11': node_E[:, 0], 'E22': node_E[:, 1], 'E12': node_E[:, 2]}
        S_components = {'S11': node_S[:, 0], 'S22': node_S[:, 1], 'S12': node_S[:, 2]}
        F_components = {'F11': node_F[:, 0], 'F12': node_F[:, 1], 'F21': node_F[:, 2], 'F22': node_F[:, 3]}
        P_components = {'P11': node_P[:, 0], 'P12': node_P[:, 1], 'P21': node_P[:, 2], 'P22': node_P[:, 3]}
        
        Cauchy_components = {'sigma11': node_Cauchy[:, 0], 'sigma22': node_Cauchy[:, 1], 'sigma12': node_Cauchy[:, 2]}
        
        # compute von Mises stress in 2D
        stress11 = Cauchy_components['sigma11']
        stress22 = Cauchy_components['sigma22']
        stress12 = Cauchy_components['sigma12']
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

        E_components = {
            'E11': node_E[:, 0],
            'E22': node_E[:, 1],
            'E33': node_E[:, 2],
            'E23': node_E[:, 3],
            'E13': node_E[:, 4],
            'E12': node_E[:, 5]
        }
        S_components = {
            'S11': node_S[:, 0],
            'S22': node_S[:, 1],
            'S33': node_S[:, 2],
            'S23': node_S[:, 3],
            'S13': node_S[:, 4],
            'S12': node_S[:, 5]
        }
        Cauchy_components = {
            'sigma11': node_Cauchy[:, 0],
            'sigma22': node_Cauchy[:, 1],
            'sigma33': node_Cauchy[:, 2],
            'sigma23': node_Cauchy[:, 3],
            'sigma13': node_Cauchy[:, 4],
            'sigma12': node_Cauchy[:, 5]
        }
        F_components = {'F11': node_F[:, 0], 'F12': node_F[:, 1], 'F13': node_F[:, 2],
                        'F21': node_F[:, 3], 'F22': node_F[:, 4], 'F23': node_F[:, 5],
                        'F31': node_F[:, 6], 'F32': node_F[:, 7], 'F33': node_F[:, 8],}
        
        P_components = {'P11': node_P[:, 0], 'P12': node_P[:, 1], 'P13': node_P[:, 2],
                        'P21': node_P[:, 3], 'P22': node_P[:, 4], 'P23': node_P[:, 5],
                        'P31': node_P[:, 6], 'P32': node_P[:, 7], 'P33': node_P[:, 8],}

        # Compute von Mises stress for 3D
        stress11 = Cauchy_components['sigma11']
        stress22 = Cauchy_components['sigma22']
        stress33 = Cauchy_components['sigma33']
        stress12 = Cauchy_components['sigma12']
        stress13 = Cauchy_components['sigma13']
        stress23 = Cauchy_components['sigma23']
        von_mises_stress = np.sqrt(
            0.5 * ((stress11 - stress22)**2 + (stress22 - stress33)**2 + (stress33 - stress11)**2) +
            3 * (stress12**2 + stress13**2 + stress23**2)
        )
    



    point_data = {
        **E_components,
        **S_components,
        **Cauchy_components,
        **disp_dict,
        **F_components,
        **P_components,
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
    parser.add_argument('--geometry_path', type=Path, default='.\example\project_openHole3d\geometry.yaml', help='Path to the geometry.yaml file.')
    parser.add_argument('--material_path', type=Path, default='.\example\project_openHole3d\material.yaml', help='Path to the material.yaml file.')
    parser.add_argument('--loading_path', type=Path,  default='.\example\project_openHole3d\loading.yaml', help='Path to the loading.yaml file.')
    parser.add_argument('--nPrint', type=int, default=1, help='Number of steps after which the .vtk files are saved. Controls the frequency of output for visualization.')
    parser.add_argument('--nLoad', type=int, default=100, help='Number of loading steps.')
    parser.add_argument('--incompatible_mode_element', action='store_true', help='Flag to enable incompatible mode for the element.')
    args = parser.parse_args()
    main(args)
