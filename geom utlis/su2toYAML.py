# Combine all the steps into a complete code snippet for converting SU2 to YAML format

import yaml

def read_su2_file(filepath):
    node_coords = []
    elem_conn = []
    current_section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%') or not line:
                continue

            if "NPOIN=" in line:
                current_section = 'nodes'
                continue
            elif "NELEM=" in line:
                current_section = 'elements'
                continue
            elif "NMARK=" in line or "MARKER_TAG" in line or "MARKER_ELEMS" in line:
                current_section = None

            if current_section == 'nodes':
                try:
                    coords = list(map(float, line.split()[:-1]))
                    node_coords.append(coords)
                except ValueError:
                    pass
            elif current_section == 'elements':
                try:
                    conn = list(map(int, line.split()[1:-1]))
                    elem_conn.append(conn)
                except ValueError:
                    pass
    
    return node_coords, elem_conn

def convert_to_yaml(node_coords, elem_conn, yaml_filepath, dim=3):
    node_coords_clean = [coords[:dim] for coords in node_coords]
    elem_conn_clean = [list(map(lambda x: x , conn)) for conn in elem_conn]
    if dim==3:
        type_name = 'BrickElement'
    else:
        type_name = 'QuadElement'
    
    data_dict = {
        'Element': {
            'elem-conn': elem_conn_clean,
            'num-elem': len(elem_conn_clean),
            'num-elem-node': len(elem_conn_clean[0]),
            'type': type_name
        },
        'NODE': {
            'nodal-coord': node_coords_clean,
            'num-node': len(node_coords_clean)
        },
        'PARAMETER': {'num-dim': dim}
    }

    with open(yaml_filepath, 'w') as f:
        yaml.dump(data_dict, f, default_flow_style=None)

# File paths
su2_filepath =  r'D:\finite_element_method\PyTorch-Based-Finite-Element-Analysis-Framework\example\hw4_Problem3_3d/brick_3d.su2'
yaml_filepath = r'D:\finite_element_method\PyTorch-Based-Finite-Element-Analysis-Framework\example\hw4_Problem3_3d/geometry.yaml'

# Read SU2 file and convert to YAML format
node_coords, elem_conn = read_su2_file(su2_filepath)
convert_to_yaml(node_coords, elem_conn, yaml_filepath)

yaml_filepath
