def parse_and_filter_nodes(mesh_lines, x_value=None, y_value=None, z_value=None, tolerance=1e-5):
    """
    Parses the mesh file lines and filters out nodes based on x, y, and/or z values within a specified tolerance.

    :param mesh_lines: The lines of the mesh file.
    :param x_value: The x value to filter nodes on (if None, x is not considered).
    :param y_value: The y value to filter nodes on (if None, y is not considered).
    :param z_value: The z value to filter nodes on (if None, z is not considered).
    :param tolerance: The tolerance within which to consider values as matching.
    :return: A list of node indices that match the filter conditions.
    """
    # Find the start of the nodes list
    nodes_start = None
    for i, line in enumerate(mesh_lines):
        if line.startswith('NPOIN='):
            nodes_start = i + 1
            break

    # Check if the start was found
    if nodes_start is None:
        raise ValueError("The node list start ('NPOIN=') was not found in the mesh file.")

    # Parse nodes and apply filters
    filtered_nodes = []
    for line in mesh_lines[nodes_start:]:
        if line == 'NMARK= 1\n':
            break
        parts = line.split()
        if len(parts) < 4:  # 2d case (x, y, z, index)
            x, y, index = float(parts[0]), float(parts[1]), float(parts[2])
            conditions = [
                (x_value is None or abs(x - x_value) <= tolerance),
                (y_value is None or abs(y - y_value) <= tolerance),
            ]
        else:
            x, y, z, index = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
            conditions = [
                (x_value is None or abs(x - x_value) <= tolerance),
                (y_value is None or abs(y - y_value) <= tolerance),
                (z_value is None or abs(z - z_value) <= tolerance)
            ]
        if all(conditions):
            filtered_nodes.append(index)

    return filtered_nodes



su2_filepath =  r'D:\finite_element_method\PyTorch-Based-Finite-Element-Analysis-Framework\example\project_openHole3d/openHole3d.su2'
with open(su2_filepath, 'r') as file:
    mesh_data = file.readlines()
# Apply the function to the mesh data with different filter conditions
nodes_at_x0 = parse_and_filter_nodes(mesh_data, x_value=0)
nodes_at_y0 = parse_and_filter_nodes(mesh_data, y_value=0)
nodes_at_z0 = parse_and_filter_nodes(mesh_data, z_value=0)

nodes_at_x1 = parse_and_filter_nodes(mesh_data, x_value=1)
nodes_at_y2 = parse_and_filter_nodes(mesh_data, y_value=2)
nodes_at_z1 = parse_and_filter_nodes(mesh_data, z_value=1)

# 3d
nodes_union = list(set(nodes_at_y2) )

#node_interaction = list(set(parse_and_filter_nodes(mesh_data, x_value=10)) &
#                         set(parse_and_filter_nodes(mesh_data, y_value=10)) & 
#                         set(nodes_at_z10))

# 2d
#nodes_union = list(set(nodes_at_y0) & set(nodes_at_x10) )
# node_interaction = list(set(parse_and_filter_nodes(mesh_data, x_value=10)) &
#                          set(parse_and_filter_nodes(mesh_data, y_value=10)) )

# print(node_interaction)

print('COPY TO PASTE'+'*'*20)
for idx, node_id in enumerate(nodes_union):
    #print(f'  - [{node_id}, 0, 0.0]')
    print(f'  - [{node_id}, 1, 2.0]')
    #print(f'  - [{node_id}, 2, 0.0]')
print('END'+'*'*20)