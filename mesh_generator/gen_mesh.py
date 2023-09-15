import triangle
import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

def generate_mesh_from_points(points):
    # 創建segment數據結構
    segments = [[i, i+1] for i in range(len(points)-1)] + [[len(points)-1, 0]]
    
    # 定義polygon
    poly = {'vertices': points, 'segments': segments}
    
    # 生成三角形網格, 使用 'pq' 參數和附加一個 'a' 參數以及相對應的最大面積
    mesh = triangle.triangulate(poly, 'pq30a0.1111')  # 使用 a 參數來定義每個三角形的最大面積
    
    return mesh['vertices'], mesh['triangles']

def write_output_file(filename, nodes, elements):
    with open(filename, 'w') as f:
        f.write("*PARAMETER\n")
        f.write("num-dim: 2\n")
        f.write("*MATPROP\n")
        f.write("b-plane-strain: 1\n")
        f.write("young's-modulus: 100.0\n")
        f.write("poisson's-ratio: 0.3\n")
        f.write("*NODE\n")
        f.write(f"num-node: {len(nodes)}\n")
        f.write("nodal-coord:\n")
        for node in nodes:
            f.write(f"{node[0]} {node[1]}\n")
        f.write("*ELEMENT\n")
        f.write(f"num-elem: {len(elements)}\n")
        f.write("num-elem-node: 3\n")
        f.write("elem-conn:\n")
        for elem in elements:
            f.write(f"{elem[0]+1} {elem[1]+1} {elem[2]+1}\n")

def plot_mesh(nodes, elements):
    # Convert nodes to a 2-column matrix
    nodes = np.array(nodes)

    # Create a Triangulation object
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Plot the mesh
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(triang, 'go-', lw=0.5)
    plt.title('Generated Mesh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

nodes, elements = generate_mesh_from_points(points)
write_output_file("output.txt", nodes, elements)
plot_mesh(nodes, elements)
