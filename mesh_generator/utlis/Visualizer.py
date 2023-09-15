import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

class Visualizer:
    @staticmethod
    def plot_mesh(nodes, elements):
        """Visualize the generated mesh."""
        nodes = np.array(nodes)
        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

        plt.figure(figsize=(10, 8))  # Setting a preferred figure size
        plt.gca().set_aspect('equal')
        
        # Drawing the triangles with black lines and gold nodes
        plt.triplot(triang, '-o', lw=0.5, color='gold', markerfacecolor='aqua', markersize=6)

        # Labeling the nodes
        for index, (x, y) in enumerate(nodes):
            plt.text(x, y, str(index+1), ha='center', va='center', color='blue', fontsize=9, weight='bold')

        # Labeling the elements
        for index, triangle in enumerate(elements):
            x_center = np.mean(nodes[triangle, 0])
            y_center = np.mean(nodes[triangle, 1])
            plt.text(x_center, y_center, '('+str(index+1)+')', ha='center', va='center', color='red', fontsize=7, weight='bold')

        plt.title('Generated Mesh')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()  # Ensures that labels and titles fit well within the plot
        plt.show()
