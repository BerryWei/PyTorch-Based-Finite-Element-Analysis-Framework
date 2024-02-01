from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt

# 指定存放 VTK 檔案的資料夾路徑
folder_path = Path(r"D:\finite_element_method\PyTorch-Based-Finite-Element-Analysis-Framework\example\_3d_1element_Mooney")

# 初始化儲存應力和應變的列表
stress11 = []
strain11 = []

# 迭代處理每個 VTK 檔案
for i in range(101):
    # 建立每個檔案的完整路徑
    file_path = folder_path / f"results_{i}.vtk"
    
    # 讀取 VTK 檔案
    mesh = pv.read(file_path)

    stress_data = mesh.point_data['P11'][1]  # 假設字段名為 'stress11'194
    strain_data = mesh.point_data['F11'][1]  # 假設字段名為 'strain11'

    # 添加到列表中
    stress11.append(stress_data)  # 假設我們關注的是第一個分量
    strain11.append(strain_data)


# 使用 matplotlib 繪製應力-應變曲線
plt.plot(strain11, stress11, '-o')
plt.xlabel('F11')
plt.ylabel('P11(MPa)')

plt.show()
