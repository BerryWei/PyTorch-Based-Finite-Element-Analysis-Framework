import yaml
from decimal import Decimal

with open(r"D:\finite_element_method\PyTorch-Based-Finite-Element-Analysis-Framework\example\hw7_Problem1\hw7-1", "r") as file:
    hw1_1_content = file.read()
# Splitting the content into sections
sections = [section.strip() for section in hw1_1_content.split('*') if section.strip()]

# Initializing the dictionary to store the parsed data
data = {}
boundary = {}
# Parsing each section
for section in sections:
    lines = section.split('\n')
    header = lines[0]
    content = lines[1:]
    
    if header == "PARAMETER":
        # Parsing the PARAMETER section
        for line in content:
            key, value = line.split(':')
            data['PARAMETER'] = {key.strip():int(value.strip())}
    
    elif header == "NODE":
        # Parsing the NODE section
        num_node = int(content[0].split(":")[1].strip())
        coord_lines = content[2:2+num_node]
        coords = [list(map(float, line.split())) for line in coord_lines]



        
        data["NODE"] = {
            "num-node": num_node,
            "nodal-coord": coords
        }
    
    elif "elem" in header.lower():
        # Assuming this section is for elements (given the lack of explicit header in the sample)
        num_elem = int(content[0].split(":")[1].strip())
        num_elem_node = int(content[1].split(":")[1].strip())
        elem_lines = content[3:3+num_elem]
        elems = [list(map(int, line.split())) for line in elem_lines]
        elems_adjusted = [[val-1 for val in elem] for elem in elems]
        data["Element"] = {
            "num-elem": num_elem,
            "num-elem-node": num_elem_node,
            "elem-conn": elems_adjusted,
            "type": "T3Element"
        }

    elif header == "BOUNDARY":

        num_prescribed_disp = int(content[0].split(":")[1].strip())
        prescribed_disp_lines = content[2:2+num_prescribed_disp]
        prescribed_disps = [list(map(float, line.split())) for line in prescribed_disp_lines]
        
        # Adjust the node and dof values
        for disp in prescribed_disps:
            disp[0] -= 1
            disp[1] -= 1
        
        num_prescribed_load = int(content[2+num_prescribed_disp].split(":")[1].strip())
        prescribed_load_lines = content[4+num_prescribed_disp:]
        prescribed_loads = [list(map(float, line.split())) if line else [] for line in prescribed_load_lines]

        boundary["BOUNDARY"] = {
                "num-prescribed-disp": num_prescribed_disp,
                "prescribed-disps": prescribed_disps,
                "num-prescribed-load": num_prescribed_load,
                "prescribed-loads": prescribed_loads
            }



# Saving the converted data to input.yaml
output_path = r"D:\finite_element_method\PyTorch-Based-Finite-Element-Analysis-Framework\example\hw7_Problem1/geometry.yaml"
with open(output_path, "w") as f:
    yaml.dump(data, f, default_flow_style=None)

print("num-prescribed-disp:", boundary["BOUNDARY"]["num-prescribed-disp"])
print("node#-dof#-disp:")
for disp in boundary["BOUNDARY"]["prescribed-disps"]:
    print(disp[0], disp[1], disp[2])

for disp in boundary["BOUNDARY"]["prescribed-disps"]:
    print(disp[0], disp[1], disp[2])