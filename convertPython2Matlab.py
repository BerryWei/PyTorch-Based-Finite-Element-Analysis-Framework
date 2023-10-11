import yaml

# Load the YAML file
with open("./Problem2_3layer/geometry.yaml", 'r') as file:
    data = yaml.safe_load(file)

# Convert the data from YAML to hw1-1 format
output_content = ""

# Convert PARAMETER section
if "PARAMETER" in data:
    output_content += "*PARAMETER\n"
    for key, value in data['PARAMETER'].items():
        output_content += f"{key}: {value}\n"
    output_content += "\n"

# Convert NODE section
if "NODE" in data:
    output_content += "*NODE\n"
    output_content += f"num-node: {data['NODE']['num-node']}\n"
    output_content += f"nodal-coord:\n"
    for coord in data['NODE']['nodal-coord']:
        output_content += " ".join(map(str, coord)) + "\n"
    

# Convert Element section
if "Element" in data:
    output_content += "*ELEMENT\n"
    output_content += f"num-elem: {data['Element']['num-elem']}\n"
    output_content += f"num-elem-node: {data['Element']['num-elem-node']}\n"
    output_content += "elem-conn:\n"
    for elem in data['Element']['elem-conn']:
        elems_adjusted = [val+1 for val in elem]
        output_content += " ".join(map(str, elems_adjusted)) + "\n"

# Save the output_content to hw1-1 format file
with open(r"input.matlab", "w") as output_file:
    output_file.write(output_content)
