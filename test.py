import csv
import torch

# Initialize an empty list to store the data
data_list = []

# Read the CSV file
with open('C:\\Users\\Berry\\Downloads\\k.csv', 'r', encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        try:
            data_list.append([float(x) if x else 0.0 for x in row])  # Replace empty fields with 0.0
        except ValueError as e:
            print(f"ValueError encountered at row {i}: {row}. Error message: {e}")

# Convert the list of lists into a Torch tensor
if data_list:
    tensor_data = torch.tensor(data_list)
k_matrix = tensor_data

with open('C:\\Users\\Berry\\Downloads\\r.csv', 'r', encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        try:
            data_list.append([float(x) if x else 0.0 for x in row])  # Replace empty fields with 0.0
        except ValueError as e:
            print(f"ValueError encountered at row {i}: {row}. Error message: {e}")

# Convert the list of lists into a Torch tensor
if data_list:
    tensor_data = torch.tensor(data_list)
r_matrix = tensor_data

print()