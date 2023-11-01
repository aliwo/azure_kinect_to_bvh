import os

folder_path = '강우혁chair1_1000'

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_lines = []
    joint_zero_line = lines[0]  
    new_lines.append(joint_zero_line)  
    new_lines.append(joint_zero_line.replace('Joint 0:', 'Joint 1:'))  
    new_lines.append(joint_zero_line.replace('Joint 0:', 'Joint 2:'))  
    for line in lines[1:]:
        joint_number = int(line.split(':')[0].split()[-1])
        new_joint_number = joint_number + 2
        new_lines.append(line.replace(f'Joint {joint_number}:', f'Joint {new_joint_number}:'))
    with open(file_path, 'w') as file:
        file.writelines(new_lines)

for file_name in os.listdir(folder_path):
    if file_name.endswith('_joints.txt'):
        file_path = os.path.join(folder_path, file_name)
        process_file(file_path)

