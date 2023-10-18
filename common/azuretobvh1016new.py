import numpy as np
import re
from scipy.spatial.transform import Rotation as R
import os

folder_path = 'new_3d_dychair1_1000frame'
file_names = os.listdir(folder_path)

joint_structure = {
    'PELVIS': 0,
    'SPINE_NAVAL': 1,
    'SPINE_CHEST': 2,
    'NECK': 3,
    'CLAVICLE_LEFT': 4,
    'SHOULDER_LEFT': 5,
    'ELBOW_LEFT': 6,
    'WRIST_LEFT': 7,
    'HAND_LEFT': 8,
    'HANDTIP_LEFT': 9,
    'THUMB_LEFT': 10,
    'CLAVICLE_RIGHT': 11,
    'SHOULDER_RIGHT': 12,
    'ELBOW_RIGHT': 13,
    'WRIST_RIGHT': 14,
    'HAND_RIGHT': 15,
    'HANDTIP_RIGHT': 16,
    'THUMB_RIGHT': 17,
    'HIP_LEFT': 18,
    'KNEE_LEFT': 19,
    'ANKLE_LEFT': 20,
    'FOOT_LEFT': 21,
    'HIP_RIGHT': 22,
    'KNEE_RIGHT': 23,
    'ANKLE_RIGHT': 24,
    'FOOT_RIGHT': 25,
    'HEAD': 26,
    'NOSE': 27,
    'EYE_LEFT': 28,
    'EAR_LEFT': 29,
    'EYE_RIGHT': 30,
    'EAR_RIGHT': 31
}

euler_orders = {
    'PELVIS': 'zyx',
    'SPINE_NAVAL': 'zyx',
    'NECK': 'zyx',
    'HEAD': 'zyx',
    'SHOULDER_RIGHT': 'zyx',
    'ELBOW_RIGHT': 'zyx',
    'HAND_RIGHT': 'zyx',
    'SHOULDER_LEFT': 'zyx',
    'ELBOW_LEFT': 'zyx',
    'HAND_LEFT': 'zyx',
    'KNEE_RIGHT': 'zyx',
    'ANKLE_RIGHT': 'zyx',
    'FOOT_RIGHT': 'zyx',
    'KNEE_LEFT': 'zyx',
    'ANKLE_LEFT': 'zyx',
    'FOOT_LEFT': 'zyx',
}


desired_joints = set([
    'PELVIS', 'SPINE_NAVAL', 'SPINE_CHEST', 'NECK', 'CLAVICLE_LEFT',
    'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT', 'HAND_LEFT', 'HANDTIP_LEFT',
    'THUMB_LEFT', 'CLAVICLE_RIGHT', 'SHOULDER_RIGHT', 'ELBOW_RIGHT',
    'WRIST_RIGHT', 'HAND_RIGHT', 'HANDTIP_RIGHT', 'THUMB_RIGHT', 'HIP_LEFT',
    'KNEE_LEFT', 'ANKLE_LEFT', 'FOOT_LEFT', 'HIP_RIGHT', 'KNEE_RIGHT',
    'ANKLE_RIGHT', 'FOOT_RIGHT', 'HEAD', 'NOSE', 'EYE_LEFT', 'EAR_LEFT',
    'EYE_RIGHT', 'EAR_RIGHT'
])



def add_pelvis_coordinates(data, x, y, z):
    new_data = []
    for line in data:
        new_line = [x, y, z] + line
        new_data.append(new_line)
    return new_data



def calc_world_dir(joint_data):
    joint_world_dir = {} # dir : me -> child

    joint_world_pos = {}
    for line in joint_data:
        joint_name = extract_joint_name(line)
        if joint_name == None:
            continue

        world_pos = extract_joint_world_pos(line)
        joint_world_pos[joint_name] = world_pos
    
    for joint in joint_info.values():
        joint_name = joint['name']

        children = find_children(joint_name)
        if len(children) is 0:
            joint_world_dir[joint_name] = np.array([0, 0, 0])
            continue

        child_name = children[0]
        joint_world_dir[joint_name] = joint_world_pos[child_name] - joint_world_pos[joint_name]
        joint_world_dir[joint_name] = joint_world_dir[joint_name] / np.linalg.norm(joint_world_dir[joint_name])
    return joint_world_dir

def modify_bvh_channels(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    for index, line in enumerate(lines):
        if "ROOT PELVIS" in line:
            for offset_index, offset_line in enumerate(lines[index:]):
                if "CHANNELS" in offset_line:
                    lines[index + offset_index] = "    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
                    break
            break
    with open(filename, 'w') as file:
        file.writelines(lines)

        
def extract_positions_from_file(filename):
    positions = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r'Position\(([^)]+)\)', line)
            if match:
                position_str = match.group(1).split(',')
                position = [float(x)/10 for x in position_str]  # Convert to cm
                positions.append(position)
    return positions

def parse_joint_structure(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        joint_structure, parent_joints = {}, {}
        for line in lines:
            idx, joint_name, parent_name = line.strip().split('\t')
            joint_structure[joint_name] = int(idx)
            parent_joints[joint_name] = parent_name if parent_name != "-" else None
    return joint_structure, parent_joints

def compute_offsets(positions, joint_structure, parent_joints):
    offsets = {}
    pelvis_offset = positions[0]  
    offsets["PELVIS"] = [0, 0, 0]

    for joint, parent in parent_joints.items():
        if joint != "PELVIS":
            child_position = positions[joint_structure[joint]]
            parent_position = positions[joint_structure[parent]]

            adjusted_child_position = [child_position[i] - pelvis_offset[i] for i in range(3)]
            adjusted_parent_position = [parent_position[i] - pelvis_offset[i] for i in range(3)]

            offset = [adjusted_child_position[i] - adjusted_parent_position[i] for i in range(3)]
            offsets[joint] = offset
            
    return offsets

def extract_joint_name(line):
    match = re.search(r"Joint (\d+):", line)
    if match:
        joint_index = int(match.group(1))
        if joint_index in joint_info: 
            return joint_info[joint_index]['name']
    return None  

def extract_joint_world_pos(line):
    match = re.search(r'Position\(([^)]+)\)', line)
    if match:
        pos = match.group(1)
        return np.array([float(i) for i in pos.split(',')])
    return None

def find_children(my_name):
    children = []
    for element in joint_info.values():
        if element['parent'] == my_name:
            children.append(element['name'])
    return children


def write_joint(index, indent, bvh_header):
    joint = joint_info[index]
    joint_name = joint['name']
    parent_name = joint['parent']
    offset = offsets.get(joint_name, "0.0 0.0 0.0")
    if joint_name in desired_joints: 
        bvh_header += f"{'    ' * indent}{'ROOT ' if parent_name == '-' else 'JOINT '}{joint_name}\n"
        bvh_header += f"{'    ' * indent}{{\n"
        bvh_header += f"{'    ' * (indent + 1)}OFFSET {offset}\n"
        bvh_header += f"{'    ' * (indent + 1)}CHANNELS 3 Zrotation Xrotation Yrotation\n"
        for child_index, child_info in joint_info.items():
            if child_info['parent'] == joint_name:
                bvh_header = write_joint(child_index, indent + 1, bvh_header)
        bvh_header += f"{'    ' * indent}}}\n"
    return bvh_header


joint_structure_file = 'joint_structure.txt'
with open(joint_structure_file, 'r') as file:
    joint_structure_data = file.readlines()[1:] 

joint_structure, parent_joints = parse_joint_structure(joint_structure_file)
first_frame_positions = extract_positions_from_file(os.path.join(folder_path, file_names[0]))
offsets = compute_offsets(first_frame_positions, joint_structure, parent_joints)
print(offsets)

joint_info = {}
for line in joint_structure_data:
    index, joint_name, parent_name = line.strip().split('\t')
    joint_info[int(index)] = {'name': joint_name, 'parent': parent_name}


print(joint_info)

bvh_header = "HIERARCHY\n"
joint_stack = []
bvh_header = write_joint(0, 0, bvh_header)  


for i in range(len(joint_stack)):
    bvh_header += "\t" * (len(joint_stack) - i - 1)
    bvh_header += "}\n"

bvh_motion = "MOTION\n"
frame_count = len([f for f in os.listdir(folder_path) if f.endswith('_joints.txt')])
bvh_motion += f"Frames: {frame_count}\n"
bvh_motion += "Frame Time: 0.01\n"

#0번째 frame의 world dir vec구하기
initial_world_dir = []
file_name = f'frame0_body0_timestamp0_1_joints.txt'
with open(os.path.join(folder_path, file_name), 'r') as file:
    joint_data = file.readlines()
    initial_world_dir = calc_world_dir(joint_data)

#각 frame 회전값 계산
for frame_num in range(frame_count):
    file_name = f'frame{frame_num}_body0_timestamp{frame_num * 10}_1_joints.txt'
    with open(os.path.join(folder_path, file_name), 'r') as file:
        joint_data = file.readlines()
        
        pelvis_position = None
        current_world_dir = calc_world_dir(joint_data)
        
        for line in joint_data:
            joint_name = extract_joint_name(line)
            if joint_name is None:  
                continue

            if joint_name == "PELVIS":
                position_match = re.search(r'Position\(([^)]+)\)', line)
                if position_match:
                    position = position_match.group(1)
                    pelvis_position = [float(i) for i in position.split(',')]
                    bvh_motion += f"{pelvis_position[0]/10} {pelvis_position[1]/10} {pelvis_position[2]/10} "

            children = find_children(joint_name)

            if len(children) is 0:
                bvh_motion += f"0 0 0 "
                continue
                
            from_dir = initial_world_dir[joint_name]
            to_dir = current_world_dir[joint_name]

            if np.dot(from_dir, to_dir) >= 1 - 0.00001:
                bvh_motion += f"0 0 0 "
                continue

            quat = np.cross(from_dir, to_dir)
            quat = np.append(quat, ((np.linalg.norm(from_dir) ** 2) * (np.linalg.norm(to_dir) ** 2))** 0.5 + np.dot(from_dir, to_dir))
            euler = R.from_quat(quat).as_euler('xyz', degrees=True) #설명해서 normalize 시켜준다함. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
            bvh_motion += f"{euler[0]} {euler[1]} {euler[2]} "

    bvh_motion += "\n"

with open('output.bvh', 'w') as file:
    file.write(bvh_header + bvh_motion)
modify_bvh_channels('output.bvh')
