import numpy as np
import re
from scipy.spatial.transform import Rotation as R
import os


# folder_path = __file__.replace('azuretobvh.py', '') + 'new_3d_dychair1_1000frame'
# folder_path = __file__.replace('azuretobvh.py', '') + 'new_3d_dychair1_1000frame_noflip_34joints'
# folder_path = 'sc/' + 'new_3d_dychair1_1000frame'
folder_path = 'sc/' + 'multi_joint_edited'

# folder_path = 'x_axis_60_rot'
# folder_path = 'two_joints_z_axis_60'
# folder_path = 'two_joints_z_axis_60_z_axis_-60'
# folder_path = 'pelvis_offset'
# folder_path = 'data'
file_names = os.listdir(folder_path)

offsets = {}


def extract_joint_name(line):
    match = re.search(r"Joint (\d+):", line)
    if match:
        joint_index = int(match.group(1))        
        if joint_index in joint_info:
            return joint_info[joint_index]['name']
    return None  # 일치하는 내용이 없을 경우 None 반환

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

def get_parent_name(my_name):
    for element in joint_info.values():
        if element['name'] == my_name:
            return element['parent']
    return None

def write_joint(index, indent, bvh_header):
    joint = joint_info[index]
    joint_name = joint['name']
    parent_name = joint['parent']
    offset = offsets.get(joint_name, "0.0 0.0 0.0")
    bvh_header += f"{'    ' * indent}{'ROOT ' if parent_name == '-' else 'JOINT '}{joint_name}\n"
    bvh_header += f"{'    ' * indent}{{\n"
    bvh_header += f"{'    ' * (indent + 1)}OFFSET {offset}\n"
    bvh_header += f"{'    ' * (indent + 1)}CHANNELS {'6 Xposition Yposition Zposition' if parent_name == '-' else '3'} Zrotation Yrotation Xrotation\n"
    # bvh_header += f"{'    ' * (indent + 1)}CHANNELS 3 Xrotation Yrotation Zrotation\n"
    for child_index, child_info in joint_info.items():
        if child_info['parent'] == joint_name:
            bvh_header = write_joint(child_index, indent + 1, bvh_header)
    bvh_header += f"{'    ' * indent}}}\n"
    return bvh_header

def calc_bone_length(joint_data):
    joint_world_pos = {}

    for line in joint_data:
        joint_name = extract_joint_name(line)
        if joint_name == None:
            continue

        world_pos = extract_joint_world_pos(line)
        joint_world_pos[joint_name] = world_pos

    for joint in joint_info.values():
        parent_name = joint['parent']
        my_name = joint['name']

        if parent_name not in joint_world_pos:
            offsets[my_name] = f"0 0 0"
            continue

        bone_len = joint_world_pos[my_name] - joint_world_pos[parent_name]
        bone_len /= 10
        offsets[my_name] = f"{bone_len[0]} {bone_len[1]} {bone_len[2]}"

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

def get_from_to_quat(from_dir, to_dir):
    if np.dot(from_dir, to_dir) >= 1 - 0.00001:
        quat = R.from_euler('xyz', [0, 0, 0]).as_quat()
    else:
        quat = np.cross(from_dir, to_dir)
        quat = np.append(quat, ((np.linalg.norm(from_dir) ** 2) * (np.linalg.norm(to_dir) ** 2))** 0.5 + np.dot(from_dir, to_dir))
    return quat    

#joint 구조 파악. 사용할 joint를 여기서 조정
# joint_structure_file = 'sc/' + 'joint_structure.txt'
joint_structure_file = 'sc/' + 'joint_structure_multi_joint_edited.txt'
with open(joint_structure_file, 'r') as file:
    joint_structure_data = file.readlines()[1:]  
joint_info = {}
for line in joint_structure_data:
    index, joint_name, parent_name = line.strip().split('\t')
    joint_info[int(index)] = {'name': joint_name, 'parent': parent_name}

#가장 처음 offset 구하기
file_name = f'frame0_body0_timestamp0_1_joints.txt'
with open(os.path.join(folder_path, file_name), 'r') as file:
    joint_data = file.readlines()
    calc_bone_length(joint_data)

#BVH Header 작성
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

#0번째 frame의 world dir vec | pelvis pos 구하기
initial_world_dir = []
pelvis_init_world_pos = 0
file_name = f'frame0_body0_timestamp0_1_joints.txt'
with open(os.path.join(folder_path, file_name), 'r') as file:
    joint_data = file.readlines()
    initial_world_dir = calc_world_dir(joint_data)
    pelvis_init_world_pos = extract_joint_world_pos(joint_data[0])

#각 frame 회전값 계산
for frame_num in range(frame_count):
    file_name = f'frame{frame_num}_body0_timestamp{frame_num * 10}_1_joints.txt'
    with open(os.path.join(folder_path, file_name), 'r') as file:
        joint_data = file.readlines()

        #position 게산 (pelvis)
        pelvis_world_pos = extract_joint_world_pos(joint_data[0]) - pelvis_init_world_pos
        pelvis_world_pos *= 0.1 #TODO 임시
        bvh_motion += f"{pelvis_world_pos[0]} {pelvis_world_pos[1]} {pelvis_world_pos[2]} "
        #bvh_motion += f"0 0 0 "

        #rotation 계산
        parent_quat = {}
        current_world_dir = calc_world_dir(joint_data)
        for line in joint_data:
            joint_name = extract_joint_name(line)
            if joint_name == None:
                continue

            children = find_children(joint_name)
            if len(children) is 0:
                bvh_motion += f"0 0 0 "
                continue

            is_root = get_parent_name(joint_name) is '-'
            if is_root:
                from_dir = initial_world_dir[joint_name]
            else:
                ### 각 parent quat는 로컬 rotation을 의미하는 quaternion이라, pelvis quat까지 보려면 중첩 연산을 여기서 해줘야 한다.
                ancestor_names = []
                cur_joint_name = joint_name
                parent_name = get_parent_name(cur_joint_name)
                while parent_name is not '-':
                    ancestor_names.append(parent_name)
                    cur_joint_name = parent_name
                    parent_name = get_parent_name(cur_joint_name)
                
                ancestor_names.reverse()
                accum_quat = R.identity()
                for name in ancestor_names:
                    accum_quat = accum_quat * parent_quat[name]

                from_dir = accum_quat.apply(initial_world_dir[joint_name]) 
            to_dir = current_world_dir[joint_name]
            
            quat = get_from_to_quat(from_dir, to_dir)

            # is_dummy_joint = (joint_name == "PELVIS_LEFT" or joint_name == "PELVIS_RIGHT")
            # if is_dummy_joint:
                # quat = accum_quat.inv() * R.from_quat(get_from_to_quat(initial_world_dir[joint_name], to_dir))
                # quat = quat.as_quat()

            parent_quat[joint_name] = R.from_quat(quat)

            euler = R.from_quat(quat).as_euler('xyz', degrees=True)
            
            bvh_motion += f"{euler[2]} {euler[1]} {euler[0]} "

        bvh_motion += "\n"

with open('sc/' + 'output.bvh', 'w') as file:
    file.write(bvh_header + bvh_motion)
