import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
import re
from scipy.spatial.transform import Rotation
import numpy as np


"""
V1

모든 값에 대해서 자신의 바로 위 부모값에 대해 상대계산을 하는 버전

1 - joint_name 을 바꾼것 [v]
2 - ORDERS를 바꿀 것
3 - joint_name와 orders를 맞출것 [v]

"""
PATH_FRAMES = 'new_3d_dysf1_1000frame/'
ROOT_JOINT_NAME = 'PELVIS'
TAB = '    '
OFFSET_ORDER = [0, 1, 2]
MOTION_ORDER = [0, 1, 2]
EULER_ORDER = [0, 1, 2]
JOINTS_ORDER = [
    'PELVIS',
    'SPINE_NAVAL',
    'SPINE_CHEST',
    'NECK',
    'HEAD',
    'NOSE',
    'EYE_LEFT',
    'EAR_LEFT',
    'EYE_RIGHT',
    'EAR_RIGHT',
    'CLAVICLE_LEFT',
    'SHOULDER_LEFT',
    'ELBOW_LEFT',
    'WRIST_LEFT',
    'HAND_LEFT',
    'HANDTIP_LEFT',
    'THUMB_LEFT',
    'CLAVICLE_RIGHT',
    'SHOULDER_RIGHT',
    'ELBOW_RIGHT',
    'WRIST_RIGHT',
    'HAND_RIGHT',
    'HANDTIP_RIGHT',
    'THUMB_RIGHT',
    'HIP_LEFT',
    'KNEE_LEFT',
    'ANKLE_LEFT',
    'FOOT_LEFT',
    'HIP_RIGHT',
    'KNEE_RIGHT',
    'ANKLE_RIGHT',
    'FOOT_RIGHT'
]


class Joint ():
    def __init__(self, index, name, position, parent_name, quat):
        self.index = index
        self.name = name
        self.position = position
        self.parent_name = parent_name
        self.quat = quat
        self.world_dir = None
        self.offset = None

    def set_offset(self, offset):
        self.offset = offset

    def set_world_dir(self, world_dir):
        self.world_dir = world_dir


def get_joint_names():
    with open('joint_structure.txt', 'r') as file:
        joint_data = file.read().split('\n')
        joint_data.pop(0)

    return [line.split('	') for line in joint_data]


def calc_world_dir(joints):
    for name in joints.keys():
        joint = joints[name]
        children = get_children_joints(joints=joints, parent_name=name)

        if len(children) == 0:
            joint.set_world_dir(np.array([0.0, 0.0, 0.0]))
            continue

        world_dir = children[0].position - joint.position
        world_dir /= np.linalg.norm(world_dir)
        joint.set_world_dir(world_dir=world_dir)


def compute_offset(joints, base_joints):
    pelvis = joints[ROOT_JOINT_NAME]
    if base_joints is None:
        pelvis.set_offset(np.array([0, 0, 0]))
    else:
        pelvis.set_offset(pelvis.position -
                          base_joints[ROOT_JOINT_NAME].position)

    for name in JOINTS_ORDER:
        if name == "PELVIS":
            continue

        curr_joint = joints[name]
        parent_joint = joints[curr_joint.parent_name]
        base_joint = base_joints[name] if base_joints is not None else pelvis

        adj_pos = curr_joint.position - pelvis.position
        adj_pr_pos = parent_joint.position - pelvis.position

        curr_joint.set_offset(adj_pos - adj_pr_pos)


def _parse_frame_data(frame_file, joint_names):
    joints = {}
    with open(f'{PATH_FRAMES}{frame_file}', 'r') as file:
        joint_data = file.read().split('\n')
    for line in joint_data:
        if re.search(r'Joint ([0-9]+)', line) is None:
            continue
        idx = re.search(r'Joint ([0-9]+)', line).group(1)
        position = re.search(r'Position\(([^)]+)\)', line).group(1)
        quat = re.search(r'Orientation\(([^)]+)\)', line).group(1)
        index, joint_name, parent_name = joint_names[int(idx)]

        joints[joint_name] = Joint(
            index=index,
            name=joint_name,
            parent_name=parent_name,
            position=np.array([float(x)/10 for x in position.split(',')]),
            quat=np.array([float(x) for x in quat.split(',')])
        )
    return joints


def get_base_joint_frame_data(frame_file, joint_names):
    joints = _parse_frame_data(frame_file=frame_file, joint_names=joint_names)
    compute_offset(joints=joints)
    calc_world_dir(joints=joints)
    return joints


def get_joint_frame_data(frame_file, joint_names, base_joints):
    joints = _parse_frame_data(frame_file=frame_file, joint_names=joint_names)
    compute_offset(joints=joints, base_joints=base_joints)
    calc_world_dir(joints=joints)
    return joints


def get_children_joints(joints, parent_name):
    return [joints[name] for name in joints.keys() if joints[name].parent_name == parent_name]


def write_bvh_child_joint(name: , joints: , depth=1):
    joint = joints[name]
    joint_infos = "\n".join([write_bvh_child_joint(name=joint.name, joints=joints, depth=depth+1)
                            for joint in get_children_joints(joints=joints, parent_name=name)])
    return '\n'.join([
        f'{TAB * depth}JOINT {name}',
        f'{TAB * depth}{"{"}',
        f'{TAB * (depth + 1)}OFFSET {joint.offset[OFFSET_ORDER[0]]} {joint.offset[OFFSET_ORDER[1]]} {joint.offset[OFFSET_ORDER[2]]}',
        # f'{TAB * (depth + 1)}CHANNELS 3 Zrotation Xrotation Yrotation',
        f'{TAB * (depth + 1)}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation',
        f'{joint_infos}',
        f'{TAB * depth}{"}"}',
    ])


def write_bvh_joint_structure(joints):
    pelvis = joints[ROOT_JOINT_NAME]
    joint_infos = "\n".join([write_bvh_child_joint(name=joint.name, joints=joints)
                            for joint in get_children_joints(joints=joints, parent_name=ROOT_JOINT_NAME)])
    return '\n'.join([
        f'HIERARCHY',
        f'ROOT {ROOT_JOINT_NAME}',
        '{',
        f'{TAB}OFFSET {pelvis.offset[OFFSET_ORDER[0]]} {pelvis.offset[OFFSET_ORDER[1]]} {pelvis.offset[OFFSET_ORDER[2]]}',
        f'{TAB}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation',
        f'{joint_infos}',
        '}',
    ])


def write_bvh_motion_info(frame_count: int):
    return '\n'.join([
        'MOTION',
        f'Frames: {frame_count}',
        f'Frame Time: 0.01',
    ])


def calc_euler(matrix):
    return Rotation.from_matrix(
        matrix).as_euler('xyz', degrees=True)


def norm(v):
    return v / np.linalg.norm(v)


def calc_rel_euler(m, v):
    return Rotation.from_matrix(m).apply(norm(v))


def calc_outer(v1, v2):
    return np.outer(norm(v1), norm(v2))


def calc_dist(cv, pv):
    return np.linalg.norm(cv - np.array([0, 0, 0])) - np.linalg.norm(pv - np.array([0, 0, 0]))


def write_bvh_motion_first(joint_names):
    motions = []
    for _ in joint_names:
        motions.append('0 0 0')
        motions.append('0 0 0')
    return ' '.join(motions)


def write_bvh_motion(joint_names, base_joints, curr_joints, prev_joints):
    motions = []
    parent_matrix = None
    for name in JOINTS_ORDER:
        if name == ROOT_JOINT_NAME:
            motions.append('0 0 0')
            continue

        base = base_joints[name]
        curr = curr_joints[name]

        matrix = parent_matrix
        if matrix is None:
            matrix = calc_outer(norm(base.offset), norm(curr.offset))
        else:
            matrix = calc_outer(norm(base.offset), calc_rel_euler(
                matrix, curr.offset))

        x, y, z = calc_euler(matrix)
        motions.append(f'{z} {x} {y}')

        dist = calc_dist(curr.offset, base.offset)
        motions.append(f'{0} {dist} {0}')

        children = get_children_joints(curr_joints, name)
        if len(children) == 0:
            parent_matrix = None
        else:
            parent_matrix = matrix
    motions.append('0 0 0')
    return ' '.join(motions)

# 현재 프레임 pelvis의 모션값을 반환


def calc_curr_frame_pelvis_motion(joints):
    return f"{joints[ROOT_JOINT_NAME].position[EULER_ORDER[0]]} {joints[ROOT_JOINT_NAME].position[EULER_ORDER[1]]} {joints[ROOT_JOINT_NAME].position[EULER_ORDER[2]]}"

# 현재 프레임 특정 조인트의 모션값을 반환


def calc_curr_frame_joint_motion(name, base_joints, curr_joints):
    from_dir = base_joints[name].world_dir
    to_dir = curr_joints[name].world_dir

    if not len(get_children_joints(joints=curr_joints, parent_name=name)):
        return "0 0 0"

    if np.dot(from_dir, to_dir) >= 1 - 0.00001:
        return "0 0 0"

    quat = np.cross(from_dir, to_dir)
    quat = np.append(quat, ((np.linalg.norm(from_dir) ** 2) *
                     (np.linalg.norm(to_dir) ** 2)) ** 0.5 + np.dot(from_dir, to_dir))
    # 설명해서 normalize 시켜준다함. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
    euler = Rotation.from_quat(quat).as_euler('yzx', degrees=True)
    return f"{euler[EULER_ORDER[0]]} {euler[EULER_ORDER[1]]} {euler[EULER_ORDER[2]]}"

# frame에 대해 모션 부분을 제작한다


# get frame file list in PATH_FRAMES
def get_frame_list() -> dict[int, str]:
    return {int(re.search(r'frame([0-9]+)', filename).group(1)): filename for filename in os.listdir(PATH_FRAMES)}


def print_plot(offset, frames):
    def update(frame):
        ax.cla()
        x, y, z = zip(*offsets[frame])
        ax.scatter(x, y, z)
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-30, 30])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=50)  # interval 인자 추가
    plt.show()


def main():
    frames = get_frame_list()
    frame_keys = sorted(frames.keys())

    joint_names = get_joint_names()

    base_joints = get_base_joint_frame_data(
        frame_file=frames[frame_keys[0]],
        joint_names=joint_names
    )
    prev_joints = base_joints
    # for only 2 frame
    # offsets = []
    # with open('output.bvh', 'w') as f:
    #     content = write_bvh_joint_structure(joints=base_joints)
    #     content += '\n' + write_bvh_motion_info(frame_count=2)
    #     # content += '\n' + write_bvh_base_motion(joint_names=joint_names)
    #     for i in range(2):
    #         key = frame_keys[i]
    #         curr_joints = get_joint_frame_data(
    #             frame_file=frames[key],
    #             joint_names=joint_names
    #         )
    #         # 3d test를 위해
    #         frame = []
    #         for _, name, _ in joint_names:
    #             frame.append(tuple(curr_joints[name].offset))
    #         offsets.append(frame)
    #         content += '\n' + \
    #             write_bvh_motion(
    #                 joint_names=joint_names,
    #                 base_joints=base_joints,
    #                 curr_joints=curr_joints
    #             )
    #     f.write(content)
    #     f.close()

    # return offsets, 2

    offsets = []
    with open('output.bvh', 'w') as f:
        content = write_bvh_joint_structure(joints=base_joints)
        content += '\n' + write_bvh_motion_info(frame_count=len(frame_keys))
        for key in frame_keys:
            if key == frame_keys[0]:
                content += '\n' + \
                    write_bvh_motion_first(
                        joint_names=joint_names,
                    )
                continue
            curr_joints = get_joint_frame_data(
                frame_file=frames[key],
                joint_names=joint_names,
                base_joints=base_joints
            )
            # START: 3d test를 위해
            frame = []
            for _, name, _ in joint_names:
                frame.append(tuple(curr_joints[name].offset))
            offsets.append(frame)
            # END
            content += '\n' + \
                write_bvh_motion(
                    joint_names=joint_names,
                    base_joints=base_joints,
                    curr_joints=curr_joints,
                    prev_joints=prev_joints
                )
            prev_joints = curr_joints
        f.write(content)
        f.close()

    # print_plot(offsets, len(frame_keys))


main()
