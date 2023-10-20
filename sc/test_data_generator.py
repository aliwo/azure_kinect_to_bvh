import os
import math
from operator import add
# Joint 0: Position(268.236115, -157.712402, 2427.859375)

joint_count = 3
pelvis_pos = [100, -100, 1000]

sub_path = ''
circle_radius = 200
theta = 60

frame_count = 60

for frame_num in range(frame_count):
    file_name = f'frame{frame_num}_body0_timestamp{frame_num * 10}_1_joints.txt'
    file_cont = ""

    theta_for_this_frame = theta / frame_count * (frame_num+1)
    rad = math.radians(theta_for_this_frame)
    for joint_index in range(joint_count):
        pos = pelvis_pos

        rot = [math.cos(rad), math.sin(rad), 0] #z-axis
        # rot = [0, math.cos(rad), math.sin(rad)] #x-axis
        # rot = [math.cos(rad), math.sin(rad), 0] #z-axis
        rot = [x * circle_radius for x in rot]

        second_rot = [math.cos(-rad), math.sin(-rad), 0]
        second_rot = [x * circle_radius for x in second_rot]

        if joint_index is 0:
            # pelvis_offset = [0, 0, theta_for_this_frame]
            pelvis_offset = [0, 0, 0]
            pos = list(map(add, pos, pelvis_offset))

        elif joint_index is 1:
            pos = list(map(add, pos, rot))

        elif joint_index is 2:
            pos = list(map(add, pos, rot))
            pos = list(map(add, pos, second_rot))

        file_cont += f"Joint {joint_index}: Position({pos[0]}, {pos[1]}, {pos[2]})"

        if joint_index is not joint_count - 1:
            file_cont += "\n"

    with open(sub_path + file_name, 'w+') as file:
        file.write(file_cont)