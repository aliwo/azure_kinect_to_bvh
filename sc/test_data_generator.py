import os
import math
import numpy as np
from operator import add
# Joint 0: Position(268.236115, -157.712402, 2427.859375)

joint_count = 9
pelvis_pos = [100, -100, 1000]

# sub_path = '/sc/multi_joint/'
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

        rot_a = [math.cos(rad), math.sin(rad), 0] #
        rot_b = [0, math.sin(rad), math.cos(rad)] #
        rot_c = [math.cos(math.pi - rad), math.sin(math.pi - rad), 0] #
        
        rot_a = [x * circle_radius for x in rot_a]
        rot_b = [x * circle_radius for x in rot_b]
        rot_c = [x * circle_radius for x in rot_c]

        second_rot_a = [math.cos(-rad), math.sin(-rad), 0]
        second_rot_a = [x * circle_radius for x in second_rot_a]

        second_rot_b = [0, math.sin(-rad), math.cos(-rad)]
        second_rot_b = [x * circle_radius for x in second_rot_b]

        second_rot_c = [math.cos(rad - math.pi), math.sin(rad - math.pi), 0]
        second_rot_c = [x * circle_radius for x in second_rot_c]

        if joint_index is 0:
            # pelvis_offset = [0, 0, theta_for_this_frame]
            pelvis_offset = [0, 0, 0]
            pos = list(map(add, pos, pelvis_offset))

        elif joint_index is 1:
            pos = list(map(add, pos, rot_a))

        elif joint_index is 2:
            pos = list(map(add, pos, rot_a))
            pos = list(map(add, pos, second_rot_a))

        elif joint_index is 3: #B_Pelvis
            pelvis_offset = [0, 0, 2]
            pos = list(map(add, pos, pelvis_offset))

        elif joint_index is 4:
            pos = list(map(add, pos, rot_b))
        
        elif joint_index is 5:
            pos = list(map(add, pos, rot_b))
            pos = list(map(add, pos, second_rot_b))

        elif joint_index is 6: #C_Pelvis
            pelvis_offset = [-2, 0, 0]
            pos = list(map(add, pos, pelvis_offset))

        elif joint_index is 7:
            pos = list(map(add, pos, rot_c))
        
        elif joint_index is 8:
            pos = list(map(add, pos, rot_c))
            pos = list(map(add, pos, second_rot_c))
        
        y_rad = math.radians(30)
        y_axis_rot = [ 
                    [math.cos(y_rad), 0, math.sin(y_rad)],
                    [0, 1, 0],
                    [-math.sin(y_rad), 0, math.cos(y_rad)]
                    ]
        pos = np.dot(y_axis_rot, pos)
        file_cont += f"Joint {joint_index}: Position({pos[0]}, {pos[1]}, {pos[2]})"

        if joint_index is not joint_count - 1:
            file_cont += "\n"

    with open(sub_path + file_name, 'w+') as file:
        file.write(file_cont)