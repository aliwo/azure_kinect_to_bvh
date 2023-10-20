import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import re
import numpy as np

def read_files(directory):
    files = sorted([f for f in os.listdir(directory) if f.endswith('_joints.txt')], key=lambda x: int(re.search(r'frame(\d+)_', x).group(1)))
    joint_data = []
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            lines = f.readlines()
            frame_data = []
            for i in range(0, 26):
            # for i in [0, 18, 19, 20, 21, 22, 23, 24, 25]:
                line = lines[i]
                match = re.search(r'Position\(([^)]+)\)', line)
                if match:
                    position = tuple(map(float, match.group(1).split(', ')))
                    frame_data.append(position)
            joint_data.append(frame_data)
    return joint_data

def update(frame):
    ax.cla()  
    selected_frame = frame * 10
    if selected_frame >= len(joint_data):
        selected_frame = len(joint_data) - 1
    x, y, z = zip(*joint_data[selected_frame])

    pelvis_pos = joint_data[selected_frame][0]

    x = tuple(map(lambda i: i - pelvis_pos[0], x))
    y = tuple(map(lambda i: i - pelvis_pos[1], y))
    z = tuple(map(lambda i: i - pelvis_pos[2], z))

    # theta = np.pi / 2  # 90도 회전 (라디안 단위)
    # rotation_matrix = np.array([
    # [1, 0, 0],
    # [0, np.cos(theta), -np.sin(theta)],
    # [0, np.sin(theta), np.cos(theta)]
    # ])
    # rotated_x = tuple(map(lambda i, j, k: np.dot(rotation_matrix, [i,j,k])[0], x, y, z))
    # rotated_y = tuple(map(lambda i, j, k: np.dot(rotation_matrix, [i,j,k])[1], x, y, z))
    # rotated_z = tuple(map(lambda i, j, k: np.dot(rotation_matrix, [i,j,k])[2], x, y, z))

    ax.scatter(x, y, z)
    # ax.scatter(rotated_x, rotated_y, rotated_z)
    ax.set_xlim([-800, 800])
    ax.set_ylim([-800, 800])
    ax.set_zlim([-800, 800])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

directory = 'new_3d_dychair1_1000frame'  
joint_data = read_files(directory)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(90, -90)

ani = animation.FuncAnimation(fig, update, frames=len(joint_data), interval=1)  # interval 인자 추가
plt.show()
