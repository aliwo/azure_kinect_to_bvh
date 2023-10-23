import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import re

def read_files(directory):
    files = sorted([f for f in os.listdir(directory) if f.endswith('_joints.txt')], key=lambda x: int(re.search(r'frame(\d+)_', x).group(1)))
    joint_data = []
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            lines = f.readlines()
            frame_data = []
            for line in lines:
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
    ax.scatter(x, y, z)
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([0, 3000])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

directory = 'new_3d_dysf1_1000frame'  
joint_data = read_files(directory)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ani = animation.FuncAnimation(fig, update, frames=len(joint_data)//10, interval=0.000001, repeat=True)
plt.show()