import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import glob
import time

n_laser = 1800

file_name = sorted(glob.glob('evaluation_episodes/*.npz'))
# print(file_name[-1])
log_env = np.load(file_name[-1])

robot = log_env['robot']
steps = robot.shape[0]
# print(steps)
humans = log_env['humans']
laser = log_env['laser']
laser_beam = laser.shape[1]
human_num = humans.shape[1]
goal = log_env['goal']
radius = 0.3

plt.ion()
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(steps):
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    scan_intersection = []
    for laser_i in range(laser_beam):
        scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])
    for human_i in range(human_num):
        human_circle = plt.Circle(humans[i][human_i], radius, fill=False, color='b')
        ax.add_artist(human_circle)
    ax.add_artist(plt.Circle(robot[i], radius, fill=True, color='r'))
    ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))
    plt.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)

    ii = 0
    lines = []
    while ii < n_laser:
        lines.append(scan_intersection[ii])
        ii = ii + 36
    lc = mc.LineCollection(lines)
    ax.add_collection(lc)
    plt.draw()
    plt.pause(0.001)
    plt.cla()
    time.sleep(0.2)


# metadata = dict(title='EGO', artist='Matplotlib',comment='EGO test')
# writer = FFMpegWriter(fps=5, metadata=metadata)

# fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
# ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
# plt.tick_params(labelsize=24)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname("Times New Roman") for label in labels]

 
# ax.set_xlim(-5.0, 5.0)
# ax.set_ylim(-5.0, 5.0)
 
 
# with writer.saving(fig, "ego.mp4", 100):
#     for i in range(steps):
#         ax.clear()
#         ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
#         ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
#         plt.tick_params(labelsize=24)
#         labels = ax.get_xticklabels() + ax.get_yticklabels()
#         [label.set_fontname("Times New Roman") for label in labels]

        
#         ax.set_xlim(-5.0, 5.0)
#         ax.set_ylim(-5.0, 5.0)
#         for human_i in range(human_num):
#             human_circle = plt.Circle(humans[i][human_i], radius, fill=False, color='b')
#             ax.add_artist(human_circle)
#         ax.add_artist(plt.Circle(robot[i], radius, fill=True, color='r'))
#         ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))

#         scan_intersection = []
#         for laser_i in range(laser_beam):
#             scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])
#         ii = 0
#         lines = []
#         while ii < n_laser:
#             lines.append(scan_intersection[ii])
#             ii = ii + 36
#         lc = mc.LineCollection(lines)
#         ax.add_collection(lc)

#         ax.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)
#         writer.grab_frame()