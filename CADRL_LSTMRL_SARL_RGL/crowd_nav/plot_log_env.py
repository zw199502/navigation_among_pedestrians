import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

file_name = 'test_log/cadrl3.npz'
log_env = np.load(file_name)

robot = log_env['robot']
steps = robot.shape[0]
print('steps: ', steps)
humans = log_env['humans']
human_num = humans.shape[1]
goal = log_env['goal']
radius = 0.3

# plt.ion()
# plt.show()
# fig, ax = plt.subplots(figsize=(10, 10))
# for i in range(steps):
#     ax.set_xlim(-5.0, 5.0)
#     ax.set_ylim(-5.0, 5.0)
#     for human_i in range(human_num):
#         human_circle = plt.Circle(humans[i][human_i], radius, fill=False, color='b')
#         ax.add_artist(human_circle)
#     ax.add_artist(plt.Circle(robot[i], radius, fill=True, color='r'))
#     ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))
#     plt.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)


#     plt.draw()
#     plt.pause(0.001)
#     plt.cla()

metadata = dict(title='cadrl', artist='Matplotlib',comment='cadrl test')
writer = FFMpegWriter(fps=5, metadata=metadata)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
plt.tick_params(labelsize=24)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]

 
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
 
 
with writer.saving(fig, "cadrl.mp4", 100):
    for i in range(steps):
        ax.clear()
        ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
        ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
        plt.tick_params(labelsize=24)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-5.0, 5.0)
        for human_i in range(human_num):
            human_circle = plt.Circle(humans[i][human_i], radius, fill=False, color='b')
            ax.add_artist(human_circle)
        ax.add_artist(plt.Circle(robot[i], radius, fill=True, color='r'))
        ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))
        ax.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)
        writer.grab_frame()
