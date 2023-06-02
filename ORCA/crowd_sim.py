import logging
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from matplotlib import collections as mc
from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.state import *
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time

# laser scan parameters
# number of all laser beams
n_laser = 1800
laser_angle_resolute = 0.003490659
laser_min_range = 0.27
laser_max_range = 6.0

# environment size
square_width = 10.0
# environment size


class CrowdSim:
    def __init__(self):
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.goal_distance_factor = None

        # last-time distance from the robot to the goal
        self.goal_distance_last = None

        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.circles = None # human margin
        self.circle_radius = None
        self.human_num = None


        plt.ion()
        plt.show()
        self.count = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
    def configure(self):
        self.time_limit = 100
        self.time_step = 0.2
        self.randomize_attributes = False
        self.success_reward = 1.0
        self.collision_penalty = -1.0
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5
        self.goal_distance_factor = 0.01
       
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 100, 'test': 500}

        self.circle_radius = 4.0
        self.human_num = 5

        self.robot = Robot()
        self.robot.time_step = self.time_step

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        print('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            print("Randomize human's radius and preferred speed")
        else:
            print("Not randomize human's radius and preferred speed")
        print('Square width: {}, circle width: {}'.format(square_width, self.circle_radius))
        
    def generate_random_human_position(self):
        # initial min separation distance to avoid danger penalty at beginning
        self.humans = []
        for i in range(self.human_num):
            self.humans.append(self.generate_circle_crossing_human())

        for i in range(len(self.humans)):
            human_policy = policy_factory[self.human_policy_name]()
            human_policy.time_step = self.time_step
            self.humans[i].set_policy(human_policy)

    def generate_circle_crossing_human(self):
        human = Human()
        human.time_step = self.time_step

        if self.randomize_attributes:
            # Sample agent radius and v_pref attribute from certain distribution
            human.sample_random_attributes()
        else:
            human.radius = 0.3
            human.v_pref = 1.0
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def reset(self, phase='test'):
        assert phase in ['train', 'val', 'test']
        self.global_time = 0

        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                            'val': 0, 'test': self.case_capacity['val']}
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        self.goal_distance_last = self.robot.get_goal_distance()
        
        if self.case_counter[phase] >= 0:
            # for every training/valuation/test, generate same initial human states
            np.random.seed(counter_offset[phase] + self.case_counter[phase])
            self.generate_random_human_position()
    
            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError

        # get the observation
        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)
        self_state = FullState(self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius, \
                               self.robot.gx, self.robot.gy, self.robot.v_pref, self.robot.theta)
        ob_state = [human.get_observable_state() for human in self.humans]
        ob_coordinate = JointState(self_state, ob_state)
        return ob_position, ob_coordinate

    def step(self, action):
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            human_actions.append(human.act(ob))

        # uodate states
        robot_x, robot_y, robot_theta = self.robot.compute_pose(action)
        self.robot.update_states(robot_x, robot_y, robot_theta, action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].update_states(human_action)

        # get new laser scan and grid map
        self.global_time += self.time_step
        
        # if reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        reaching_goal = goal_dist < self.robot.radius

        # collision detection between humans
        for i in range(self.human_num):
            for j in range(i + 1, self.human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # collision detection between the robot and humans
        collision = False
        dist_robot = self.discomfort_dist
        for i in range(self.human_num):
            dx = self.humans[i].px - self.robot.px
            dy = self.humans[i].py - self.robot.py
            dist_robot = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.robot.radius
            if dist_robot < 0:
                collision = True

        reward = 0
        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif (dist_robot < self.discomfort_dist):
            # penalize agent for getting too close 
            reward = (dist_robot - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dist_robot)
        else:
            reward = 0
            done = False
            info = Nothing()

        if reaching_goal:
            reward = reward + self.success_reward
            done = True
            info = ReachGoal()
        else:
            reward = reward + self.goal_distance_factor * (self.goal_distance_last - goal_dist)
        self.goal_distance_last = goal_dist
  
        for i, human in enumerate(self.humans):
            # let humans move circularly from two points
            if human.reached_destination():
                self.humans[i].gx = -self.humans[i].gx
                self.humans[i].gy = -self.humans[i].gy

        # get the observation
        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)
        self_state = FullState(self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius, \
                               self.robot.gx, self.robot.gy, self.robot.v_pref, self.robot.theta)
        ob_state = [human.get_observable_state() for human in self.humans]
        ob_coordinate = JointState(self_state, ob_state)
        return ob_position, ob_coordinate, reward, done, info

    def render(self):
        self.ax.set_xlim(-5.0, 5.0)
        self.ax.set_ylim(-5.0, 5.0)
        for human in self.humans:
            human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
            self.ax.add_artist(human_circle)
        self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
        self.ax.add_artist(plt.Circle((self.robot.gx, self.robot.gy), self.robot.radius, fill=True, color='g'))
        plt.text(-4.5, -4.5, str(round(self.global_time, 2)), fontsize=20)
        # x, y, theta = self.robot.px, self.robot.py, self.robot.theta
        # dx = cos(theta)
        # dy = sin(theta)
        # self.ax.arrow(x, y, dx, dy,
        #     width=0.01,
        #     length_includes_head=True, 
        #     head_width=0.15,
        #     head_length=1,
        #     fc='r',
        #     ec='r')
        
        # if self.count == 3:
        #     self.ax.add_artist(plt.Circle((0.0, 4.0), 0.3, fill=True, color='g'))
        #     plt.savefig('fig_goal.png')
        self.count = self.count + 1
        plt.draw()
        plt.pause(0.001)
        plt.cla()
