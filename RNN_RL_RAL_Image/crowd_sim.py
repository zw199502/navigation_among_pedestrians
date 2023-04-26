import logging
import matplotlib.pyplot as plt
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
import cv2

class CrowdSim:
    def __init__(self, args):
        self.image_size = args.image_size
        self.square_width = 10.0
        self.harf_area = self.square_width / 2.0
        self.image_resolution = self.square_width / self.image_size
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        
        
        
        self.global_time = None
        self.time_limit = 20
        self.time_step = 0.2
        self.randomize_attributes = False
        self.success_reward = 1.0
        self.collision_penalty = -0.6
        self.outside_reward = -0.1
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = -0.6
        self.goal_distance_factor = 0.8
        self.inflation_grid = [240, 230, 220, 200, 180, 160, 130, 100, 70, 40, 10]
        
       
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 100, 'test': 500}

       
        self.circles = None # human margin
        self.circle_radius = 4.0 # distribution margin
        self.human_num = 5
        self.human_radius = 0.3

        self.humans = []
        self.robot = Robot()
        self.robot.time_step = self.time_step
        self.robot.radius = 0.3
        self.goal_reach_larger_dis = 0.5
        self.goal_reach_larger_area = False

        self.initial_min_goal_dis = self.robot.radius

        self.robot_grid_area, self._robot_grid_raidus = self.occupy_grid(self.robot.radius, inflation=True)
        self.goal_grid_area, self._goal_grid_raidus = self.occupy_grid(self.initial_min_goal_dis)
        self.humans_grid_area = []
        self.humans_grid_raidus = []
        self.posititon_saving = None
        self.current_step = 0
        self.arrival_time = self.time_limit * self.time_step

        self.goal_map = None
        self.image = None

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        print('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            print("Randomize human's radius and preferred speed")
        else:
            print("Not randomize human's radius and preferred speed")
        print('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def occupy_grid(self, radius, inflation=False):
        if inflation:
            radius_up = radius
            grid = int(radius_up / self.image_resolution)
            if radius_up > grid * self.image_resolution:
                grid = grid + 1
                radius_up = grid * self.image_resolution
            inflation_len = len(self.inflation_grid)
            grid_large = grid + inflation_len
            radius_large = grid_large * self.image_resolution
            occ_size = grid_large * 2 + 1
            occupation = np.zeros((occ_size, occ_size), dtype=np.uint8)
            for i in range(grid_large + 1):
                for j in range(grid_large + 1):
                    dy = i * self.image_resolution
                    dx = j * self.image_resolution
                    dis = hypot(dx, dy)
                    if dis <= radius_up:
                        occupation[grid_large - i][grid_large - j] = 255
                        occupation[grid_large + i][grid_large - j] = 255
                        occupation[grid_large + i][grid_large + j] = 255
                        occupation[grid_large - i][grid_large + j] = 255
                    elif dis >= radius_large: 
                        break
                    else:
                        k = int((dis - radius_up) / self.image_resolution)
                        occupation[grid_large - i][grid_large - j] = self.inflation_grid[k]
                        occupation[grid_large + i][grid_large - j] = self.inflation_grid[k]
                        occupation[grid_large + i][grid_large + j] = self.inflation_grid[k]
                        occupation[grid_large - i][grid_large + j] = self.inflation_grid[k]
            return occupation, grid_large

        else:
            radius_up = radius
            grid = int(radius_up / self.image_resolution)
            if radius_up > grid * self.image_resolution:
                grid = grid + 1
                radius_up = grid * self.image_resolution

            occ_size = grid * 2 + 1
            occupation = np.zeros((occ_size, occ_size), dtype=np.uint8)
            for i in range(grid + 1):
                for j in range(grid + 1):
                    dy = i * self.image_resolution
                    dx = j * self.image_resolution
                    dis = hypot(dx, dy)
                    if dis <= radius_up:
                        occupation[grid - i][grid - j] = 255
                        occupation[grid + i][grid - j] = 255
                        occupation[grid + i][grid + j] = 255
                        occupation[grid - i][grid + j] = 255
                    else:
                        break
            return occupation, grid

    def position_to_map(self, px, py):
        # Cartesian coordinate to pixel coordinate
        # Cartesian frame
        """
                                ^ X axis
                                |
                                |
                                |
                                |
                                |
                                |
                                |
                                |
        <-------------------------
        Y axis
        """
        map_h = int((self.harf_area - px) / self.image_resolution)
        map_w = int((self.harf_area - py) / self.image_resolution)
        return map_h, map_w

    def inflation_area(self, px, py, grid_radius):
        map_h, map_w = self.position_to_map(px, py)

        map_up = map_h - grid_radius
        if map_up < 0:
            up = 0
            up_inflation = -map_up
        else:
            up = map_up
            up_inflation = 0

        map_down = map_h + grid_radius
        if map_down >= self.image_size:
            down = self.image_size - 1
            down_inflation = 2 * grid_radius - (map_down - down)
        else:
            down = map_down
            down_inflation = 2 * grid_radius

        map_left = map_w - grid_radius
        if map_left < 0:
            left = 0
            left_inflation = -map_left
        else:
            left = map_left
            left_inflation = 0

        map_right = map_w + grid_radius
        if map_right >= self.image_size:
            right = self.image_size - 1
            right_inflation = 2 * grid_radius - (map_right - right)
        else:
            right = map_right
            right_inflation = 2 * grid_radius
        return up, up_inflation, down, down_inflation, left, left_inflation, right, right_inflation

    def collision_check(self):
        collision_map = self.image[..., 0].astype(np.int32)
        robot_map = self.image[..., 2].astype(np.int32)
        occupation_and_robot_map = collision_map + robot_map
        if np.any(occupation_and_robot_map == (255 * 2)):
            return True, -1
        elif np.all(occupation_and_robot_map <= 255):
            return False, 0
        else:
            discomfort_grid_value = occupation_and_robot_map.max() - 255
            return False, discomfort_grid_value

    def outside_check(self, map_h, map_w, grid_radius, inflation=False):
        if inflation:
            inflation_len = len(self.inflation_grid)
        else:
            inflation_len = 0
        delta_grid = grid_radius - inflation_len
        if map_h - delta_grid < 0 or map_w - delta_grid < 0 or \
            map_h + delta_grid >= self.image_size or map_w + delta_grid >= self.image_size:
            return True
        return False

    def get_image(self):
        occupation_map_all = np.zeros((self.human_num, self.image_size, self.image_size, 1), dtype=np.uint8)
        for i in range(self.human_num):
            human_margin = self.harf_area - self.human_radius + 0.1
            if self.humans[i].px < human_margin and self.humans[i].px > -human_margin \
                and self.humans[i].py < human_margin and self.humans[i].py > -human_margin:  # humans with far location are fake
                up, up_inflation, down, down_inflation, \
                    left, left_inflation, right, right_inflation = \
                    self.inflation_area(self.humans[i].px, self.humans[i].py, self.humans_grid_raidus[i])
                occupation_map_all[i, up:down+1, left:right+1, 0] = \
                    self.humans_grid_area[i][up_inflation:down_inflation+1, left_inflation:right_inflation+1]
        occupation_map_reduce_sum = np.sum(occupation_map_all, axis=0)
        occupation_map_binary = occupation_map_reduce_sum > 0
        occupation_map = occupation_map_binary * 255
        occupation_map = occupation_map.astype(np.uint8)

        robot_map = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        up, up_inflation, down, down_inflation, \
            left, left_inflation, right, right_inflation = \
            self.inflation_area(self.robot.px, self.robot.py, self._robot_grid_raidus)
        robot_map[up:down+1, left:right+1, 0] = \
            self.robot_grid_area[up_inflation:down_inflation+1, left_inflation:right_inflation+1]

        # rgb image, occupation map is red, goal map is green, robot map is blue
        # OpenCV is BGR
        goal_map_zero = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        # return np.concatenate((occupation_map, self._goal_map, robot_map), axis=-1) # image without goal
        return np.concatenate((occupation_map, goal_map_zero, robot_map), axis=-1)

        
    def generate_random_human_position(self):
        # initial min separation distance to avoid danger penalty at beginning
        self.humans = []
        for i in range(self.human_num):
            self.humans.append(self.generate_circle_crossing_human())
            grid_area, grid_radius = self.occupy_grid(self.humans[i].radius)
            self.humans_grid_area.append(grid_area)
            self.humans_grid_raidus.append(grid_radius)

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
        self.log_env = {}
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                            'val': 0, 'test': self.case_capacity['val']}
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        
        # if self.case_counter[phase] >= 0:
        #     # for every training/valuation/test, generate same initial human states
        #     np.random.seed(counter_offset[phase] + self.case_counter[phase])
        #     self.generate_random_human_position()
    
        #     # case_counter is always between 0 and case_size[phase]
        #     self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        # else:
        #     raise NotImplementedError
        self.generate_random_human_position()

        self._goal_map = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        up, up_inflation, down, down_inflation, \
            left, left_inflation, right, right_inflation = \
            self.inflation_area(self.robot.gx, self.robot.gy, self._goal_grid_raidus)
        self._goal_map[up:down+1, left:right+1, 0] = \
            self.goal_grid_area[up_inflation:down_inflation+1, left_inflation:right_inflation+1]
        
        self.image = self.get_image()
        
        self.arrival_time = self.time_limit * self.time_step
        self.goal_reach_larger_area = False

        # get the observation
        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)
        self_state = FullState(self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius, \
                               self.robot.gx, self.robot.gy, self.robot.v_pref, self.robot.theta)
        ob_state = [human.get_observable_state() for human in self.humans]
        ob_coordinate = JointState(self_state, ob_state)
        self.log_env['robot'] = [np.array([self.robot.px, self.robot.py])]
        self.log_env['goal'] = [np.array([self.robot.gx, self.robot.gy])]
        humans_position = []
        for human in self.humans:
            humans_position.append(np.array([human.px, human.py]))
        self.log_env['humans'] = [np.array(humans_position)]
      
        return (self.image.flatten() / 255.0 - 0.5)

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
        self.image = self.get_image()
        self.global_time += self.time_step

        robot_map_h, robot_map_w = self.position_to_map(robot_x, robot_y)
        collide, discomfort_grid_value = self.collision_check()
        outside = self.outside_check(robot_map_h, robot_map_w, self._robot_grid_raidus, inflation=True)

        goal_map_h, goal_map_w = self.position_to_map(self.robot.gx, self.robot.gy)
        goal_dis_map = hypot(goal_map_h - robot_map_h, goal_map_w - robot_map_w) * self.image_resolution
        goal_reach = (goal_dis_map <= self.initial_min_goal_dis) 
        goal_dis_real = self.robot.get_goal_distance()
        goal_reach_larger_area = goal_dis_real < self.goal_reach_larger_dis

        if goal_reach_larger_area and (not self.goal_reach_larger_area):
            self.goal_reach_larger_area = True
            self.arrival_time = self.global_time
            print('goal reaching on larger area at time: ', self.arrival_time)
        

        # collision detection between humans
        for i in range(self.human_num):
            for j in range(i + 1, self.human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        reward = 0.0
        done = False
        if goal_reach:
            reward = 1.0
            info = ReachGoal()
            done = True
        elif collide:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif outside:
            reward = self.outside_reward
            done = True
            info = Collision()
        else:
            reward = self.discomfort_penalty_factor * discomfort_grid_value / 255.0 + \
                self.goal_distance_factor * (1.0 - goal_dis_map / self.square_width)
            done = False
            info = Nothing()

        if self.global_time >= self.time_limit:
            done = True
            info = Timeout()

        if self.goal_reach_larger_area:
            info = ReachGoal()
  
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
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)
        self_state = FullState(self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius, \
                               self.robot.gx, self.robot.gy, self.robot.v_pref, self.robot.theta)
        ob_state = [human.get_observable_state() for human in self.humans]
        ob_coordinate = JointState(self_state, ob_state)
        self.log_env['robot'].append(np.array([self.robot.px, self.robot.py]))
        self.log_env['goal'].append(np.array([self.robot.gx, self.robot.gy])) 
        humans_position = []
        for human in self.humans:
            humans_position.append(np.array([human.px, human.py]))
        self.log_env['humans'].append(np.array(humans_position)) 
        
        return (self.image.flatten() / 255.0 - 0.5), reward, done, info

    def render(self):
        cv2.imshow('image', self.image)
        cv2.waitKey(1)
            