import gym
import numpy as np
import common
from math import hypot, asin, atan2, cos, sin
import cv2
import time
# import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist 
from threading import Lock

###
# real time for one step: from 1ms to 5ms
###
class Navigation:

  def __init__(self, config, mode):
    assert config.render_size[0] == config.render_size[1]
    self._mode = mode # train or eval
    self._size = config.render_size
    self._initial_min_goal_dis = config.initial_min_goal_dis
    self._human_num_max = config.human_num_max
    self._goal_reward_factor = config.goal_reward_factor
    self._collision_reward_factor = config.collision_reward_factor
    self._randomize_attributes = config.randomize_attributes
    self._max_human_speed = config.max_human_speed
    self._max_robot_speed = config.max_robot_speed
    self._inflation_grid = config.inflation_grid
    self._time_step = config.time_step
    self._time_limit = config.time_limit
    self._random_robot = config.random_robot
    self._motion_area = config.motion_area
    self._random_human_num = config.random_human_num
    self._outside_reward = config.outside_reward
    self._collision_reward = config.collision_reward
    self._goal_reach_larger_dis = config.goal_reach_larger_dis
    self._scenario = config.scenario
    self._human_radius = config.human_radius
    self._robot_radius = config.robot_radius
    self._real_world = config.real_world
    self._goal_position = config.goal_position

    self._map_resolution = self._motion_area / self._size[0]
    self._harf_area = self._motion_area / 2.0
    if self._scenario == 'comparison':
      # the motion area is bigger and the maximum speed is larger
      # this scenario is used for comparing with ORCA, CADRL, RGL, etc.
      self._circle_radius = self._harf_area - 1.0
    else:
      self._circle_radius = self._harf_area - 0.5

    if self._real_world:
      self._lock_robot_pose = Lock()
      self._lock_position_person1 = Lock()
      self._lock_position_person2 = Lock()
      self._lock_position_person3 = Lock()
      self._robot_pose = np.zeros(3) # x, y, yaw
      self._position_person1 = 999.9 * np.ones(2)
      self._position_person2 = 999.9 * np.ones(2)
      self._position_person3 = 999.9 * np.ones(2)
      self._pub_cmd_vel = rospy.Publisher('cmd_vel', Twist)
      if self._scenario == 'quadruped_motion_capture':
        # get pose from the motion capture system
        sub_robot_pose = ("/vrpn_client_node/QuadrupedRobot/pose", PoseStamped, self.robot_pose_callback)
        sub_person1_pose = ("/vrpn_client_node/person1/pose", PoseStamped, self.person1_pose_callback)
        sub_person2_pose = ("/vrpn_client_node/person2/pose", PoseStamped, self.person2_pose_callback)
        sub_person3_pose = ("/vrpn_client_node/person3/pose", PoseStamped, self.person3_pose_callback)
      elif self._scenario == 'quadruped_loam':
        # get pose from the LOAM SLAM algorithm
        sub_robot_pose = rospy.Subscriber('/robot/pose', PoseStamped, self.robot_pose_callback)
        # get human pose according to the clusters of point cloud
        sub_person1_pose = ("/person1/pose", PoseStamped, self.person1_pose_callback)
        sub_person2_pose = ("/person2/pose", PoseStamped, self.person2_pose_callback)
        sub_person3_pose = ("/person3/pose", PoseStamped, self.person3_pose_callback)
      else:
        raise NotImplementedError(self._scenario)
    
    self._robot = common.Robot(self._robot_radius, self._time_step)
    self._robot_grid_area, self._robot_grid_raidus = self.occupy_grid(self._robot.radius, inflation=True)
    self._goal_grid_area, self._goal_grid_raidus = self.occupy_grid(self._initial_min_goal_dis)
    self._humans = []
    self._humans_grid_area = []
    self._humans_grid_raidus = []
    self._posititon_saving = None
    self._current_step = 0

    self._human_num = None
    self._goal_map = None
    self._image = None
    self._goal_reach_larger_area = False

    if self._real_world:
      # sleep for a while to subscribe initial pose message
      time.sleep(1.0)

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, (self._size[0], self._size[1], 3), dtype=np.uint8),
        'reward': gym.spaces.Box(-1.0, 1.0, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
    }
    return spaces

  @property
  def act_space(self):
    action = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    return {'action': action}

  def cal_yaw_from_quaternion(self, q_0, q_1, q_2, q_3):
    # roll = atan2(2.0 * (q_0 * q_1 + q_2 * q_3), 1.0 - 2.0 * (q_1 * q_1 + q_2 * q_2))
    # pitch = asin(2.0 * (q_0 * q_2 - q_1 * q_3))
    yaw = atan2(2.0 * (q_0 * q_3 + q_1 * q_2), 1.0 - 2.0 * (q_2 * q_2 + q_3 * q_3))
    return yaw

  def robot_pose_callback(self, msg):
    position = msg.pose.position
    orientation = msg.pose.orientation
    yaw = self.cal_yaw_from_quaternion(orientation.w, orientation.x, orientation.y, orientation.z)
    self._lock_robot_pose.acquire()
    self._robot_pose[0] = position.x
    self._robot_pose[1] = position.y
    self._robot_pose[2] = yaw
    self._lock_robot_pose.release()

  def person1_pose_callback(self, msg):
    position = msg.pose.position
    self._lock_position_person1.acquire()
    self._position_person1[0] = position.x
    self._position_person1[1] = position.y
    self._lock_position_person1.release()

  def person2_pose_callback(self, msg):
    position = msg.pose.position
    self._lock_position_person2.acquire()
    self._position_person2[0] = position.x
    self._position_person2[1] = position.y
    self._lock_position_person2.release()

  def person3_pose_callback(self, msg):
    position = msg.pose.position
    self._lock_position_person3.acquire()
    self._position_person3[0] = position.x
    self._position_person3[1] = position.y
    self._lock_position_person3.release()
    

  def occupy_grid(self, radius, inflation=False):
    if inflation:
      radius_up = radius
      grid = int(radius_up / self._map_resolution)
      if radius_up > grid * self._map_resolution:
        grid = grid + 1
        radius_up = grid * self._map_resolution
      inflation_len = len(self._inflation_grid)
      grid_large = grid + inflation_len
      radius_large = grid_large * self._map_resolution
      occ_size = grid_large * 2 + 1
      occupation = np.zeros((occ_size, occ_size), dtype=np.uint8)
      for i in range(grid_large + 1):
        for j in range(grid_large + 1):
          dy = i * self._map_resolution
          dx = j * self._map_resolution
          dis = hypot(dx, dy)
          if dis <= radius_up:
            occupation[grid_large - i][grid_large - j] = 255
            occupation[grid_large + i][grid_large - j] = 255
            occupation[grid_large + i][grid_large + j] = 255
            occupation[grid_large - i][grid_large + j] = 255
          elif dis >= radius_large: 
            break
          else:
            k = int((dis - radius_up) / self._map_resolution)
            occupation[grid_large - i][grid_large - j] = self._inflation_grid[k]
            occupation[grid_large + i][grid_large - j] = self._inflation_grid[k]
            occupation[grid_large + i][grid_large + j] = self._inflation_grid[k]
            occupation[grid_large - i][grid_large + j] = self._inflation_grid[k]
      return occupation, grid_large

    else:
      radius_up = radius
      grid = int(radius_up / self._map_resolution)
      if radius_up > grid * self._map_resolution:
        grid = grid + 1
        radius_up = grid * self._map_resolution

      occ_size = grid * 2 + 1
      occupation = np.zeros((occ_size, occ_size), dtype=np.uint8)
      for i in range(grid + 1):
        for j in range(grid + 1):
          dy = i * self._map_resolution
          dx = j * self._map_resolution
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
    map_h = int((self._harf_area - px) / self._map_resolution)
    map_w = int((self._harf_area - py) / self._map_resolution)
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
    if map_down >= self._size[0]:
      down = self._size[0] - 1
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
    if map_right >= self._size[0]:
      right = self._size[0] - 1
      right_inflation = 2 * grid_radius - (map_right - right)
    else:
      right = map_right
      right_inflation = 2 * grid_radius
    return up, up_inflation, down, down_inflation, left, left_inflation, right, right_inflation

  def collision_check(self):
    collision_map = self._image[..., 0].astype(np.int32)
    robot_map = self._image[..., 2].astype(np.int32)
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
      inflation_len = len(self._inflation_grid)
    else:
      inflation_len = 0
    delta_grid = grid_radius - inflation_len
    if map_h - delta_grid < 0 or map_w - delta_grid < 0 or \
      map_h + delta_grid >= self._size[0] or map_w + delta_grid >= self._size[0]:
      return True
    return False

  def get_image(self):
    occupation_map_all = np.zeros((self._human_num, self._size[0], self._size[1], 1), dtype=np.uint8)

    for i in range(self._human_num):
      if self._humans[i].px < self._harf_area and self._humans[i].px > -self._harf_area \
        and self._humans[i].py < self._harf_area and self._humans[i].py > -self._harf_area:  # humans with far location are fake
        up, up_inflation, down, down_inflation, \
          left, left_inflation, right, right_inflation = \
          self.inflation_area(self._humans[i].px, self._humans[i].py, self._humans_grid_raidus[i])
        occupation_map_all[i, up:down+1, left:right+1, 0] = \
          self._humans_grid_area[i][up_inflation:down_inflation+1, left_inflation:right_inflation+1]
    occupation_map_reduce_sum = np.sum(occupation_map_all, axis=0)
    occupation_map_binary = occupation_map_reduce_sum > 0
    occupation_map = occupation_map_binary * 255
    occupation_map = occupation_map.astype(np.uint8)

    robot_map = np.zeros((self._size[0], self._size[1], 1), dtype=np.uint8)
    up, up_inflation, down, down_inflation, \
      left, left_inflation, right, right_inflation = \
      self.inflation_area(self._robot.px, self._robot.py, self._robot_grid_raidus)
    robot_map[up:down+1, left:right+1, 0] = \
      self._robot_grid_area[up_inflation:down_inflation+1, left_inflation:right_inflation+1]

    # rgb image, occupation map is red, goal map is green, robot map is blue
    # OpenCV is BGR
    return np.concatenate((occupation_map, self._goal_map, robot_map), axis=-1)

  def step(self, action, full_state=False):
    assert np.isfinite(action['action']).all(), action['action']
    _action = action['action'] * self._max_robot_speed

    if self._real_world:
      dx_world = _action[0] * self._time_step
      dy_world = _action[1] * self._time_step
      # it is better that the robot yaw angle is a small positive value
      # because the maximum Y(sideward) speed of the robot is small
      ct = cos(self._robot_pose[2])
      st = sin(self._robot_pose[2])
      dx_robot = dy_world * st + dx_world * ct
      dy_robot = dy_world * ct - dx_world * st
      move_cmd = Twist()
      move_cmd.linear.x = dx_robot / self._time_step
      move_cmd.linear.y = dy_robot / self._time_step
      self._pub_cmd_vel.publish(move_cmd)

      time.sleep(self._time_step)

      robot_x = self._robot_pose[0]
      robot_y = self._robot_pose[1]
      robot_theta = self._robot_pose[2]

      self._humans[0].set_position(self._position_person1)
      self._humans[1].set_position(self._position_person2)
      self._humans[2].set_position(self._position_person3)

      self._posititon_saving['robot'].append(np.array([self._robot.px, self._robot.py]))
      self._posititon_saving['person1'].append(np.array([self._humans[0].px, self.self._humans[0].py]))
      self._posititon_saving['person2'].append(np.array([self._humans[1].px, self.self._humans[1].py]))
      self._posititon_saving['person3'].append(np.array([self._humans[2].px, self.self._humans[2].py]))

    else:
      human_actions = []
      for human in self._humans:
        # observation for humans is always coordinates
        ob = [other_human.get_observable_state() for other_human in self._humans if other_human != human]
        human_actions.append(human.act(ob))

      # uodate states
      for i in range(self._human_num):
        self._humans[i].update_states(human_actions[i])

      human_outside = False
      for i in range(self._human_num):
        human_map_h, human_map_w = self.position_to_map(self._humans[i].px, self._humans[i].py)
        if self.outside_check(human_map_h, human_map_w, self._humans_grid_raidus[i]):
          human_outside = True
          break
    
      robot_x, robot_y, robot_theta = self._robot.compute_pose(_action)
    self._robot.update_states(robot_x, robot_y, robot_theta, _action)
    
    self._image = self.get_image()

    robot_map_h, robot_map_w = self.position_to_map(robot_x, robot_y)
    collide, discomfort_grid_value = self.collision_check()
    outside = self.outside_check(robot_map_h, robot_map_w, self._robot_grid_raidus, inflation=True)

    reward = 0.0
    goal_reach = self._image[robot_map_h, robot_map_w, 1] > 0
    goal_dis = self._robot.get_goal_distance()
    goal_reach_larger_area = goal_dis < self._goal_reach_larger_dis
    if goal_reach_larger_area and (not self._goal_reach_larger_area):
      self._goal_reach_larger_area = True
      print('goal reaching on larger area')

    is_last = False
    self._current_step += 1
    if human_outside:
      print('human outside')
      is_last = True
    if self._current_step % self._time_limit == 0:
      print('time out')
      is_last = True

    if goal_reach:
      reward = 1.0
      is_last = True
      print('goal reaching on smaller area')
    elif collide:
      reward = self._collision_reward
      is_last = True
      print('collision happens!')
    elif outside:
      reward = self._outside_reward
      is_last = True
      print('robot outside')

    else:
      goal_map_h, goal_map_w = self.position_to_map(self._robot.gx, self._robot.gy)
      goal_dis = hypot(goal_map_h - robot_map_h, goal_map_w - robot_map_w) * self._map_resolution
      # print(goal_dis)
      reward = self._collision_reward_factor * discomfort_grid_value / 255.0 + \
               self._goal_reward_factor * (1.0 - goal_dis / self._motion_area)

    if is_last and self._real_world:
      # stop the robot
      move_cmd = Twist()
      move_cmd.linear.x = 0.0
      move_cmd.linear.y = 0.0
      self._pub_cmd_vel.publish(move_cmd)
      for key in self._posititon_saving.keys():
        self._posititon_saving[key] = np.array(self._posititon_saving[key])
        np.savez_compressed('position.npz', **self._posititon_saving)
      
    # print('reward: ', reward)

    if not self._real_world:
      for i, human in enumerate(self._humans):
        # let humans move circularly from two points
        if human.reached_destination():
          self._humans[i].gx = -self._humans[i].gx
          self._humans[i].gy = -self._humans[i].gy

    obs = {
        'reward': reward,
        'is_first': False,
        'is_last': is_last,
        'is_terminal': goal_reach or collide or outside,
        'image': self._image,
        'goal_reach_larger_area': self._goal_reach_larger_area
    }

    # the ob_coordinate is used for ORCA policy
    self_state = common.FullState(self._robot.px, self._robot.py, self._robot.vx, self._robot.vy, self._robot.radius, \
                            self._robot.gx, self._robot.gy, self._robot.v_pref, self._robot.theta)
    ob_state = [human.get_observable_state() for human in self._humans]
    ob_coordinate = common.JointState(self_state, ob_state)

    if full_state:
      return [obs, ob_coordinate]
    else:
      return obs

  def generate_random_humans(self):
    # initial min separation distance to avoid danger penalty at beginning
    self._humans = []
    self._humans_grid_area = []
    self._humans_grid_raidus = []
    if self._random_human_num:
      self._human_num = np.random.randint(self._human_num_max + 1, size=1)[0]
    else:
      self._human_num = self._human_num_max
    for i in range(self._human_num):
      human = self.generate_circle_crossing_human()
      self._humans.append(human)

  def generate_circle_crossing_human(self):
    human = common.Human(self._human_radius, self._time_step)
    if self._randomize_attributes:
      human.v_pref = np.random.uniform(self._max_human_speed / 2.0, self._max_human_speed, 1)[0]
    else:
      human.v_pref = self._max_human_speed
    while True:
      if self._scenario == 'quadruped_motion_capture' or self._scenario == 'quadruped_loam':
        ### the position and goal are randomly set on a square
        position_and_goal = np.random.uniform(-self._circle_radius, self._circle_radius, 4)
        if hypot(position_and_goal[0] - position_and_goal[2], position_and_goal[1] - position_and_goal[3]) \
              >= self._goal_reach_larger_dis * 6:
          px = position_and_goal[0]
          py = position_and_goal[1]
          gx = position_and_goal[2]
          gy = position_and_goal[3]
        else:
          continue
      
      else:
        ### the position and goal are randomly set around a circle curve
        angle = np.random.random() * np.pi * 2
        # add some noise to simulate all the possible cases robot could meet with human
        px_noise = (np.random.random() - 0.5) * human.v_pref
        py_noise = (np.random.random() - 0.5) * human.v_pref
        px = self._circle_radius * np.cos(angle) + px_noise
        py = self._circle_radius * np.sin(angle) + py_noise
        gx = -px
        gy = -py
      ### if the initial human location is too close to other humans and the robot,
      ### or the goal human location is too to other humans' goal locations and the robot's goal location,
      ### reset the human position
      collide = False
      for agent in [self._robot] + self._humans:
        min_dist = human.radius + agent.radius + self._robot.radius
        if hypot(px - agent.px, py - agent.py) < min_dist or \
            hypot(px - agent.gx, py - agent.gy) < min_dist:
          collide = True
          break
      if not collide:
        break
    # px, py, gx, gy, vx, vy, theta
    human.set(px, py, gx, gy, 0, 0, 0)
    grid_area, grid_radius = self.occupy_grid(human.radius)
    self._humans_grid_area.append(grid_area)
    self._humans_grid_raidus.append(grid_radius)
    human_policy = common.ORCA(time_step=self._time_step, max_speed=self._max_human_speed, radius=human.radius)
    human.set_policy(human_policy)
    return human

  def reset(self, full_state=False):
    if self._real_world:
      self._posititon_saving = {
        'robot': [],
        'person1': [],
        'person2': [],
        'person3': []
      }
      # please set your own goal position
      self._robot.set(self._robot_pose[0], self._robot_pose[1], self._goal_position[0], self._goal_position[1], 0, 0, self._robot_pose[2])
      # in real world, human number is a constant --- 3,
      # if human number is less than 3,
      # add some fake persons whose position are more than 999 or any large values
      self._human_num = self._human_num_max  
      self._humans = []
      self._humans_grid_area = []
      self._humans_grid_raidus = []

      human1 = common.Human(self._human_radius, self._time_step)
      human1.px = self._position_person1[0]
      human1.py = self._position_person1[1]
      self._humans.append(human1)

      human2 = common.Human(self._human_radius, self._time_step)
      human2.px = self._position_person2[0]
      human2.py = self._position_person2[1]
      self._humans.append(human2)

      human3 = common.Human(self._human_radius, self._time_step)
      human3.px = self._position_person3[0]
      human3.py = self._position_person3[1]
      self._humans.append(human3)

      for human in self._humans:
        grid_area, grid_radius = self.occupy_grid(human.radius)
        self._humans_grid_area.append(grid_area)
        self._humans_grid_raidus.append(grid_radius)
      
      self._posititon_saving['robot'].append(np.array([self._robot.px, self._robot.py]))
      self._posititon_saving['person1'].append(np.array([self._humans[0].px, self.self._humans[0].py]))
      self._posititon_saving['person2'].append(np.array([self._humans[1].px, self.self._humans[1].py]))
      self._posititon_saving['person3'].append(np.array([self._humans[2].px, self.self._humans[2].py]))

    else:
      # px, py, gx, gy, vx, vy, theta
      self._robot.set(-self._circle_radius, 0, self._circle_radius, 0, 0, 0, np.pi / 2)
      if self._random_robot:
        ### robot position and goal are randomly set on a square
        while True:
          if self._scenario == 'quadruped_motion_capture':
            position_and_goal = np.random.uniform(-self._circle_radius, self._circle_radius, 4)
            if hypot(position_and_goal[0] - position_and_goal[2], position_and_goal[1] - position_and_goal[3]) \
                  >= self._goal_reach_larger_dis * 3:
              self._robot.px = position_and_goal[0]
              self._robot.py = position_and_goal[1]
              self._robot.gx = position_and_goal[2]
              self._robot.gy = position_and_goal[3]
              break
          elif self._scenario == 'quadruped_loam': # corridor 
            self._robot.px = -self._circle_radius
            self._robot.py = np.random.uniform(-1.0, 1.0, 1)[0]
            self._robot.gx = np.random.uniform(-self._circle_radius + self._goal_reach_larger_dis * 3, self._circle_radius, 1)[0]
            self._robot.gy = np.random.uniform(-1.0, 1.0, 1)[0]
          else:
            raise NotImplementedError(self._scenario)

      self.generate_random_humans()

    self._goal_map = np.zeros((self._size[0], self._size[1], 1), dtype=np.uint8)
    up, up_inflation, down, down_inflation, \
      left, left_inflation, right, right_inflation = \
      self.inflation_area(self._robot.gx, self._robot.gy, self._goal_grid_raidus)
    self._goal_map[up:down+1, left:right+1, 0] = \
      self._goal_grid_area[up_inflation:down_inflation+1, left_inflation:right_inflation+1]
    
    self._image = self.get_image()
    
    self._current_step = 0
    self._goal_reach_larger_area = False
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._image,
        'goal_reach_larger_area': self._goal_reach_larger_area
    }

    # the ob_coordinate is used for ORCA policy
    self_state = common.FullState(self._robot.px, self._robot.py, self._robot.vx, self._robot.vy, self._robot.radius, \
                            self._robot.gx, self._robot.gy, self._robot.v_pref, self._robot.theta)
    ob_state = [human.get_observable_state() for human in self._humans]
    ob_coordinate = common.JointState(self_state, ob_state)

    if full_state:
      return [obs, ob_coordinate]
    else:
      return obs

  def render(self):
    cv2.imshow('image', self._image)
    cv2.waitKey(1)

