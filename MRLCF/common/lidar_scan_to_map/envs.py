import atexit
import sys 
import traceback

import cloudpickle
import gym
import numpy as np
import common
from C_library.lidar_simulation import *
from math import hypot, exp, fabs
import cv2
import time

class Navigation:

  def __init__(self, config, mode):
    assert config.render_size[0] == config.render_size[1]
    # domain: omnidirectional, differential
    # task: static, dynamic
    self._size = tuple(config.render_size)
    self._initial_min_goal_dis = config.initial_min_goal_dis
    self._human_num = config.human_num
    self._goal_reward_factor = config.goal_reward_factor
    self._collision_reward_factor = config.collision_reward_factor
    self._randomize_attributes = config._randomize_attributes
    # laser scan parameters
    # number of all laser beams
    self._num_scan = 1800
    self._laser_resolution = 0.003490659
    # environment size
    self._harf_area = 5.0
    # environment size
    self._time_step = 0.25
    self._circle_radius = self._harf_area - 1.0
    
    self._lines = [
                  [(-self._harf_area, -self._harf_area), (-self._harf_area,  self._harf_area)], 
                  [(-self._harf_area,  self._harf_area), ( self._harf_area,  self._harf_area)], 
                  [( self._harf_area,  self._harf_area), ( self._harf_area, -self._harf_area)], 
                  [( self._harf_area, -self._harf_area), (-self._harf_area, -self._harf_area)]
                  ]
    self._min_scan = None
    self._min_scan_end = None
    
    self._robot = common.Robot(self._time_step)
    self._humans = []

    self._map_resolution = self._harf_area * 2.0 / self._size[0]
    lethal_dis = self._robot.radius
    lethal_grid = int(lethal_dis / self._map_resolution)
    if lethal_dis > lethal_grid * self._map_resolution:
      self._lethal_grid = lethal_grid + 1
      self._lethal_dis = self._lethal_grid * self._map_resolution
    else:
      self._lethal_grid = lethal_grid
      self._lethal_dis = lethal_dis

    self._discomfort_dist = self._lethal_dis
    self._inflation = self.occupy_inflation()
    self._inflation_resize = np.resize(self._inflation, (2 * self._lethal_grid + 1, 2 * self._lethal_grid + 1))
    self._image = None

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, self._size + (1,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    return spaces

  @property
  def act_space(self):
    action = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    return {'action': action}

  def occupy_inflation(self):
    inflation = np.zeros((self._lethal_grid * 2 + 1) * (self._lethal_grid * 2 + 1), dtype=np.uint8)
    for i in range(self._lethal_grid * 2 + 1):
      for j in range(self._lethal_grid * 2 + 1):
        dx = (i - self._lethal_grid) * self._map_resolution
        dy = (j - self._lethal_grid) * self._map_resolution
        dis = hypot(dx, dy)
        if dis <= self._lethal_dis:
          inflation[i * (self._lethal_grid * 2 + 1) + j] = 255
    return inflation

  def generate_random_humans(self):
    # initial min separation distance to avoid danger penalty at beginning
    self._humans = []
    human_num = self._human_num
    for i in range(human_num):
      self._humans.append(self.generate_circle_crossing_human())

  def generate_circle_crossing_human(self):
    human = common.Human(self._time_step)

    if self._randomize_attributes:
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
      px = self._circle_radius * np.cos(angle) + px_noise
      py = self._circle_radius * np.sin(angle) + py_noise
      collide = False
      for agent in [self._robot] + self._humans:
        min_dist = human.radius + agent.radius + self._discomfort_dist
        if np.linalg.norm((px - agent.px, py - agent.py)) < min_dist or \
            np.linalg.norm((px - agent.gx, py - agent.gy)) < min_dist:
          collide = True
          break
      if not collide:
        break
    # px, py, gx, gy, vx, vy, theta
    human.set(px, py, -px, -py, 0, 0, 0)
    human_policy = common.ORCA()
    human_policy.time_step = self._time_step
    human.set_policy(human_policy)
    return human
  
  def position_to_map(self, px, py):
    map_h = int((self._harf_area - py) / self._map_resolution)
    map_w = int((self._harf_area + px) / self._map_resolution)
    return map_h, map_w

  def inflation_area(self, px, py):
    map_h, map_w = self.position_to_map(px, py)

    map_up = map_h - self._lethal_grid
    if map_up < 0:
      up = 0
      up_inflation = -map_up
    else:
      up = map_up
      up_inflation = 0

    map_down = map_h + self._lethal_grid
    if map_down >= self._size[0]:
      down = self._size[0] - 1
      down_inflation = 2 * self._lethal_grid - (map_down - down)
    else:
      down = map_down
      down_inflation = 2 * self._lethal_grid

    map_left = map_w - self._lethal_grid
    if map_left < 0:
      left = 0
      left_inflation = -map_left
    else:
      left = map_left
      left_inflation = 0

    map_right = map_w + self._lethal_grid
    if map_right >= self._size[0]:
      right = self._size[0] - 1
      right_inflation = 2 * self._lethal_grid - (map_right - right)
    else:
      right = map_right
      right_inflation = 2 * self._lethal_grid
    return up, up_inflation, down, down_inflation, left, left_inflation, right, right_inflation


  def get_map(self):
    occupation_map = np.zeros(self._size + (1,), np.uint8)
    goal_map = np.zeros(self._size + (1,), np.uint8)
    robot_map = np.zeros(self._size + (1,), np.uint8)
    human_num = len(self._humans)
    line_num = len(self._lines)
    InitializeEnv(line_num, human_num, self._num_scan, self._laser_resolution,
                  self._harf_area, self._size[0], self._map_resolution, self._lethal_grid)
    
    for i in range (line_num):
      set_lines(4 * i    , self._lines[i][0][0])
      set_lines(4 * i + 1, self._lines[i][0][1])
      set_lines(4 * i + 2, self._lines[i][1][0])
      set_lines(4 * i + 3, self._lines[i][1][1])
    for i in range (human_num):
      set_circles(3 * i    , self._humans[i].px)
      set_circles(3 * i + 1, self._humans[i].py)
      set_circles(3 * i + 2, self._humans[i].radius)
    for i in range(self._inflation.shape[0]):
      set_inflation(i, self._inflation[i])
    set_robot_pose(self._robot.px, self._robot.py, self._robot.theta)
    # t1 = time.time()
    cal_laser()
    # t2 = time.time()
    # print('scan time: ', t2 - t1)  # from 0.002 to 0.008 second

    self._min_scan = get_min_scan()
    self._min_scan_end = [get_min_scan_end_x(), get_min_scan_end_y()]

    # t3 = time.time()
    cal_grid_map()
    # t4 = time.time()
    # print('map time: ', t4 - t3) # from 0.009 to 0.031 second
    for i in range(self._size[0]):
      for j in range(self._size[1]):
        occupation_map[i, j, 0] = get_grid_map(i * self._size[0] + j)
    
    up, up_inflation, down, down_inflation, \
      left, left_inflation, right, right_inflation = self.inflation_area(self._robot.gx, self._robot.gy)
    goal_map[up:down+1, left:right+1, 0] = \
      self._inflation_resize[up_inflation:down_inflation+1, left_inflation:right_inflation+1]

    up, up_inflation, down, down_inflation, \
      left, left_inflation, right, right_inflation = self.inflation_area(self._robot.px, self._robot.py)
    robot_map[up:down+1, left:right+1, 0] = \
      self._inflation_resize[up_inflation:down_inflation+1, left_inflation:right_inflation+1]

    ReleaseEnv()
    # rgb image, occupation map is red, goal map is green, robot map is blue
    # OpenCV is BGR
    return np.concatenate((occupation_map, goal_map, robot_map), axis=-1)

  def step(self, action):
    assert np.isfinite(action['action']).all(), action['action']

    human_actions = []
    for human in self._humans:
      # observation for humans is always coordinates
      ob = [other_human.get_observable_state() for other_human in self._humans if other_human != human]
      human_actions.append(human.act(ob))

    # uodate states
    _action = action['action']
    robot_x, robot_y, robot_theta = self._robot.compute_pose(_action)
    self._robot.update_states(robot_x, robot_y, robot_theta, _action)
    for i, human_action in enumerate(human_actions):
      self._humans[i].update_states(human_action)

    self._image = self.get_map()

    reward = 0.0
    goal_reach = False
    collide = False
    robot_map_h, robot_map_w = self.position_to_map(robot_x, robot_y)
    
    if self._image[robot_map_h, robot_map_w, 0] > 0:
      collide = True
      reward = -1.0
    elif self._image[robot_map_h, robot_map_w, 1] > 0:
      goal_reach = True
      reward = 1.0
    else:
      goal_map_h, goal_map_w = self.position_to_map(self._robot.gx, self._robot.gy)
      collision_map_h, collision_map_w = self.position_to_map(self._min_scan_end[0], self._min_scan_end[1])

      goal_robot_h = fabs(goal_map_h - robot_map_h)
      goal_robot_w = fabs(goal_map_w - robot_map_w)
      goal_dis = hypot(goal_robot_h, goal_robot_w) * self._map_resolution - self._lethal_dis
      reward = self._goal_reward_factor * exp(-0.25 * goal_dis)

      collision_robot_h = fabs(collision_map_h - robot_map_h)
      collision_robot_w = fabs(collision_map_w - robot_map_w)
      collision_dis  = hypot(collision_robot_h, collision_robot_w) * self._map_resolution - self._lethal_dis
      if collision_dis <= 1.0 * self._lethal_dis:
        reward += self._collision_reward_factor * exp(-2.0 * collision_dis)
    # print('reward: ', reward)

    for i, human in enumerate(self._humans):
      # let humans move circularly from two points
      if human.reached_destination():
        self._humans[i].gx = -self._humans[i].gx
        self._humans[i].gy = -self._humans[i].gy

    obs = {
        'reward': reward,
        'is_first': False,
        'is_last': collide,
        'is_terminal': goal_reach or collide,
        'image': self._image
    }

    # the ob_coordinate is used for ORCA planner
    self_state = common.FullState(self._robot.px, self._robot.py, self._robot.vx, self._robot.vy, self._robot.radius, \
                            self._robot.gx, self._robot.gy, self._robot.v_pref, self._robot.theta)
    ob_state = [human.get_observable_state() for human in self._humans]
    ob_coordinate = common.JointState(self_state, ob_state)

    return obs

  def reset(self):
    # px, py, gx, gy, vx, vy, theta
    self._robot.set(0, -self._circle_radius, 0, self._circle_radius, 0, 0, np.pi / 2)
    # while True:
    #   position_and_goal = np.random.uniform(-self._circle_radius, -self._circle_radius, 4)
    #   if hypot(position_and_goal[0] - position_and_goal[2], position_and_goal[1] - position_and_goal[3]) \
    #         >= self._initial_min_goal_dis:
    #     self._robot.px = position_and_goal[0]
    #     self._robot.py = position_and_goal[1]
    #     self._robot.gx = position_and_goal[2]
    #     self._robot.gy = position_and_goal[3]
    #     break

    # self._human_num = np.random.choice(6, 1)[0] + 3  # from 3 to 8
    self.generate_random_humans()
    self._image = self.get_map()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._image
    }

    # the ob_coordinate is used for ORCA planner
    self_state = common.FullState(self._robot.px, self._robot.py, self._robot.vx, self._robot.vy, self._robot.radius, \
                            self._robot.gx, self._robot.gy, self._robot.v_pref, self._robot.theta)
    ob_state = [human.get_observable_state() for human in self._humans]
    ob_coordinate = common.JointState(self_state, ob_state)
    return obs

  def render(self):
    cv2.imshow('image', self._image)
    cv2.waitKey(1)

class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class Async:

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _CLOSE = 4
  _EXCEPTION = 5

  def __init__(self, constructor, strategy='thread'):
    self._pickled_ctor = cloudpickle.dumps(constructor)
    if strategy == 'process':
      import multiprocessing as mp
      context = mp.get_context('spawn')
    elif strategy == 'thread':
      import multiprocessing.dummy as context
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = context.Pipe()
    self._process = context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    self._receive()  # Ready.
    self._obs_space = None
    self._act_space = None

  def access(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      pass  # The connection was already closed.
    self._process.join(5)

  @property
  def obs_space(self):
    if not self._obs_space:
      self._obs_space = self.access('obs_space')()
    return self._obs_space

  @property
  def act_space(self):
    if not self._act_space:
      self._act_space = self.access('act_space')()
    return self._act_space

  def step(self, action, blocking=False):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=False):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    try:
      ctor = cloudpickle.loads(self._pickled_ctor)
      env = ctor()
      conn.send((self._RESULT, None))  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print('Error in environment process: {}'.format(stacktrace))
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass  # The connection was already closed.

