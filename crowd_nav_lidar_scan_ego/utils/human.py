from utils.state import *
import numpy as np
from numpy.linalg import norm

class Human():
    def __init__(self):
        self.v_pref = 1.0
        self.radius = 0.0
        self.policy = None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def set_policy(self, policy):
        self.policy = policy

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)
        
    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy

    def get_goal_distance(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position()))

    def compute_position(self, action):
        px = self.px + action[0] * self.time_step
        py = self.py + action[1] * self.time_step
        return px, py

    def update_states(self, action):
        """
        Perform an action and update the state
        """
        pos = self.compute_position(action)
        self.px, self.py = pos
        self.vx = action[0]
        self.vy = action[1]

    def reached_destination(self):
        return self.get_goal_distance() < self.radius
