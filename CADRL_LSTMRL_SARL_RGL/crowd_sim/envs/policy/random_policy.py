import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class RandomPolicy(Policy):
    def __init__(self):
        
        super().__init__()
        self.name = 'RandomPolicy'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.safety_space = 0

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        velocity = np.random.uniform(-1.0, 1.0, 2)
        # print(velocity)
        action = ActionXY(velocity[0], velocity[1])
        self.last_state = state
        return action

