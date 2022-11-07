import random
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state1, state2, action, reward, next_state1, next_state2, done):
        self.buffer.append([state1, state2, action, reward, next_state1, next_state2, done])
    
    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states1, states2, actions, rewards, next_states1, next_states2, dones = map(np.asarray, zip(*sample))
        states1 = np.array(states1).reshape(batch_size, -1)
        states2 = np.array(states2).reshape(batch_size, -1)
        next_states1 = np.array(next_states1).reshape(batch_size, -1)
        next_states2 = np.array(next_states2).reshape(batch_size, -1)
        return states1, states2, actions, rewards, next_states1, next_states2, dones
    
    def size(self):
        return len(self.buffer)