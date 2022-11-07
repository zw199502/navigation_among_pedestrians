import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.optimizers import Adam
# revise it from crowd_sim.py
lidar_dim = 1800
n_lidar = 4
position_dim = 2
# revise it from crowd_sim.py

class Lidar_DQN:
    def __init__(self):        
        self.lr = None
        # used for deciding whether choosing action randomly or deterministicly
        self.action_dim = None
        self.model = None
        self.target_model = None
        self.gamma = None
        
    def configure(self, action_dim, gamma=0.95):
        self.gamma = gamma
        self.action_dim = action_dim
        self.model = self.create_model()
        self.model.summary()
        self.target_model = self.create_model()
        self.target_update()

    def set_lr(self, lr=0.0001):
        self.lr = lr
        self.model.compile(optimizer=Adam(self.lr), loss='mse')
        self.target_model.compile(optimizer=Adam(self.lr), loss='mse')

    def create_model(self):
        lidar_input_shape = (lidar_dim * n_lidar,)
        lidar_input = Input(lidar_input_shape)
        lidar_input_reshape = Reshape((n_lidar, lidar_dim, 1), input_shape=lidar_input_shape)(lidar_input)
        conv1 = Conv2D(32, kernel_size=[2, 20], strides=(1, 10), activation='relu')(lidar_input_reshape)
        pool1 = MaxPool2D(pool_size=[1, 5])(conv1)
        conv2 = Conv2D(32, kernel_size=[2, 2],  strides=(1, 1), activation='relu')(pool1)
        pool2 = MaxPool2D(pool_size=[1, 2])(conv2)
        flatten1 = Flatten()(pool2)
        linear1 = Dense(128, activation='relu')(flatten1)
        linear2 = Dense(128, activation='relu')(linear1)
        linear3 = Dense(64,  activation='relu')(linear2)

        position_input_shape = (position_dim,)
        position_input = Input(position_input_shape)

        concat_input = concatenate([linear3, position_input], axis=-1)
        linear4 = Dense(128, activation='relu')(concat_input)
        linear5 = Dense(64,  activation='relu')(linear4)
        linear6 = Dense(32,  activation='relu')(linear5)
        out_q = Dense(self.action_dim, activation='linear')(linear6)

        _model = Model([lidar_input, position_input], out_q)
        return _model

    def target_update(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def optimize_batch(self, train_batches, memory, batch_size):
        for _ in range(train_batches):
            states1, states2, actions, rewards, next_states1, next_states2, dones = memory.sample(batch_size)
            targets = self.target_model.predict([states1, states2])
            next_q_values = self.target_model.predict([next_states1, next_states2]).max(axis=1)
            targets[range(batch_size), actions] = rewards + (1 - dones) * next_q_values * self.gamma
            self.model.fit([states1, states2], targets, epochs=1, verbose=0)

    def get_action(self, lidar, position): 
        state1 = np.reshape(lidar, [1, lidar_dim * n_lidar])
        state2 = np.reshape(position, [1, position_dim])
        out_q = self.model.predict([state1, state2])[0]
        max_action_idx = np.argmax(out_q)
        return max_action_idx

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self, fn):
        self.model.load_weights(fn)
