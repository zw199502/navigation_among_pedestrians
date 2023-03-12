"""
Never Modify this file! Always copy the settings you want to change to your local file.
"""


import numpy as np


class Config(object):
    def __init__(self):
        pass


class BaseEnvConfig(object):
    env = Config()
    env.time_limit = 20
    env.time_step = 0.2
    env.val_size = 100
    env.test_size = 500
    env.train_size = np.iinfo(np.uint32).max - 2000
    env.randomize_attributes = False
    env.robot_sensor_range = 10

    reward = Config()
    reward.success_reward = 1
    reward.collision_penalty = -0.25
    reward.outside_penalty = -0.05
    reward.discomfort_dist = 0.2
    reward.discomfort_penalty_factor = 0.5

    sim = Config()
    sim.train_val_scenario = 'circle_crossing'
    sim.test_scenario = 'circle_crossing'
    sim.square_width = 10
    sim.circle_radius = 4
    sim.human_num = 5
    sim.nonstop_human = True
    sim.centralized_planning = True

    humans = Config()
    humans.visible = True
    humans.policy = 'orca'
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = 'coordinates'

    robot = Config()
    robot.visible = False
    robot.policy = 'none'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = 'coordinates'

    def __init__(self, debug=False):
        if debug:
            self.env.val_size = 1
            self.env.test_size = 1


class BasePolicyConfig(object):
    rl = Config()
    rl.gamma = 0.9

    om = Config()
    om.cell_num = 4
    om.cell_size = 1
    om.om_channel_size = 3

    action_space = Config()
    action_space.kinematics = 'holonomic'
    action_space.speed_samples = 5
    action_space.rotation_samples = 16
    action_space.sampling = 'exponential'
    action_space.query_env = False
    action_space.rotation_constraint = np.pi / 3

    cadrl = Config()
    cadrl.mlp_dims = [150, 100, 100, 1]
    cadrl.multiagent_training = False

    lstm_rl = Config()
    lstm_rl.global_state_dim = 50
    lstm_rl.mlp1_dims = [150, 100, 100, 50]
    lstm_rl.mlp2_dims = [150, 100, 100, 1]
    lstm_rl.multiagent_training = True
    lstm_rl.with_om = False
    lstm_rl.with_interaction_module = True

    srl = Config()
    srl.mlp1_dims = [150, 100, 100, 50]
    srl.mlp2_dims = [150, 100, 100, 1]
    srl.multiagent_training = True
    srl.with_om = True

    sarl = Config()
    sarl.mlp1_dims = [150, 100]
    sarl.mlp2_dims = [100, 50]
    sarl.attention_dims = [100, 100, 1]
    sarl.mlp3_dims = [150, 100, 100, 1]
    sarl.multiagent_training = True
    sarl.with_om = True
    sarl.with_global_state = True

    gcn = Config()
    gcn.multiagent_training = True
    gcn.num_layer = 2
    gcn.X_dim = 32
    gcn.wr_dims = [64, gcn.X_dim]
    gcn.wh_dims = [64, gcn.X_dim]
    gcn.final_state_dim = gcn.X_dim
    gcn.gcn2_w1_dim = gcn.X_dim
    gcn.planning_dims = [150, 100, 100, 1]
    gcn.similarity_function = 'embedded_gaussian'
    gcn.layerwise_graph = True
    gcn.skip_connection = False

    gnn = Config()
    gnn.multiagent_training = True
    gnn.node_dim = 32
    gnn.wr_dims = [64, gnn.node_dim]
    gnn.wh_dims = [64, gnn.node_dim]
    gnn.edge_dim = 32
    gnn.planning_dims = [150, 100, 100, 1]

    def __init__(self, debug=False):
        pass


class BaseTrainConfig(object):
    trainer = Config()
    trainer.batch_size = 100
    trainer.optimizer = 'Adam'

    imitation_learning = Config()
    imitation_learning.il_episodes = 50
    imitation_learning.il_policy = 'orca'
    imitation_learning.il_epochs = 50
    imitation_learning.il_learning_rate = 0.001
    imitation_learning.safety_space = 0.15

    train = Config()
    train.rl_train_epochs = 1
    train.rl_learning_rate = 0.0001
    # number of batches to train at the end of training episode il_episodes
    train.train_batches = 200
    # training episodes in outer loop
    train.train_episodes = 20000
    train.train_steps = 1000000
    # number of episodes sampled in one training episode
    train.sample_episodes = 1
    train.target_update_interval = 50000
    train.evaluation_interval = 5000
    # the memory pool can roughly store 2K episodes, total size = episodes * 50
    train.capacity = 100000
    train.step_start = 0.8
    train.step_end = 0.1
    train.step_decay = 800000
    train.checkpoint_interval = 50000

    train.train_with_pretend_batch = False

    def __init__(self, debug=False, il_random=False):
        if debug:
            self.imitation_learning.il_episodes = 10
            self.imitation_learning.il_epochs = 5
            self.train.train_episodes = 1
            self.train.checkpoint_interval = self.train.train_episodes
            self.train.evaluation_interval = 1
            self.train.target_update_interval = 1
        if il_random:
            self.imitation_learning.il_episodes = 50
            self.imitation_learning.il_policy = 'random_policy'
            self.imitation_learning.il_learning_rate = 0.0001


        
