from policy.lidar_dqn import Lidar_DQN
from policy.orca import ORCA

policy_factory = dict()
policy_factory['lidar_dqn'] = Lidar_DQN
policy_factory['orca'] = ORCA

