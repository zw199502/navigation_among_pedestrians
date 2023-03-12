import sys
import logging
import argparse
import os
import shutil
import importlib.util
import numpy as np
import torch
import gym
import copy
import re
from tensorboardX import SummaryWriter
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import VNRLTrainer, MPRLTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
import random

def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return None


def main(args):
    set_random_seeds(args.randomseed)
   
    args.config = os.path.join(args.output_dir, 'config.py')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model_1000044.pth')

    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

   
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy_config = config.PolicyConfig()
    policy = policy_factory[policy_config.name]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    if args.real_world:
        from crowd_sim.envs.crowd_sim_real import CrowdSim
    else:
        from crowd_sim.envs.crowd_sim import CrowdSim
    env = CrowdSim()
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    env.set_robot(robot)

    # read training parameters
    train_config = config.TrainConfig(args.debug, args.il_random)
    rl_learning_rate = train_config.train.rl_learning_rate
    train_steps = train_config.train.train_steps
    sample_episodes = train_config.train.sample_episodes
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    step_start = train_config.train.step_start
    step_end = train_config.train.step_end
    step_decay = train_config.train.step_decay
    checkpoint_interval = train_config.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
  

    model.load_state_dict(torch.load(il_weight_file))

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)

    writer = SummaryWriter(log_dir=args.output_dir)
    explorer = Explorer(env, robot, device, writer, memory, policy.gamma, target_policy=policy, name=args.policy)
    success_rate, collision_rate, outside_rate, avg_nav_time, avg_cr,ave_r, running_steps \
             = explorer.run_k_episodes(500, 'test', current_step=0, print_failure=True)
    print('success_rate: ', success_rate, '  collision_rate: ', collision_rate, '  avg_nav_time: ', avg_nav_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--config', type=str, default='configs/icra_benchmark/cadrl.py')
    parser.add_argument('--output_dir', type=str, default='data/cadrl')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true', help='gpu is not supported')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--il_random', default=False, action='store_true')
    parser.add_argument('--real_world', default=False, action='store_true')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)

    # arguments for GCN
    # parser.add_argument('--X_dim', type=int, default=32)
    # parser.add_argument('--layers', type=int, default=2)
    # parser.add_argument('--sim_func', type=str, default='embedded_gaussian')
    # parser.add_argument('--layerwise_graph', default=False, action='store_true')
    # parser.add_argument('--skip_connection', default=True, action='store_true')

    sys_args = parser.parse_args()

    # if sys_args.real_world:
    #     if sys_args.config != 'configs/icra_benchmark/lstm_rl_real.py':
    #         print('real_world mode only support lstm_rl_real config')
    #         raise NotImplementedError(sys_args.config)

    main(sys_args)
