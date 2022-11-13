import sys
import logging
import argparse
import os
import numpy as np
import tensorflow as tf
from crowd_sim import CrowdSim
from info import *
from policy.policy_factory import policy_factory
import time
tf.keras.backend.set_floatx('float32')

def list_sum(input_list):
    _sum = 0
    for data in input_list:
        _sum = _sum + data
    return _sum

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0

def run_k_episodes(k, phase, episode=None):
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0
    min_dist = []
    collision_cases = []
    timeout_cases = []
    for i in range(k):
        lidar, position, ob_coordinate = env.reset(phase)
        env.render()
        done = False
        while not done:
            action_idx = target_policy.get_action(lidar, position)
            _action = action_space[action_idx]
            next_lidar, next_position, next_ob_coordinate, reward, done, info = env.step(_action)
            env.render()
            lidar, position = next_lidar, next_position
            ob_coordinate = next_ob_coordinate

            if isinstance(info, Danger):
                too_close += 1
                min_dist.append(info.min_dist)
            time.sleep(env.time_step - 0.13)

        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(env.global_time)
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(env.global_time)
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env.time_limit)
        else:
            raise ValueError('Invalid end signal from environment')

    success_rate = success / k
    collision_rate = collision / k
    assert success + collision + timeout == k
    avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit

    extra_info = '' if episode is None else 'in episode {} '.format(episode)
    logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}'.
        format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time))
    if phase in ['val', 'test']:
        num_step = sum(success_times + collision_times + timeout_times) / env.time_step
        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                        too_close / num_step, average(min_dist))
        logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
        logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

def main():
    # final test
    run_k_episodes(env.case_size['test'], 'test', episode=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    # action dimension in one directioon
    parser.add_argument('--v_sample_num', type=int, default='9')    # odd integer
    # train policy
    parser.add_argument('--policy', type=str, default='lidar_dqn')
    parser.add_argument('--output_dir', type=str, default='data/output_orca')
    parser.add_argument('--gpu', default=True, action='store_true')
    args = parser.parse_args()

    log_file = os.path.join(args.output_dir, 'output_test.log')

    # configure logging
    mode = 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    # allocate gpu
    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        gpu_index = 0
        tf.config.experimental.set_visible_devices(devices=gpus[gpu_index], device_type='GPU')
        # tf.config.experimental.set_virtual_device_configuration(gpus[gpu_index], \
        # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        device = 'gpu:' + str(gpu_index)
    else:
        device = 'cpu'
    logging.info('device: ' + device)

    # configure environment
    env = CrowdSim()
    env.configure()
    
    # policy --> map dqn
    target_policy = policy_factory[args.policy]()
    model_weight_file = os.path.join(args.output_dir, 'weight_episode_12000.h5')
    v_sample_num = args.v_sample_num
    target_policy.configure(v_sample_num * v_sample_num) # square of a odd integer
    target_policy.set_lr()
    target_policy.load_model(model_weight_file)
    delta_v = 2.0 / (v_sample_num - 1) # speed from -1m/s to 1m/s
    v_sample = []
    for i in range(v_sample_num):
        v = -1.0 + delta_v * i
        v_sample.append(v)
    action_space = []
    for vx in v_sample:
        for vy in v_sample:
            action_space.append((vx, vy))
    main()
