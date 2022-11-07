import sys
import logging
import argparse
import os
import shutil
import numpy as np
import tensorflow as tf
from crowd_sim import CrowdSim
from info import *
from memory import ReplayMemory
from policy.policy_factory import policy_factory
from math import fabs

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

def update_memory(lidars, positions, actions, rewards, next_lidars, next_positions, dones):
    length = len(lidars)
    for i in range(length):
        lidar, position, action, reward = lidars[i], positions[i], actions[i], rewards[i]
        next_lidar, next_position = next_lidars[i], next_positions[i]
        done = dones[i]
        memory.put(lidar, position, action, reward, next_lidar, next_position, done)

def run_k_episodes(k, phase, epsilon, if_update_memory=False, if_initialize_memory=False, if_orca=False, episode=None):
    global success_rate_log, success_rate_file, step_log
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    for i in range(k):
        lidar, position, ob_coordinate = env.reset(phase)
        # env.render()
        done = False
        lidars, positions, actions, rewards, next_lidars, next_positions, dones = [], [], [], [], [], [], []
        rewards = []
        while not done:
            if if_orca:
                action = im_policy.predict(ob_coordinate)
                vx_idx = 0
                vy_idx = 0
                for i_x in range(v_sample_num):
                    if fabs(v_sample[i_x] - action[0]) <= delta_v / 2.0:
                        vx_idx = i_x
                        break
                for i_y in range(v_sample_num):
                    if fabs(v_sample[i_y] - action[1]) <= delta_v / 2.0:
                        vy_idx = i_y
                        break
                action_idx = vx_idx * v_sample_num + vy_idx
            else:
                action_idx = np.random.randint(v_sample_num * v_sample_num, size=1)[0]
            if not if_initialize_memory:
                probability = np.random.uniform(0.0, 1.0, 1)[0]
                if phase == 'train' and probability < epsilon:
                    if if_orca:
                        action_idx_noise = np.random.randint(7, size=2)
                        vx_idx = vx_idx + (action_idx_noise[0] - 3)
                        if vx_idx < 0:
                            vx_idx = 0
                        elif vx_idx >= v_sample_num:
                            vx_idx = v_sample_num - 1
                        vy_idx = vy_idx + (action_idx_noise[1] - 3)
                        if vy_idx < 0:
                            vy_idx = 0
                        elif vy_idx >= v_sample_num:
                            vy_idx = v_sample_num - 1
                        action_idx = vx_idx * v_sample_num + vy_idx
                else:
                    action_idx = target_policy.get_action(lidar, position)
            _action = action_space[action_idx]
            next_lidar, next_position, next_ob_coordinate, reward, done, info = env.step(_action)
            # env.render()
            lidars.append(lidar)
            positions.append(position)
            actions.append(action_idx)
            rewards.append(reward)
            next_lidars.append(next_lidar)
            next_positions.append(next_position)
            dones.append(done)
            lidar, position = next_lidar, next_position
            ob_coordinate = next_ob_coordinate

            if isinstance(info, Danger):
                too_close += 1
                min_dist.append(info.min_dist)
        
        if phase == 'train':
            step_log += len(lidars)

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

        if if_update_memory:
            if isinstance(info, ReachGoal) or isinstance(info, Collision):
                # only add positive(success) or negative(collision) experience in experience set
                update_memory(lidars, positions, actions, rewards, next_lidars, next_positions, dones)

        cumulative_rewards.append(sum([pow(rl_gamma, t * env.time_step * env.robot.v_pref)
                                        * reward for t, reward in enumerate(rewards)]))

    success_rate = success / k
    collision_rate = collision / k
    assert success + collision + timeout == k
    avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit

    extra_info = '' if episode is None else 'in episode {} '.format(episode)
    logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                    format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, average(cumulative_rewards)))
    if phase in ['val', 'test']:
        num_step = sum(success_times + collision_times + timeout_times) / env.time_step
        success_rate_log.append(np.array([step_log, success_rate]))
        np.savetxt(success_rate_file, np.array(success_rate_log))
        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                        too_close / num_step, average(min_dist))
        logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
        logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

def main():
    epsilon = epsilon_start
    target_policy.target_update()
    # fill memory
    run_k_episodes(initialize_memory, 'train', epsilon, if_update_memory=True, if_initialize_memory=True, if_orca=if_orca)
    logging.info('Experience set initial size: %d/%d', memory.size(), memory.capacity)
    # initial training
    target_policy.set_lr(it_learning_rate)
    for ii in range(it_iterations):
        target_policy.optimize_batch(it_batches, memory, batch_size)
        target_policy.target_update()

    episode = 0
    target_policy.set_lr(rl_learning_rate)
    while episode < train_episodes:
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        
        if episode % evaluation_interval == 0:
            run_k_episodes(env.case_size['val'], 'val', epsilon, episode=episode)
        # sample k (default is 1) episodes into memory and optimize over the generated memory
        run_k_episodes(sample_episodes, 'train', epsilon, if_update_memory=True, if_orca=if_orca, episode=episode)

        target_policy.optimize_batch(train_batches, memory, batch_size)

        episode += 1

        if episode % target_update_interval == 0:
            target_policy.target_update()

        # save the model as checkpoint
        if episode % checkpoint_interval == 0:
            weight_file = args.output_dir + '/weight_episode_{}.h5'.format(episode)
            target_policy.save_model(weight_file)
       
    # final test
    run_k_episodes(env.case_size['test'], 'test', epsilon, episode=episode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    # whether using orca to initialize the memory
    parser.add_argument('--if_orca', default=True, action='store_true')
    # policy used to initialize memory
    parser.add_argument('--im_policy', type=str, default='orca')
    parser.add_argument('--im_safety_space', type=float, default='0.20') # default 0.15
    parser.add_argument('--initialize_memory', type=int, default='2500')
    # initial training learning rate
    parser.add_argument('--it_learning_rate', type=float, default='0.001')
    # initial training iterations
    parser.add_argument('--it_iterations', type=int, default='20')
    # initial training batches at each iteration
    parser.add_argument('--it_batches', type=int, default='100')
    # action dimension in one directioon
    parser.add_argument('--v_sample_num', type=int, default='9')    # odd integer
    # train policy
    parser.add_argument('--policy', type=str, default='lidar_dqn')
    parser.add_argument('--rl_learning_rate', type=float, default='0.0001') # default 0.0001
    parser.add_argument('--rl_gamma', type=float, default='0.95')  # default 0.95
    # because the value network is too bulky, the batch size can not be large.
    # otherwise, the gpu memory will be run out of
    # the memory size can't be too large either because of physical RAM memory.    
    parser.add_argument('--batch_size', type=int, default='128')    # defalut 128
    parser.add_argument('--train_batches', type=int, default='100') 
    parser.add_argument('--train_episodes', type=int, default='400000') # total training episodes
    parser.add_argument('--target_update_interval', type=int, default='20') # update the target model every target_update_interval episodes
    parser.add_argument('--sample_episodes', type=int, default='1')  # how many episodes for each iteration
    parser.add_argument('--evaluation_interval', type=int, default='100') # evaluate the model every evaluation_interval episodes
    parser.add_argument('--capacity', type=int, default='400000')  # size of experience pool, default is 100000
    # probability of randomly choosing action
    parser.add_argument('--epsilon_start', type=float, default='0.8')
    parser.add_argument('--epsilon_end', type=float, default='0.03')
    # stop decaying epsilon after epsilon_decay episodes
    parser.add_argument('--epsilon_decay', type=int, default='50000')
    # save network weights every checkpoint_interval episodes
    parser.add_argument('--checkpoint_interval', type=int, default='2000')
    
    parser.add_argument('--output_dir', type=str, default='data/output_orca')
    parser.add_argument('--gpu', default=True, action='store_true')
    args = parser.parse_args()

    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y':
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
    if make_new_dir:
        os.makedirs(args.output_dir)

    # save configuration
    file_configure_name = os.path.join(args.output_dir, 'configure.txt')
    file_configure = open(file_configure_name, 'a')
    if args.if_orca:
        file_configure.write('if_orca: ' + 'true' + '\n')
    else:
        file_configure.write('if_orca: ' + 'false' + '\n')
    file_configure.write('im_safety_space: ' + (str(args.im_safety_space)) + '\n')
    file_configure.write('initialize_memory: ' + (str(args.initialize_memory)) + '\n')
    file_configure.write('it_learning_rate: ' + (str(args.it_learning_rate)) + '\n')
    file_configure.write('it_iterations: ' + (str(args.it_iterations)) + '\n')
    file_configure.write('it_batches: ' + (str(args.it_batches)) + '\n')
    file_configure.write('v_sample_num: ' + (str(args.v_sample_num)) + '\n')
    file_configure.write('rl_learning_rate: ' + (str(args.rl_learning_rate)) + '\n')
    file_configure.write('rl_gamma: ' + (str(args.rl_gamma)) + '\n') 
    file_configure.write('batch_size: ' + (str(args.batch_size)) + '\n')
    file_configure.write('train_batches: ' + (str(args.train_batches)) + '\n') 
    file_configure.write('train_episodes: ' + (str(args.train_episodes)) + '\n')
    file_configure.write('target_update_interval: ' + (str(args.target_update_interval)) + '\n')
    file_configure.write('sample_episodes: ' + (str(args.sample_episodes)) + '\n')
    file_configure.write('evaluation_interval: ' + (str(args.evaluation_interval)) + '\n')
    file_configure.write('capacity: ' + (str(args.capacity)) + '\n')
    file_configure.write('epsilon_start: ' + (str(args.epsilon_start)) + '\n')
    file_configure.write('epsilon_end: ' + (str(args.epsilon_end)) + '\n')
    file_configure.write('epsilon_decay: ' + (str(args.epsilon_decay)) + '\n')
    file_configure.write('checkpoint_interval: ' + (str(args.checkpoint_interval)) + '\n')
    file_configure.write('output_dir: ' + args.output_dir + '\n')
    file_configure.close()
    # save configuration

    # tensorboard log
    tensorboard_log_name = os.path.join(args.output_dir, 'tensorboard_log')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_name, histogram_freq=1)
    # tensorboard log

    success_rate_log = []
    step_log = 0
    success_rate_file = "success_rate.txt"
    log_file = os.path.join(args.output_dir, 'output.log')

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
        tf.config.experimental.set_virtual_device_configuration(gpus[gpu_index], \
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        device = 'gpu:' + str(gpu_index)
    else:
        device = 'cpu'
    logging.info('device: ' + device)

    if_orca = args.if_orca
    it_learning_rate = args.it_learning_rate
    it_iterations = args.it_iterations
    it_batches = args.it_batches
    rl_learning_rate = args.rl_learning_rate
    rl_gamma = args.rl_gamma
    batch_size = args.batch_size  
    train_batches = args.train_batches
    target_update_interval = args.target_update_interval
    train_episodes = args.train_episodes
    sample_episodes = args.sample_episodes
    evaluation_interval = args.evaluation_interval
    capacity = args.capacity
    epsilon_start = args.epsilon_start
    epsilon_end = args.epsilon_end
    epsilon_decay = args.epsilon_decay
    checkpoint_interval = args.checkpoint_interval
    memory = ReplayMemory(capacity)

    # configure environment
    env = CrowdSim()
    env.configure()
    
    # configure policy
    # policy used to initialize memory --> orca
    initialize_memory = args.initialize_memory
    im_policy = policy_factory[args.im_policy]()
    im_policy.time_step = env.time_step
    im_policy.safety_space = args.im_safety_space
    # policy --> map dqn
    target_policy = policy_factory[args.policy]()
    v_sample_num = args.v_sample_num
    target_policy.configure(v_sample_num * v_sample_num, rl_gamma) # square of a odd integer
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
