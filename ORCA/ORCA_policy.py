import sys
import logging
import argparse
import os
import shutil
import numpy as np
from crowd_sim import CrowdSim
from info import *
from policy.policy_factory import policy_factory
from math import fabs
import time


def run_k_episodes(k, phase):
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
        position, ob_coordinate = env.reset(phase)
        env.render()
        done = False
        while not done:
            # t1 = time.time()
            action = im_policy.predict(ob_coordinate)
            next_position, next_ob_coordinate, reward, done, info = env.step(action)
            env.render()
            position = next_position
            ob_coordinate = next_ob_coordinate

            if isinstance(info, Danger):
                too_close += 1
                min_dist.append(info.min_dist)
            time.sleep(env.time_step)
            env.render()
            # t2 = time.time()
            # print(t2 - t1)

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

    logging.info('success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}'.
                    format(success_rate, collision_rate, avg_nav_time))
    if phase in ['val', 'test']:
        logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
        logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    
if __name__ == '__main__':
    # configure logging
    log_file = 'log.log'
    mode = 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    # configure environment
    env = CrowdSim()
    env.configure()
    im_policy = policy_factory['orca']()
    im_policy.time_step = env.time_step
    im_policy.safety_space = 0.0  # increasing it can improve success rate
    run_k_episodes(env.case_size['test'], 'test')

