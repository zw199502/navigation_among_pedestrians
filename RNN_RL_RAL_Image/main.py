import argparse
import numpy as np
import random
import os
import torch
import time

from algos import TD3
from utils import memory
from info import *
from crowd_sim import CrowdSim

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, eval_episodes=100, test=False):
    policy.eval_mode()
    avg_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    collision_cases = []
    timeout_cases = []
    for i in range(eval_episodes):
        image  = eval_env.reset()
        # eval_env.render()
        # time.sleep(0.2)
        done = False
        hidden = None
        ep_step = 0
        while not done:
            with torch.no_grad():
                t1 = time.time()
                action, hidden = policy.select_action(image, hidden)
                t2 = time.time()
                # print('time: ', t2 - t1)
                
            # env.render(mode='human', close=False)
            image, reward, done, info = eval_env.step(action)
            # eval_env.render()
            # time.sleep(0.2)
            avg_reward += reward
            ep_step = ep_step + 1
        # file_name = './evaluation_episodes/eval_' + str(time.time()) + '.npz'
        if i < 10:
            file_name = file_prefix + '/evaluation_episodes/eval_' + str(time.time()) + '.npz'
            np.savez_compressed(file_name, **eval_env.log_env)
        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(eval_env.global_time)
            print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(eval_env.global_time)
            print('evaluation episode ' + str(i) + ', collision occur at evaluation step: ' + str(ep_step))
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(eval_env.time_limit)
            print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
        else:
            raise ValueError('Invalid end signal from environment')

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_env.time_limit

    policy.train_mode()
    
    return success_rate, collision_rate, avg_nav_time


def main():
    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    # To Do, revise DDPG according to TD3
    parser.add_argument("--policy", default="TD3")
    # device
    parser.add_argument("--device", type=str, default='cuda:0')
    # OpenAI gym environment name
    parser.add_argument("--env", default="Navigation")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=2, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=1e4, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=2e4, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.25)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=100, type=int)
    # Memory size
    parser.add_argument("--memory_size", default=1e5, type=int)
    # Learning rate
    parser.add_argument("--lr", default=3e-4, type=float)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.25)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Model width
    parser.add_argument("--hidden_size", default=512, type=int)
    # please set it as True
    parser.add_argument("--recurrent", default=True, action="store_true")
    # Save model and optimizer parameters
    parser.add_argument("--save_model", default=True, action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    # parser.add_argument("--load_model", type=str, default="/models/step_320000")
    parser.add_argument("--load_model", type=str, default="")
    # Don't train and just run the model
    parser.add_argument("--test", default=False, action="store_true")
    # environment settings
    parser.add_argument("--env_type", type=str, default="crowd_sim")
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--image_feature_dim", type=int, default=1024)

    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(file_prefix + '/results'):
        os.makedirs(file_prefix + '/results')

    if args.save_model and not os.path.exists(file_prefix + '/models'):
        os.makedirs(file_prefix + '/models')

    if not os.path.exists(file_prefix + '/evaluation_episodes'):
        os.makedirs(file_prefix + '/evaluation_episodes')

 
        

    env = CrowdSim(args)
    eval_env = CrowdSim(args)

    # Set seeds
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    image_size = args.image_size
    image_feature_dim = args.image_feature_dim
    action_dim = args.action_dim
    max_action = 1.0

    # TODO: Add this to parameters
    recurrent_actor = args.recurrent
    recurrent_critic = args.recurrent

    kwargs = {
        "image_size": image_size,
        "image_feature_dim": image_feature_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": args.discount,
        "tau": args.tau,
        "recurrent_actor": recurrent_actor,
        "recurrent_critic": recurrent_critic
    }

    
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    kwargs["device"] = args.device
    policy = TD3.TD3(**kwargs)

   

    if args.load_model != "":
        policy.load(f"{file_prefix}{args.load_model}")

    if args.test:
        success_rate, collision_rate, avg_nav_time = eval_policy(policy, eval_env, eval_episodes=500, test=True)
        print('success_rate, collision_rate, avg_nav_time')
        print(success_rate, collision_rate, avg_nav_time)
        ####
        # seed_1 0.518 0.05 15.079
        # seed_2 0.222 0.088 12.086
        # seed_2 complex  0.4 0.012 13.751
        ####
        return

    replay_buffer = memory.ReplayBuffer(
        image_size, action_dim, args.hidden_size,
        args.memory_size, recurrent=recurrent_actor, device=args.device)

    # Evaluate untrained policy
    success_rate, collision_rate, avg_nav_time = eval_policy(policy, eval_env)
    print('success_rate, collision_rate, avg_nav_time at step 0')
    print(success_rate, collision_rate, avg_nav_time)
    evaluations = [success_rate]

    image  = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    hidden = policy.get_initial_states()

    for t in range(1, int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = np.random.uniform(-1.0, 1.0, action_dim)
            with torch.no_grad():
                _, next_hidden = policy.select_action(image, hidden)
        else:
            with torch.no_grad():
                a, next_hidden = policy.select_action(image, hidden)
            action = (
                a + np.random.normal(
                    0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
        if t == args.start_timesteps:
            print('replay buffer has been initialized')
        # Perform action
        next_image, reward, done, info = env.step(action)

        if isinstance(info, Timeout):
            done_bool = 0.0
        else:
            done_bool = float(done)


        # Store data in replay buffer
        replay_buffer.add(
            image, action, next_image, reward, done_bool, hidden, next_hidden)

        image = next_image
        hidden = next_hidden
        episode_reward += reward

        # Train agent after collecting sufficient data
        if (not policy.on_policy) and t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            if isinstance(info, ReachGoal):
                print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + ', goal reaching at train step: ' + str(episode_timesteps))
            elif isinstance(info, Collision):
                print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + ', collision occur at train step: ' + str(episode_timesteps))
            elif isinstance(info, Timeout):
                print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + ', time out: ' + str(episode_timesteps))
            else:
                raise ValueError('Invalid end signal from environment')
            # Reset environment
            image = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            hidden = policy.get_initial_states()

        # Evaluate episode
        if t % args.eval_freq == 0:
            file_name = 'step_' + str(t)
            success_rate, collision_rate, avg_nav_time = eval_policy(policy, eval_env)
            print('success_rate, collision_rate, avg_nav_time at step ' + str(t))
            print(success_rate, collision_rate, avg_nav_time)
            evaluations.append(success_rate)

            if args.save_model:
                policy.save(f'{file_prefix}/models/{file_name}')

            np.savetxt(f'{file_prefix}/results/{file_name}.txt', evaluations)


if __name__ == "__main__":
    # file_prefix = '/home/zw/expand_disk/ubuntu/RNN_RL_RAL_Image'
    # file_prefix = '/mnt/ssd1/zhuwei/RNN_RL_RAL_Image'
    file_prefix = './logdir'
    main()
