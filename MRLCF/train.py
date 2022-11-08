import collections
import logging
import os
import pathlib
import re
import sys
import warnings
# import rospy

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common

import tensorflow as tf

def main():


  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.list_physical_devices('GPU'), message
  gpus = tf.config.list_physical_devices('GPU')
  # specify the gpu
  gpu_index = config.gpu
  tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
  device = 'gpu:' + str(gpu_index)
  print(device)

  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.replay.minlen,
      maxlen=config.replay.maxlen))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until)

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  print('Create envs.')
  # multiple environments can be created
  # here, only one environment is built
  train_env = common.Navigation(config, 'train')
  eval_env = common.Navigation(config, 'eval')

  act_space = train_env.act_space
  train_driver = common.Driver([train_env])
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda transition, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver([eval_env])
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(act_space)
    if config.ORCA:  # some bugs exist, please DO NOT use this mode temporarily
      ORCA_agent = common.ORCA(safety_space=config.safety_space, 
                               time_step=config.time_step, 
                               max_speed=config.max_robot_speed, 
                               radius=config.robot_radius)
      train_driver_orca = common.DriverOrca([train_env])
      train_driver_orca.on_episode(lambda ep: per_episode(ep, mode='train'))
      train_driver_orca.on_step(lambda transition, worker: step.increment())
      train_driver_orca.on_step(train_replay.add_step)
      train_driver_orca.on_reset(train_replay.add_step)
      train_driver_orca(ORCA_agent, steps=prefill, episodes=1)
    else:
      train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=5)
    train_driver.reset()
    eval_driver.reset()

  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, act_space, step)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  def train_step(transition, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)
  train_driver.on_step(train_step)

  success_rate_array = []
  while step < config.steps:
    logger.write()
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    success_rate = eval_driver(eval_policy, episodes=config.eval_eps)
    print('evaluation success rate: ', success_rate)
    success_rate_array.append(np.array([int(step), success_rate]))
    np.savetxt('success_rate.txt', np.array(success_rate_array))
    print('Start training.')
    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')

if __name__ == '__main__':
  # rospy.init_node('navigation', anonymous=True) #make node 
  main()
  # rospy.spin()
