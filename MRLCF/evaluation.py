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

  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)

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
  eval_env = common.Navigation(config, 'final_eval')
  act_space = eval_env.act_space

  eval_driver = common.Driver([eval_env])
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, act_space, step)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables_95_435000.pkl').exists():
    agnt.load(logdir / 'variables_95_435000.pkl')
  else:
    print('Please train the model at first.')
    raise NotImplementedError()
    
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')
  print('start to evaluate')
  success_rate, average_time = eval_driver(eval_policy, episodes=500)
  print('success_rate: ', success_rate)
  print('average_time: ', average_time)


if __name__ == '__main__':
  # rospy.init_node('navigation', anonymous=True) #make node 
  main()
  # rospy.spin()
