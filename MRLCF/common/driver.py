import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._act_spaces = [env.act_space for env in envs]
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    success = 0
    while step < steps or episode < episodes:
      obs = {
          i: self._envs[i].reset()
          for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
      for i, ob in obs.items():
        self._obs[i] = ob() if callable(ob) else ob
        act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
        self._eps[i] = [tran]
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      actions, self._state = policy(obs, self._state, **self._kwargs)
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      obs = [e.step(a) for e, a in zip(self._envs, actions)]
      obs = [ob() if callable(ob) else ob for ob in obs]
      for i, (act, ob) in enumerate(zip(actions, obs)):
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
        self._eps[i].append(tran)
        step += 1
        if ob['is_last']:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          [fn(ep, **self._kwargs) for fn in self._on_episodes]
          episode += 1
          success += ob['goal_reach_larger_area']
      self._obs = obs
    if episodes > 0:
      return (success / episodes)
    else: 
      return 0

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value

class DriverOrca(Driver):
  def __init__(self, envs, **kwargs):
    super().__init__(envs, **kwargs)
    self._full_state = True

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      obs_image = {}
      obs_proprioceptive = {}
      for i, ob_both in enumerate(self._obs):
        if ob_both is None or ob_both[0]['is_last']:
          obs_both = self._envs[i].reset(full_state=self._full_state)
        obs_image[i] = obs_both[0]
        obs_proprioceptive[i] = obs_both[1]

      for i, ob_image in obs_image.items():
        self._obs[i] = [ob_image, obs_proprioceptive[i]]
        act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob_image, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
        self._eps[i] = [tran]
      
      actions = []
      for ob_both in self._obs:
        actions.append({'action': policy.predict(ob_both[1])})
      
      assert len(actions) == len(self._envs)
      # zip: Get elements from multiple lists
      obs = [e.step(a, full_state=self._full_state) for e, a in zip(self._envs, actions)]
      for i, (act, ob_both) in enumerate(zip(actions, obs)):
        tran = {k: self._convert(v) for k, v in {**ob_both[0], **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
        self._eps[i].append(tran)
        step += 1
        if ob_both[0]['is_last']:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          [fn(ep, **self._kwargs) for fn in self._on_episodes]
          episode += 1
      self._obs = obs