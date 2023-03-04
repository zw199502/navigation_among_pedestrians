from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'cadrl'


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False, il_random=False):
        super(TrainConfig, self).__init__(debug, il_random)
