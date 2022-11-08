# navigation_among_pedestrians
We proposed a model-based deep reinforcement learning algorithm for the navigation and collision-free motion planning among crowds. The baselines inlcude the EGO (https://ieeexplore.ieee.org/abstract/document/9197148), RGL (https://ieeexplore.ieee.org/abstract/document/9340705), SARL (https://ieeexplore.ieee.org/abstract/document/8794134), CADRL (https://ieeexplore.ieee.org/abstract/document/7989037), LSTM_RL (https://ieeexplore.ieee.org/abstract/document/8593871). We refer to the open-sourced project from https://github.com/ChanganVR/RelationalGraphLearning to implement RGL, SARL, CADRL and LSTM_RL. Because the EGO algorithm is not publicly available, we developed it on our own understanding.

# prerequisite
- pytorch, the version depends on your device. We found that the code from https://github.com/ChanganVR/RelationalGraphLearning doesn't support GPU.
