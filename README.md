# navigation_among_pedestrians
We proposed a model-based deep reinforcement learning algorithm for the navigation and collision-free motion planning among crowds. The baselines inlcude the EGO (https://ieeexplore.ieee.org/abstract/document/9197148), RGL (https://ieeexplore.ieee.org/abstract/document/9340705), SARL (https://ieeexplore.ieee.org/abstract/document/8794134), CADRL (https://ieeexplore.ieee.org/abstract/document/7989037), LSTM_RL (https://ieeexplore.ieee.org/abstract/document/8593871), and ORCA (https://link.springer.com/chapter/10.1007/978-3-642-19457-3_1). We refer to the open-sourced project from https://github.com/ChanganVR/RelationalGraphLearning to implement RGL, SARL, CADRL and LSTM_RL. Because the EGO algorithm is not publicly available, we developed it on our own understanding.

# prerequisite
- pytorch, the version depends on your device. We found that the code from https://github.com/ChanganVR/RelationalGraphLearning doesn't support GPU.
- Cython
- socialforce, https://github.com/ChanganVR/socialforce
- gym
- tensorflow-gpu, version >= 2.4
- tensorflow-probability, the version should match the version of tensorflow
- Python-RVO2,https://github.com/sybrenstuvel/Python-RVO2, only Linux supported
- rospks, if you want to use ROS to deploy your algorithm on mobile robots
- ros-navigation and ros-gmapping, optional
- LCM, if you use Unitree A1, https://github.com/unitreerobotics/unitree_legged_sdk/tree/v3.3.4
- Velodyne Lidar, https://github.com/ros-drivers/velodyne
- if you are lack of any packages, please install them by yourselves
- please create two environments with anaconda, one for pytorch and the other for tensorflow
- we are using ubuntu20.04 and ros--noetic. If you use other versions, you may have some problems.

# fold introduction
### crowd_nav_lidar_scan_ego
- enter the directory C_library and compile the cython file
```python setup.py build_ext --inplace```. This Cython file is used to simulate Lidar scan
- revise the configuration from train.py
- particularly, select whether to use imitation learning or not in the train.py file
```parser.add_argument('--if_orca', default=True, action='store_true')```, and specify your GPU ```gpu_index = 0```. Normally, if without imitation learning, the training result will be very terrible
- train your model, ```python train.py```
- test, ```python test.py```, load your own network weights ```model_weight_file = os.path.join(args.output_dir, 'weight_episode_12000.h5')```

### RelationalGraphLearning
- enter the fold RelationalGraphLearning and install the project, ```pip install -e .```
- select whether to use imitation learning or not in the crowd_nav/train.py file, ```parser.add_argument('--il_random', default=False, action='store_true')```
- revise the log file name in crowd_nav/train.py file, ```parser.add_argument('--output_dir', type=str, default='data/sarl')```
- train your model in the directory crowd_nav, ```python train.py --policy rgl```, you can replace the rgl with sarl, cadrl, and lstm_rl
- test the model ```python test.py --policy rgl```, change the model directory ```parser.add_argument('-m', '--model_dir', type=str, default='/data_rgl_il/rgl')```, select the network weights ```model_weights = os.path.join(args.model_dir, 'rl_model_4.pth')```

### MRLCF
- train your model, ```python train.py --logdir ./logdir/online/1 --configs online```, another configs 'quadruped_motion_capture' is used for a quadruped robot. if your want to use your own robots, please revise the configs.yaml
- visualize the training, ```tensorboard --logdir ./logdir```

### ORCA
- run orca policy, ```python ORCA_policy.py```
- change the human number in crowd_sim.py, ```self.human_num = 5```

### unitree_legged_sdk
- this is a ros package, please compile it with ```catkin_make```
- motion_capture.launch, if you have a motion capture system to localize the robot
- control_via_keyboard, control the robot via keyboard
- speed_calibration.launch, the speed command does not match real speed, e.g. if you send forward speed 0.3m/s, the real speed may be 0.2m/s
- unitree_planning, receive the command velocities and control the real robot
- please run the launch file in the ```sudo su``` mode to get the permission of memory

### A_LOAM
- it is a ros package, please compile it with ```catkin_make```
- aloam_velodyne_VLP16.launch, SLAM algorithm in 3D space to localize the robot
- human_detection.launch, detect humans on a specific area and localize the robot.
- gmapping.launch, create grid map on 2D plane
- quadruped_dwa.launch, move_base

# Real experiments
- the robot moves on a 4x4m area
- first, compile your Velodyne ros package, connect the Velodyne Lidar sensor, and use the command ```curl http://192.168.1.201/cgi/short_dist --data "enable=on"``` to shorten the default minimum range of the Lidar. The default minimum range is 0.5m and the shorted range is 0.1m. Although the range is shortened, the accuracy and stability are degraded
- second, compile the A_LOAM ros package, ```roslauch  roslaunch aloam_velodyne human_detection.launch```. you need to wait 5 seconds and then let pedestrians move into the area.
- third, compile the unitree ros package, ```roslaunch unitree_legged_sdk unitree_planning.launch```
- fourth, run the DRL-based motion planner
- please source all ros packages in the ~/.bashrc file, like ```source XXX/devel/setup.bash```
- please study our source codes for more details
- please feel free to contact us if you have any problems, zhu.wei.r5@dc.tohoku.ac.jp
