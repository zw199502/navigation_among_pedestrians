# A-LOAM
## Advanced implementation of LOAM

A-LOAM is an Advanced implementation of LOAM (J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time), which uses Eigen and Ceres Solver to simplify code structure. This code is modified from LOAM and [LOAM_NOTED](https://github.com/cuitaixiang/LOAM_NOTED). This code is clean and simple without complicated mathematical derivation and redundant operations. It is a good learning material for SLAM beginners.

<img src="https://github.com/HKUST-Aerial-Robotics/A-LOAM/blob/devel/picture/kitti.png" width = 55% height = 55%/>

**Modifier:** [Tong Qin](http://www.qintonguav.com), [Shaozu Cao](https://github.com/shaozu)


## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04.
ROS Kinetic or Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)


### 1.2. **Ceres Solver**
Download ceres 1.14
https://github.com/ceres-solver/ceres-solver/releases/tag/1.14.0
cd ceres-solver
mkdir build
cd build
cmake ..
make -j3
sudo make install

### 1.3. **PCL**
Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).
refer https://blog.csdn.net/m0_48919875/article/details/123863892

sudo apt-get update
sudo apt-get install git build-essential linux-libc-dev
sudo apt-get install cmake cmake-gui
sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev
sudo apt-get install mpi-default-dev openmpi-bin openmpi-common
sudo apt-get install libflann1.9 libflann-dev  # ubuntu20.4对应1.9
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libqhull* libgtest-dev
sudo apt-get install freeglut3-dev pkg-config
sudo apt-get install libxmu-dev libxi-dev
sudo apt-get install mono-complete
sudo apt-get install libopenni-dev
sudo apt-get install libopenni2-dev

sudo apt-get install libx11-dev libxext-dev libxtst-dev libxrender-dev libxmu-dev libxmuu-dev
sudo apt-get install build-essential libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install cmake cmake-gui

Download VTK 7.1.1.zip from https://vtk.org/download/#earlier
unzip vtk and create a folder 'build'
openn a terminate:
cmake-gui 
choose VTK source fold
choose the build directory in the source fold
configure
generate

in the build directory in the source fold
cmake -DCMAKE_TYPE=None ..
make -j8
sudo make install


## 2. Build A-LOAM

Clone the repository and catkin_make:

```
    cd ~/catkin_ws/src
    git clone https://github.com/HKUST-Aerial-Robotics/A-LOAM.git
    cd ../
    catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3  -DCATKIN_WHITELIST_PACKAGES="cloud_msgs"
    source ~/catkin_ws/devel/setup.bash
```

## 3. ALOAM

when connecting Velodyne, enable smaller range

```
    curl http://192.168.1.201/cgi/short_dist --data "enable=on"
```

```
    roslaunch aloam_velodyne aloam_velodyne_VLP_16.launch
```

## 4. gmapping

when connecting Velodyne, enable smaller range

```
    curl http://192.168.1.201/cgi/short_dist --data "enable=on"
```

```
    roslaunch aloam_velodyne gmapping.launch
```

in the unitree workspace

```
    sudo su # root mode
    source devel/setup.bash
    roslaunch unitree_legged_sdk control_via_keyboard.launch
```

locomote for a while to get precise odometry from ALOAM, then execute gmapping

```
    roslaunch aloam_velodyne gmapping.launch 
```

save grid map
```
     rosrun map_server map_saver -f /home/zw/ws_loam/src/A_LOAM/map/map
```

## move base

when connecting Velodyne, enable smaller range

```
    curl http://192.168.1.201/cgi/short_dist --data "enable=on"
```

```
    roslaunch aloam_velodyne quadruped_dwa.launch
```

in the unitree workspace

```
    sudo su # root mode
    source devel/setup.bash
    roslaunch unitree_legged_sdk control_via_keyboard.launch
```

locomote for a while to get precise odometry from ALOAM, next stop and terminate the control_via_keyboard node, then execute the and planning node the move base node

```
    roslaunch unitree_legged_sdk unitree_planning.launch
    roslaunch aloam_velodyne quadruped_move_base.launch 
```


