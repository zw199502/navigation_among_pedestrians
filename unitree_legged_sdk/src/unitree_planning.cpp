/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <math.h>
#include <iostream>
#include <chrono>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <std_msgs/Float32MultiArray.h>
#include <ros/ros.h> 
#include <geometry_msgs/Twist.h>

using namespace UNITREE_LEGGED_SDK;

pthread_mutex_t command_lock;


class Custom
{
public:
    Custom(ros::NodeHandle* nodehandle); // constructor
    void UDPRecv();
    void UDPSend();
    void RobotControl();
    void cmd_vel_callback(const geometry_msgs::Twist::ConstPtr &msg);

    Control *control;
    UDP *udp;
    HighCmd cmd = {0};
    HighState state = {0};
    ros::NodeHandle nh_; // creating ROS NodeHandle
    ros::Subscriber Command_sub_;
    float dt = 0.002;     // 0.001~0.01
    float forward_speed = 0.0;
    float side_speed = 0.0;
    float rotate_speed = 0.0;
    int motion_mode = 0;
    const int N_forward = 9;
    const int N_backward = 9;
    const int N_sideward = 10;
    const float unitree_forward_speed[10] = {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45};
    const float real_forward_speed[10] = {0.0, 0.067, 0.143, 0.214, 0.291, 0.367, 0.436, 0.505, 0.580, 0.660};
    const float unitree_backward_speed[10] = {0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00};
    const float real_backward_speed[10] = {0.0, 0.063, 0.161, 0.240, 0.305, 0.381, 0.455, 0.524, 0.600, 0.690};
    const float unitree_sideward_speed[11] = {0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00};
    const float real_sideward_speed[11] = {0.0, 0.062, 0.090, 0.115, 0.138, 0.160, 0.190, 0.213, 0.234, 0.262, 0.272};
};

Custom::Custom(ros::NodeHandle* nodehandle):nh_(*nodehandle) //Constructor
{
    Command_sub_ = nh_.subscribe<geometry_msgs::Twist>("/cmd_vel", 5, &Custom::cmd_vel_callback, this, ros::TransportHints().tcpNoDelay());
    std::cout << "check point" << std::endl;
    control = new Control(LeggedType::A1, HIGHLEVEL);
    
    control->InitCmdData(cmd);
    udp = new UDP();
}

void Custom::cmd_vel_callback(const geometry_msgs::Twist::ConstPtr &msg)
{   
    float command_forward, command_sideward, command_rotation;
    float unitree_forward = 0.0, unitree_sideward = 0.0, unitree_rotation = 0.0;
    command_forward = msg->linear.x;
    command_sideward = msg->linear.y;
    command_rotation = msg->angular.z;
    if(command_forward > 1.0 || command_forward < -1.0){
        pthread_mutex_lock(&command_lock);
        forward_speed = 0.0;
        side_speed = 0.0;
        rotate_speed = 0.0;
        motion_mode = 1;  // stop
        pthread_mutex_unlock(&command_lock);
        return;
    }
    if(command_forward != 0){
        if(command_forward > 0){
            if(command_forward > real_forward_speed[N_forward]){
                command_forward = unitree_forward_speed[N_forward];
            }
            else{
                for(int i = 0; i < N_forward; i++){
                    if(command_forward > real_forward_speed[i] && command_forward <= real_forward_speed[i + 1]){
                        unitree_forward = (command_forward - real_forward_speed[i]) * (unitree_forward_speed[i + 1] - unitree_forward_speed[i]) / (real_forward_speed[i + 1] - real_forward_speed[i]) + unitree_forward_speed[i];
                        break;
                    }
                }
            }
            
        }
        else{
            command_forward = -command_forward;
            if(command_forward > real_backward_speed[N_backward]){
                command_forward = unitree_backward_speed[N_backward];
            }
            else{
                for(int i = 0; i < N_backward; i++){
                    if(command_forward > real_backward_speed[i] && command_forward <= real_backward_speed[i + 1]){
                        unitree_forward = (command_forward - real_backward_speed[i]) * (unitree_backward_speed[i + 1] - unitree_backward_speed[i]) / (real_backward_speed[i + 1] - real_backward_speed[i]) + unitree_backward_speed[i];
                        break;
                    }
                }
            }
            unitree_forward = -unitree_forward;
        }
    }
    
    if(command_sideward != 0){
        if(command_sideward < 0){
            command_sideward = -command_sideward;
            if(command_sideward > real_sideward_speed[N_sideward]){
                command_sideward = unitree_sideward_speed[N_sideward];
            }
            for(int i = 0; i < N_sideward; i++){
                if(command_sideward > real_sideward_speed[i] && command_sideward <= real_sideward_speed[i + 1]){
                    unitree_sideward = (command_sideward - real_sideward_speed[i]) * (unitree_sideward_speed[i + 1] - unitree_sideward_speed[i]) / (real_sideward_speed[i + 1] - real_sideward_speed[i]) + unitree_sideward_speed[i];
                    break;
                }
            }
            unitree_sideward = -unitree_sideward;
        }
        else{
            if(command_sideward > real_sideward_speed[N_sideward]){
                command_sideward = unitree_sideward_speed[N_sideward];
            }
            for(int i = 0; i < N_sideward; i++){
                if(command_sideward > real_sideward_speed[i] && command_sideward <= real_sideward_speed[i + 1]){
                    unitree_sideward = (command_sideward - real_sideward_speed[i]) * (unitree_sideward_speed[i + 1] - unitree_sideward_speed[i]) / (real_sideward_speed[i + 1] - real_sideward_speed[i]) + unitree_sideward_speed[i];
                    break;
                }
            }
        }
    }
    
    pthread_mutex_lock(&command_lock);
    motion_mode = 2;  // move
    forward_speed = unitree_forward;
    side_speed = unitree_sideward;
    rotate_speed = unitree_rotation;
    pthread_mutex_unlock(&command_lock);
    ROS_INFO("command_forward_speed: %.4f, command_side_speed: %.4f, unitree_forward_speed: %.4f, unitree_side_speed: %.4f", msg->linear.x, msg->linear.y, forward_speed, side_speed);
}

void Custom::UDPRecv()
{
    udp->Recv();
}

void Custom::UDPSend()
{  
    udp->Send();
}

void Custom::RobotControl() 
{
    udp->GetRecv(state);

    cmd.forwardSpeed = forward_speed;
    cmd.sideSpeed = side_speed;
    // cmd.rotateSpeed = rotate_speed;
    cmd.mode = motion_mode;

    udp->SetSend(cmd);
}

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "navigation_node");
    ros::NodeHandle node;

    std::cout << "Control level is set to HIGH-level." << std::endl
              << "WARNING: Make sure the robot is standing on the ground." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    
    Custom custom(&node);

    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();
    
    ros::Rate rate(100);
    while (node.ok()){
        ros::spinOnce();
        rate.sleep();
    };

    return 0; 
}
