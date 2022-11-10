/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <math.h>
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <std_msgs/Float32MultiArray.h>
#include <ros/ros.h> 

using namespace UNITREE_LEGGED_SDK;

pthread_mutex_t command_lock;
float forward_speed = 0.0;
float side_speed = 0.0;
float rotate_speed = 0.0;
int motiontime = 0;

void cmd_vel_callback(const std_msgs::Float32MultiArray::ConstPtr &msg)
{   
    pthread_mutex_lock(&command_lock);
    motiontime = 0;
    forward_speed = msg->data[0];
    if (forward_speed >= 0){
        forward_speed = forward_speed * 0.2 / 0.3;
    }
    else{
        forward_speed = forward_speed * 0.5 / 0.3;
    }
    if (forward_speed > 1.0){
        forward_speed = 1.0;
    }
    else{
        if (forward_speed < -1.0){
         forward_speed = -1.0;
        }
    }

    // dynamic navigation
    side_speed = msg->data[1] / 0.3;
    if (side_speed > 1.0){
        side_speed = 1.0;
    }
    else{
        if (side_speed < -1.0){
         side_speed = -1.0;
        }
    }
    // dynamic navigation


    // static naviagtion
    // rotate_speed = msg->data[1] * 1.5 / 3.1415926;  # normal ratio to the command
    // rotate_speed = msg->data[2] * 1.8 / 3.1415926;
    // if (rotate_speed > 1.0){
    //     rotate_speed = 1.0;
    // }
    // else{
    //     if (rotate_speed < -1.0){
    //      rotate_speed = -1.0;
    //     }
    // } 
    // static naviagtion

    // ROS_INFO("forward_speed: %.4f, side_speed: %.4f", forward_speed, side_speed);
    pthread_mutex_unlock(&command_lock);
}


class Custom
{
public:
    Custom(); // constructor
    void UDPRecv();
    void UDPSend();
    void RobotControl();

    Control *control;
    UDP *udp;
    HighCmd cmd = {0};
    HighState state = {0};
    float dt = 0.002;     // 0.001~0.01

};

Custom::Custom() //Constructor
{
    std::cout << "check point" << std::endl;
    control = new Control(LeggedType::A1, HIGHLEVEL);
    
    control->InitCmdData(cmd);
    udp = new UDP();
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
    motiontime += 2;
    udp->GetRecv(state);
    // printf("%d   %f\n", motiontime, state.forwardSpeed);

    cmd.forwardSpeed = forward_speed;

    // dynamic navigation
    cmd.sideSpeed = side_speed;
    // dynamic navigation

    // static navigation
    // cmd.rotateSpeed = rotate_speed;
    // static navigation

    cmd.mode = 2;
    // ROS_INFO("forward_speed: %.4f, side_speed: %.4f", forward_speed, side_speed);
    if (motiontime > 10000){
        cmd.forwardSpeed = 0.0f;

        // dynamic navigation
        cmd.sideSpeed = 0.0f;
        // dynamic navigation

        // static navigation
        // cmd.rotateSpeed = 0.0f;
        // static navigation
        cmd.mode = 1;
    }
    // cmd.rotateSpeed = 0.0f;
    // cmd.bodyHeight = 0.0f;

    // cmd.mode = 0;
    // cmd.roll  = 0;
    // cmd.pitch = 0;
    // cmd.yaw = 0;

    // if(motiontime>500 && motiontime<1000){
    //     cmd.mode = 1;
    //     cmd.bodyHeight = -0.3f;
    // }

    // if(motiontime>1000 && motiontime<1500){
    //     cmd.mode = 1;
    //     cmd.bodyHeight = 0.3f;
    // }

    // if(motiontime>1500 && motiontime<2000){
    //     cmd.mode = 1;
    //     cmd.bodyHeight = 0.0f;
    // }

    // if(motiontime>2000 && motiontime<3000){
    //     cmd.mode = 2;
    // }
    // command forward -0.43
    // x1 = 1.0, y1 = 0.245
    // x2 = 0.166, y2 = 0.319 
    // command forward -0.43
    // actual 0.279

    // command side -0.75
    // x1 = 0.166, y1 = 0.319 
    // x2 = 0.22, y2 = -0.195
    // command side -0.75
    // actual 0.172

    // command forward 0.3
    // x1 = 0.22, y1 = -0.195
    // x2 = 1.587, y2 = 0.032
    // command forward 0.3
    // actual 0.462

    // command side -1.0
    // x1 = 0.583, y1 = -0.435
    // x2 = 0.451, y2 = -1.21 
    // command side -1.0
    // actual 0.262

    // command forward 0.2
    // x1 = 0.477, y1 = -1.139
    // x2 = 1.257, y2 = -0.625
    // command forward 0.2
    // actual 0.311

    // if(motiontime>3000 && motiontime<6000){ // 0.002 * 1500 = 3s
    //     cmd.mode = 2;
    //     cmd.forwardSpeed = 0.2f; // -1  ~ +1
    //     // cmd.sideSpeed = -1.0f; 
    // }

    // if(motiontime>6000 ){
    //     cmd.mode = 1;
    // }

    udp->SetSend(cmd);
}

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "navigation_node");
    ros::NodeHandle node;
    ros::Subscriber sub_command = node.subscribe<std_msgs::Float32MultiArray>("/cmd_vel", 5, cmd_vel_callback, ros::TransportHints().tcpNoDelay());

    std::cout << "Control level is set to HIGH-level." << std::endl
              << "WARNING: Make sure the robot is standing on the ground." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    
    Custom custom;

    

    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();
    
    ros::Rate rate(20);
    while (node.ok()){
        ros::spinOnce();
        rate.sleep();
    };

    return 0; 
}
