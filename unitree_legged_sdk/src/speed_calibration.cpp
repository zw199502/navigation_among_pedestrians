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
#include <geometry_msgs/PoseStamped.h>

using namespace UNITREE_LEGGED_SDK;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

pthread_mutex_t command_lock;




class Custom
{
public:
    Custom(ros::NodeHandle* nodehandle); // constructor
    void UDPRecv();
    void UDPSend();
    void RobotControl();
    void pose_callback(const geometry_msgs::PoseStamped::ConstPtr &msg);

    Control *control;
    UDP *udp;
    HighCmd cmd = {0};
    HighState state = {0};
    ros::NodeHandle nh_; // creating ROS NodeHandle
    ros::Subscriber Pose_sub_ ;
    float dt = 0.002;     // 0.001~0.01
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    float forward_speed = 0.0;
    float side_speed = 0.0;
    float rotate_speed = 0.0;
    
    int motiontime = 0;
    long t1 = 0;
    long t2 = 0;
    double x_s = 0.0;
    double y_s = 0.0;
    double theta = 0.0;
    double x_e = 0.0;
    double y_e = 0.0;
    double speed_x = 0.0, speed_y = 0.0;

};

Custom::Custom(ros::NodeHandle* nodehandle):nh_(*nodehandle) //Constructor
{
    Pose_sub_  = nh_.subscribe<geometry_msgs::PoseStamped>("/vrpn_client_node/QuadrupedRobot/pose", 20, &Custom::pose_callback, this, ros::TransportHints().tcpNoDelay());
    std::cout << "check point" << std::endl;
    control = new Control(LeggedType::A1, HIGHLEVEL);
    
    control->InitCmdData(cmd);
    udp = new UDP();
    
}

void Custom::pose_callback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{   
    // pthread_mutex_lock(&command_lock);
    x = msg->pose.position.x;
    y = msg->pose.position.y;
    double q_x = msg->pose.orientation.x;
    double q_y = msg->pose.orientation.y;
    double q_z = msg->pose.orientation.z;
    double q_w = msg->pose.orientation.w;
    yaw = atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z));
    // pthread_mutex_unlock(&command_lock);
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
    if(motiontime <= 3000){
        x_s = x;
        y_s = y;
        theta = yaw;
        if (motiontime == 3000){
            t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout<<"time start: "<<t1<<std::endl;
            std::cout<<"x_s: "<<x_s<<"  y_s: "<<y_s<<"  theta: "<<theta<<std::endl;
        }
        
    }
    motiontime += 2;
    udp->GetRecv(state);
    // printf("%d   %f\n", motiontime, state.forwardSpeed);

    if(motiontime >= 2000 && motiontime < 6500)
    {
        cmd.forwardSpeed = 0.45;
        cmd.sideSpeed = 0.0;

        // static navigation
        // cmd.rotateSpeed = rotate_speed;
        // static navigation

        cmd.mode = 2;
    }
    
    // ROS_INFO("forward_speed: %.4f, side_speed: %.4f", forward_speed, side_speed);
    if (motiontime >= 6500){
        cmd.forwardSpeed = 0.0f;

        // dynamic navigation
        cmd.sideSpeed = 0.0f;
        // dynamic navigation

        // static navigation
        // cmd.rotateSpeed = 0.0f;
        // static navigation
        cmd.mode = 1;
        if(motiontime == 6500){
            t2 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout<<"time end: "<<t2<<std::endl;
            std::cout<<"delta time: "<<(t2 - t1)<<std::endl;
            x_e = x;
            y_e = y;
            std::cout<<"x_e: "<<x_e<<"  y_e: "<<y_e<<std::endl;
            double dy = y_e - y_s;
            double dx = x_e - x_s;
            double st = sin(theta);
            double ct = cos(theta);
            double x_trans = dy * st + dx * ct;
            double y_trans = dy * ct - dx * st;
            speed_x = x_trans / (t2 - t1) * 1000.0;
            speed_y = y_trans / (t2 - t1) * 1000.0;
            // std::cout << x_s << ", " << y_s << ", " << x_e << ", " << y_e << ", " << t1 << ", " << t2 << std::endl;
            std::cout<<"speed_x: "<<speed_x<<"  speed_y: "<<speed_y<<std::endl;
        }
        
    }


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
