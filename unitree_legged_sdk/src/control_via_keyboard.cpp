#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <termios.h>
#include <stdio.h> 
#include <pthread.h>
#include <std_msgs/Float32MultiArray.h>

int ch;
pthread_mutex_t lock;

void* getch(void *arg)
{
    
	int in;
    struct termios new_settings;
    struct termios stored_settings;
    tcgetattr(0,&stored_settings);
    new_settings = stored_settings;
    new_settings.c_lflag &= (~ICANON);
    new_settings.c_cc[VTIME] = 0;
    tcgetattr(0,&stored_settings);
    new_settings.c_cc[VMIN] = 1;
    tcsetattr(0,TCSANOW,&new_settings);
    
	while(ch != 'q'){
		in = getchar();
		pthread_mutex_lock(&lock);
		ch = in;
		pthread_mutex_unlock(&lock);
		switch (ch)
		{
		case 'q':
			printf("  already quit!\n");

		case 'w':
			printf("  move forward!\n");
			break;

		case 's':
			printf("  move backward!\n");
			break;

		case 'r':
			printf("  stop forward!\n");
			break;

		case 'a':
			printf("  move left!\n");
			break;

		case 'd':
			printf("  move right!\n");
			break;

		case 'f':
			printf("  stop sideforward!\n");
			break;

		case 'j':
			printf("  turn left!\n");
			break;

		case 'l':
			printf("  turn right!\n");
			break;

		default:
			// printf("Stop!\n");
			break;
		}
	}
    
    tcsetattr(0,TCSANOW,&stored_settings);
	return NULL;
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "keyboard_input_node");

	ros::NodeHandle nh;

	ros::Rate loop_rate(20);

	ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 5);
	// ros::Publisher pub = nh.advertise<std_msgs::Float32MultiArray>("cmd_vel", 1);
	

	if (pthread_mutex_init(&lock, NULL) != 0)
    {
        printf("\n mutex init failed\n");
        return 1;
    }

	pthread_t keyboard_thread;
	int error = pthread_create(&keyboard_thread, NULL, &getch, NULL);
	if (error != 0)
		printf("\ncan't create thread :[%s]", strerror(error));
	else
		printf("\n Thread created successfully\n");

	geometry_msgs::Twist twist;
	twist.linear.x = 0.0;
	twist.linear.y = 0.0;
	twist.linear.z = 0.0;
	twist.angular.x = 0.0;
	twist.angular.y = 0.0;
	twist.angular.z = 0.0;

	long count = 0;

	while (ros::ok())
	{
		switch (ch)
		{
		case 'q':
			return 0;

		case 'w':
			twist.linear.x = 0.2;
			break;

		case 's':
			twist.linear.x = -0.2;
			break;

		case 'r':
			twist.linear.x = 0.0;
			break;

		case 'a':
			twist.linear.y = 0.2;
			break;

		case 'd':
			twist.linear.y = -0.2;
			break;

		case 'f':
			twist.linear.y = 0.0;
			break;

		case 'j':
			twist.angular.z = 0.2;
			break;

		case 'l':
			twist.angular.z = -0.2;
			break;

		default:
			// printf("Stop!\n");
			break;
		}
		// std_msgs::Float32MultiArray velocities;
		// velocities.data.clear();
		// velocities.data.push_back(twist.linear.x);
		// velocities.data.push_back(twist.linear.y);
		// velocities.data.push_back(twist.angular.z);
		pub.publish(twist);

		ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}
