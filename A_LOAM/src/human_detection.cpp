#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <tuple>
#include <chrono>
#include <mutex>
#include <thread>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/PointCloud2.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#define MAP_SIZE 200  //pixel
#define MAP_SIZE_FLATTEN 40000
#define MAX_INITIAL_GRID_MAP 50

const float map_range = 4.0; // meter 
const float half_map_range = 2.0;
const float map_resolution = 0.02;

// maximum humans: 3
ros::Publisher punPosePerson1, punPosePerson2, punPosePerson3;
ros::Publisher pubPointsPerson1, pubPointsPerson2, pubPointsPerson3, pubPointsPersons;
// initial grip map without pedestrians
int initial_grid_map[MAP_SIZE_FLATTEN] = {0};
int initial_grid_map_left = 0;
int initial_grid_map_right = MAP_SIZE - 1;
int count_grip_map = 0;
// pedestrian pointcloud in the world frame
pcl::PointCloud<PointType> PedestrianPointCloud;
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);
std::mutex RobotPoseMutex;

struct Box
{
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
};


Box BoundingBox(pcl::PointCloud<PointType>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointType minPoint, maxPoint;
    /*Get min and max coordinates in the cluster*/
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

	return box;
}

// Create clusters based on distance of points. 
// KD Tree based on euclidean distance is used to cluster points into cluster of obstacles
std::vector<pcl::PointCloud<PointType>::Ptr> create_clusters(pcl::PointCloud<PointType>::Ptr cloud){

    // Array to store individual clusters
    std::vector<pcl::PointCloud<PointType>::Ptr> clusters;

    // Initialize KD Tree with cloud
    pcl::search::KdTree<PointType>::Ptr kd_tree (new pcl::search::KdTree<PointType>);
    kd_tree->setInputCloud(cloud);

    // Declare variables for clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;

    // Set parameters for clustering
    ec.setClusterTolerance(0.1); // search radius
	ec.setMinClusterSize(250);
	ec.setMaxClusterSize(800);
	ec.setSearchMethod(kd_tree);
	ec.setInputCloud(cloud); 

    // Extract clusters   
    ec.extract(cluster_indices);

    // Append individual clusters to an array
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){

        pcl::PointCloud<PointType>::Ptr cloud_cluster (new pcl::PointCloud<PointType>);

        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
            cloud_cluster->points.push_back(cloud->points[*pit]);
        }

        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // std::cout<< "Cluster with " << cloud_cluster->points.size() << " points" << std::endl;
        clusters.push_back(cloud_cluster);
    
    }

    return clusters;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

void create_grid_map(float x, float y, float z, float intensity, uint8_t *temp_map){
	/*world Cartesian coordinate
									   ^ X axis
		                               |
									   |
									   |
									   |
									   |
									   |
									   |
		<-------------------------------
		Y axis
	*/
	if(z > -0.3 && z < 1.7 && x > -map_range && x < map_range && y > -map_range && y < map_range){
		// point to map
		PointType pointOriginal, pointMap;
		pointOriginal.x = x;
		pointOriginal.y = y;
		pointOriginal.z = z;
        pointOriginal.intensity = intensity;
		pointAssociateToMap(&pointOriginal, &pointMap);
		if(pointMap.x > -half_map_range && pointMap.x < half_map_range && pointMap.y > -half_map_range && pointMap.y < half_map_range){
			int grid_h = (half_map_range - pointMap.x) / map_resolution;
			int grid_w = (half_map_range - pointMap.y) / map_resolution;
			if(grid_h >= 0 && grid_h < MAP_SIZE && grid_w >= 0 && grid_w < MAP_SIZE){
				if(count_grip_map < MAX_INITIAL_GRID_MAP){
					initial_grid_map[grid_h * MAP_SIZE + grid_w] += 1;
				}
				else if(count_grip_map == MAX_INITIAL_GRID_MAP){
					// figure out the left and right margins
					for(int i = 0; i < MAP_SIZE_FLATTEN; i++){
						if(initial_grid_map[i] > 0.8 * MAX_INITIAL_GRID_MAP){
							if(grid_w < MAP_SIZE / 2 && grid_w > initial_grid_map_left){
								initial_grid_map_left = grid_w;
							}
							else if(grid_w > MAP_SIZE / 2 && grid_w < initial_grid_map_right){
								initial_grid_map_right = grid_w;
							}
						}
					}
				}
				else{
					// remove marginal static obstacles
					if(grid_w > initial_grid_map_left + 1 && grid_w < initial_grid_map_right - 1){
						*(temp_map + grid_h * MAP_SIZE + grid_w) = 255;
						// add the point which may belong to pedestrians
                        PointType temp_point;
                        temp_point.x = pointMap.x;
                        temp_point.y = pointMap.y;
                        temp_point.z = pointMap.z;
                        temp_point.intensity = pointMap.intensity;
                        // std::cout<<"x: "<<temp_point.x<<"  y: "<<temp_point.y<<"  z: "<<temp_point.z<<std::endl;
						PedestrianPointCloud.push_back(temp_point);
					}
				}
			}
		}
	}
}

// receive original velodyne pointcloud
// extract humans in the world frame
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
	TicToc t_whole;
	// Load structure of PointCloud2
	int offset_x = -1;
	int offset_y = -1;
	int offset_z = -1;
	int offset_i = -1;
	int offset_r = -1;

	for (size_t i = 0; i < laserCloudMsg->fields.size(); i++)
	{
		if (laserCloudMsg->fields[i].datatype == sensor_msgs::PointField::FLOAT32)
		{
			if (laserCloudMsg->fields[i].name == "x")
			{
				offset_x = laserCloudMsg->fields[i].offset;
			}
			else if (laserCloudMsg->fields[i].name == "y")
			{
				offset_y = laserCloudMsg->fields[i].offset;
			}
			else if (laserCloudMsg->fields[i].name == "z")
			{
				offset_z = laserCloudMsg->fields[i].offset;
			}
			else if (laserCloudMsg->fields[i].name == "intensity")
			{
				offset_i = laserCloudMsg->fields[i].offset;
			}
		}
		else if (laserCloudMsg->fields[i].datatype == sensor_msgs::PointField::UINT16)
		{
			if (laserCloudMsg->fields[i].name == "ring")
			{
				offset_r = laserCloudMsg->fields[i].offset;
			}
		}
	}

	// create a occupation map and extract human positions
	if ((offset_x >= 0) && (offset_y >= 0) && (offset_r >= 0))
	{
		uint8_t map_flatten[MAP_SIZE_FLATTEN] = {0};

		PedestrianPointCloud.clear();  // clear all points
		
		if ((offset_x == 0) &&
			(offset_y == 4) &&
			(offset_i % 4 == 0) &&
			(offset_r % 4 == 0))
		{

			const size_t X = 0;
			const size_t Y = 1;
			const size_t Z = 2;
            const size_t INTENSITY = offset_i / 4;
            
			for (sensor_msgs::PointCloud2ConstIterator<float> it(*laserCloudMsg, "x"); it != it.end(); ++it)
			{
				const float x = it[X];  // x
				const float y = it[Y];  // y
				const float z = it[Z];  // z
                const float intensity = it[INTENSITY];  // intensity
				create_grid_map(x, y, z, intensity, map_flatten);
			}
		}
		else
		{
			ROS_WARN_ONCE("VelodyneLaserScan: PointCloud2 fields in unexpected order. Using slower generic method.");
			sensor_msgs::PointCloud2ConstIterator<uint16_t> iter_r(*laserCloudMsg, "ring");
			sensor_msgs::PointCloud2ConstIterator<float> iter_x(*laserCloudMsg, "x");
			sensor_msgs::PointCloud2ConstIterator<float> iter_y(*laserCloudMsg, "y");
			sensor_msgs::PointCloud2ConstIterator<float> iter_z(*laserCloudMsg, "z");
            sensor_msgs::PointCloud2ConstIterator<float> iter_i(*laserCloudMsg, "intensity");
			for ( ; iter_r != iter_r.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_i)
			{
				const float x = *iter_x;  // x
				const float y = *iter_y;  // y
				const float z = *iter_z;  // z
                const float intensity = *iter_i;  // intensity
				create_grid_map(x, y, z, intensity, map_flatten);
			}

		}
        // std::cout<<"point extraction done"<<std::endl;
		if(count_grip_map > MAX_INITIAL_GRID_MAP){

            if(!PedestrianPointCloud.empty()){
                TicToc t_cluster;
                pcl::PointCloud<PointType>::Ptr cloud;
                cloud = PedestrianPointCloud.makeShared();

                sensor_msgs::PointCloud2 pointsPersonsMsg;
                pcl::toROSMsg(PedestrianPointCloud, pointsPersonsMsg);
                pointsPersonsMsg.header.stamp = laserCloudMsg->header.stamp;
                pointsPersonsMsg.header.frame_id = "camera_init";
                pubPointsPersons.publish(pointsPersonsMsg);

                // Output cloud for downsampled cloud
                pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>);
                // Downsample the point cloud to lower memory usage and faster processing
                pcl::VoxelGrid<PointType> voxel_grid_filter;
                voxel_grid_filter.setInputCloud(cloud);
                voxel_grid_filter.setLeafSize (0.02f, 0.02f, 0.02f); // voxel size, perhaps 0.06 is better
                voxel_grid_filter.filter(*cloud_filtered); 
                // Create clusters of obstacle from raw points
                std::vector<pcl::PointCloud<PointType>::Ptr> clusters = create_clusters(cloud_filtered);

                // bounding box around obstacles
                int clusterId = 0;

                for(pcl::PointCloud<PointType>::Ptr cluster : clusters)
                {
                    Box box = BoundingBox(cluster);
                    float delta_z = box.z_max - box.z_min;
                    std::cout<<"clusterId: "<<clusterId<<"  delta_z: "<<delta_z<<std::endl;

                    ++clusterId;
                    if(clusterId == 1){
                        sensor_msgs::PointCloud2 pointsPerson1Msg;
                        pcl::toROSMsg(*cluster, pointsPerson1Msg);
                        pointsPerson1Msg.header.stamp = laserCloudMsg->header.stamp;
                        pointsPerson1Msg.header.frame_id = "camera_init";
                        // pubPointsPerson1.publish(pointsPerson1Msg);
                    }
                    else if(clusterId == 2){
                        sensor_msgs::PointCloud2 pointsPerson2Msg;
                        pcl::toROSMsg(*cluster, pointsPerson2Msg);
                        pointsPerson2Msg.header.stamp = laserCloudMsg->header.stamp;
                        pointsPerson2Msg.header.frame_id = "camera_init";
                        // pubPointsPerson2.publish(pointsPerson2Msg);
                    }
                    else if(clusterId == 3){
                        sensor_msgs::PointCloud2 pointsPerson3Msg;
                        pcl::toROSMsg(*cluster, pointsPerson3Msg);
                        pointsPerson3Msg.header.stamp = laserCloudMsg->header.stamp;
                        pointsPerson3Msg.header.frame_id = "camera_init";
                        // pubPointsPerson3.publish(pointsPerson3Msg);
                    }
                }
                // printf("cluster time %f ms \n", t_cluster.toc()); // around 1ms
            }

			// cv::Mat labels, stats, centroids;
			// cv::Mat map = cv::Mat(MAP_SIZE, MAP_SIZE, CV_8UC1, map_flatten);
			// cv::Mat dilated_map;
			// cv::Mat kernel = cv::getStructuringElement(0, cv::Size(2, 2));
			// cv::dilate(map, dilated_map, kernel);
			// int n_connected_areas = cv::connectedComponentsWithStats(
			// 	dilated_map, //binary image, pixel value either 0 or 255
			// 	labels,     //labelled image with the same size as the orginal image
			// 	stats, //nccomps×5, x_top_left, y_top_left, width, height, area size
			// 	centroids //nccomps×2, x, y
			// );
			// printf("whole gmapping time %f ms +++++\n", t_whole.toc());  // about 0.5ms
			// for (int i = 1; i < n_connected_areas; i++) {
			// 	cv::Vec2d pt = centroids.at<cv::Vec2d>(i, 0);
			// 	int x_left = stats.at<int>(i, cv::CC_STAT_LEFT);
			// 	int y_top = stats.at<int>(i, cv::CC_STAT_TOP);
			// 	int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
			// 	int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
			// 	int area = stats.at<int>(i, cv::CC_STAT_AREA);
			// 	printf("count_grid_map : %d, area : %d, center point(%.2f, %.2f)\n", count_grip_map, area, pt[0], pt[1]);  // area size
			// }

			// cv::imshow("grid_map", map);
			// cv::waitKey(10);

			geometry_msgs::PoseStamped PosePerson1;
			PosePerson1.header = laserCloudMsg->header;
			PosePerson1.pose.position.x = 0.0;
			punPosePerson1.publish(PosePerson1);
			
		}
		count_grip_map++;
	}
	else
	{
		ROS_ERROR("VelodyneLaserScan: PointCloud2 missing one or more required fields! (x,y,ring)");
	}
}

void robotPoseHandler(const geometry_msgs::PoseStampedConstPtr &robotPoseMsg)
{
    RobotPoseMutex.lock();
    t_w_curr.x() = robotPoseMsg->pose.position.x;
    t_w_curr.y() = robotPoseMsg->pose.position.y;
    t_w_curr.z() = robotPoseMsg->pose.position.z;
    q_w_curr.x() = robotPoseMsg->pose.orientation.x;
    q_w_curr.y() = robotPoseMsg->pose.orientation.y;
    q_w_curr.z() = robotPoseMsg->pose.orientation.z;
    q_w_curr.w() = robotPoseMsg->pose.orientation.w;
    RobotPoseMutex.unlock();
}

int main(int argc, char **argv){

    ros::init(argc, argv, "LIDAR Obstacle Detection");
	ros::NodeHandle nh;

	ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    ros::Subscriber subRobotPose = nh.subscribe<geometry_msgs::PoseStamped>("/robot/pose", 100, robotPoseHandler);
    punPosePerson1 = nh.advertise<geometry_msgs::PoseStamped>("/person1/pose", 10);
	punPosePerson2 = nh.advertise<geometry_msgs::PoseStamped>("/person2/pose", 10);
	punPosePerson3 = nh.advertise<geometry_msgs::PoseStamped>("/person3/pose", 10);
    pubPointsPerson1 = nh.advertise<sensor_msgs::PointCloud2>("/cloudpoints_person1", 100);
    pubPointsPerson2 = nh.advertise<sensor_msgs::PointCloud2>("/cloudpoints_person2", 100);
    pubPointsPerson3 = nh.advertise<sensor_msgs::PointCloud2>("/cloudpoints_person3", 100);
    pubPointsPersons = nh.advertise<sensor_msgs::PointCloud2>("/cloudpoints_persons", 100);
	ros::spin();
    return 0;

}
