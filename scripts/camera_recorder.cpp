#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include "camera_recorder/RecordService.h"

bool is_recording = false; // Flag to track recording state
std::string save_path = "/home/sam/Desktop/camera_recordings/";

// ROS service callback function
bool recordCallback(camera_recorder::RecordService::Request &req,
                    camera_recorder::RecordService::Response &res) {
    if (req.record.data) is_recording = true;
    else is_recording = false;
    res.success.data = true;
    return true;
}

int main(int argc, char** argv) {
    // Initialize ROS node
    ros::init(argc, argv, "camera_recorder");
    ros::NodeHandle nh;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    // ROS service server for recording
    ros::ServiceServer service = nh.advertiseService("record_service", recordCallback);

    // Main loop
    while (ros::ok()) {
        // Wait for frames
        rs2::frameset frames = pipe.wait_for_frames();

        // Get the color frame
        rs2::video_frame color_frame = frames.get_color_frame();
        // Get the depth frame
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        // Get timestamp
        auto timestamp = frames.get_timestamp();

        // Convert color frame to OpenCV Mat object
        cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(color_image, color_image, cv::COLOR_RGB2BGR); // Convert from RGB to BGR

        // Convert depth frame to OpenCV Mat object
        cv::Mat depth_image(cv::Size(depth_frame.get_width(), depth_frame.get_height()), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        // Save color and depth frames if recording
        if (is_recording) {
            std::string color_filename = save_path + "color_frame_" + std::to_string(timestamp) + ".png";
            std::string depth_filename = save_path + "depth_frame_" + std::to_string(timestamp) + ".png";
            cv::imwrite(color_filename, color_image);
            cv::imwrite(depth_filename, depth_image);
        }

        // Display images
        cv::imshow("Color Image", color_image);
        cv::imshow("Depth Image", depth_image);

        // Exit the loop if the user presses the 'q' key
        if (cv::waitKey(1) == 'q') {
            break;
        }

        ros::spinOnce(); // Handle ROS callbacks
    }

    return 0;
}
