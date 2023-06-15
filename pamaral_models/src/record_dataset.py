#!/usr/bin/env python3

import rospy
import subprocess
import time

from std_srvs.srv import Empty

# Initialize the ROS node
rospy.init_node('service_client_node')

# Wait for the service to become available
rospy.wait_for_service('/front_camera/start_capture')
rospy.wait_for_service('/front_camera/stop_capture')

rosbag_cmd = ['rosbag', 'record', 'front_camera/color/image_raw']
rosbag_process = subprocess.Popen(rosbag_cmd)

for i in range(3,0,-1):
    print(f"Starting in {i}")

    time.sleep(1)

service_proxy = rospy.ServiceProxy('/front_camera/start_capture', Empty)

service_proxy()

for i in range(30,0,-1):
    print(f"Stopping in {i}")

    time.sleep(1)

service_proxy = rospy.ServiceProxy('/front_camera/stop_capture', Empty)

rosbag_process.terminate()

service_proxy()
