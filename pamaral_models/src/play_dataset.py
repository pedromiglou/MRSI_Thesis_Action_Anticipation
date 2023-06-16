#!/usr/bin/env python3

import os
import rospy
import subprocess
import time

from datetime import datetime
from std_srvs.srv import Empty
from std_msgs.msg import String

# Initialize the ROS node
rospy.init_node('service_client_node')

pub = rospy.Publisher('csv_filename', String, queue_size=10)

folder_path = '/home/miglou/catkin_ws/src/MRSI_Thesis_Action_Anticipation/data/bag_files'

input()

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Create the absolute path to the file
    file_path = os.path.join(folder_path, filename)

    # Check if the file path is a file (not a directory)
    if os.path.isfile(file_path):
        message = String()
        message.data = filename[:-4]

        # Publish the message
        pub.publish(message)

        rosbag_cmd = ['rosbag', 'play', file_path]
        rosbag_process = subprocess.Popen(rosbag_cmd)

        rosbag_process.wait()
