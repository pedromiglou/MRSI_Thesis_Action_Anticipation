#!/usr/bin/env python3

import rospy
import subprocess
import time

from datetime import datetime
from std_srvs.srv import Empty

# Initialize the ROS node
rospy.init_node('service_client_node')

# Wait for the service to become available
rospy.wait_for_service('/front_camera/start_capture')
rospy.wait_for_service('/front_camera/stop_capture')

output_folder = rospy.get_param(rospy.search_param('output_folder')) + "/mp_metrics"

# people = ["pedro", "joel", "manuel"]
# for i, p in enumerate(people):
#     print(f"{i} - {p}")

# person = int(input("?"))

status = ["visible", "partially visible", "obstructed"]
for i, p in enumerate(status):
    print(f"{i} - {p}")

s = int(input("?"))

objects = ["cube", "bottle", "phone", "screwdriver"]
for i, o in enumerate(objects):
    print(f"{i} - {o}")

object = int(input("?"))

t = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

filename = f"{output_folder}/{objects[object]}_{status[s]}_{t}.bag"

rosbag_cmd = ['rosbag', 'record', '-j', '-O', filename, 'front_camera/color/image_raw']
rosbag_process = subprocess.Popen(rosbag_cmd)

for i in range(2,0,-1):
    print(f"Starting in {i}")

    time.sleep(1)

service_proxy = rospy.ServiceProxy('/front_camera/start_capture', Empty)

service_proxy()

for i in range(25,0,-1):
    print(f"Stopping in {i}")

    time.sleep(1)

service_proxy = rospy.ServiceProxy('/front_camera/stop_capture', Empty)

rosbag_process.terminate()

service_proxy()
