#!/usr/bin/env python3

import os
import rospy
import subprocess

from datetime import datetime

def main():
    default_node_name = 'bag_recorder'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_topic'))
    output_folder = rospy.get_param(rospy.search_param('output_folder'))

    # Create a new folder if needed
    os.makedirs(output_folder, exist_ok=True)

    file_path = os.path.join(output_folder, f'{input_topic.replace("/","_")}_{datetime.now().strftime("%d_%m_%Y_%H:%M:%S")}.bag')
    rosbag_cmd = ['rosbag', 'record', '-O', file_path, input_topic]
    rosbag_process = subprocess.Popen(rosbag_cmd)
    rospy.on_shutdown(rosbag_process.terminate)

    rospy.spin()


if __name__ == '__main__':
    main()
