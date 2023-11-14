#!/usr/bin/env python3

import os
import rosbag
import rospy

from datetime import datetime
from sensor_msgs.msg import Image


class VideoBagRecorder:
    def __init__(self, input_topic, output_folder):
        self.input_topic = input_topic

        # Create a rosbag to store the images
        file_path = os.path.join(output_folder, f'{input_topic.replace("/","_")}_{datetime.now().strftime("%d_%m_%Y_%H:%M:%S")}.bag')
        self.bag = rosbag.Bag(file_path, 'w')

        # Subscribe to the image topic
        rospy.Subscriber(input_topic, Image, self.image_callback)

    def image_callback(self, msg):
        # Save the image to the bag file
        self.bag.write(self.input_topic, msg, msg.header.stamp)


def main():
    default_node_name = 'bag_recorder'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_topic'))
    output_folder = rospy.get_param(rospy.search_param('output_folder'))

    # Create a new folder if needed
    os.makedirs(output_folder, exist_ok=True)

    VideoBagRecorder(input_topic=input_topic, output_folder=output_folder)

    rospy.spin()


if __name__ == '__main__':
    main()
