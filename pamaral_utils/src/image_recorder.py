#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageRecorder:
    def __init__(self, input_topic, output_folder):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        self.output_folder = output_folder
        self.counter = 0

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        # Save the image to a file
        file_path = f"{self.output_folder}/{str(self.counter).zfill(5)}.png"
        self.counter += 1
        cv2.imwrite(file_path, cv_image)


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'image_recorder'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))
    output_folder = rospy.get_param(rospy.search_param('output_image_folder'))

    # Create a new folder
    output_folder += input("What should the name of the new folder be? ")
    os.makedirs(output_folder)

    ImageRecorder(input_topic=input_topic, output_folder=output_folder)

    rospy.spin()


if __name__ == '__main__':
    main()
