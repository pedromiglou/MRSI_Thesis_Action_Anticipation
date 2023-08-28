#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Define the callback function for each image topic
def image_callback1(msg):
    save_image(msg, "image1")

def image_callback2(msg):
    save_image(msg, "image2")

def image_callback3(msg):
    save_image(msg, "image3")

# Save the received image with a consecutively increased ID for each topic
def save_image(msg, basename):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Increment the ID for the corresponding topic and create the filename
        if basename == "image1":
            save_image.id1 += 1
            filename = f"{basename}_{save_image.id1}.png"
        elif basename == "image2":
            save_image.id2 += 1
            filename = f"{basename}_{save_image.id2}.png"
        elif basename == "image3":
            save_image.id3 += 1
            filename = f"{basename}_{save_image.id3}.png"
        else:
            rospy.logerr("Invalid basename provided.")

        cv2.imwrite(filename, cv_image)
        rospy.loginfo("Image saved as %s", filename)
    except Exception as e:
        rospy.logerr("Error saving image: %s", str(e))

if __name__ == '__main__':
    rospy.init_node('image_saver_node')

    # Initialize the ID counters for each topic
    save_image.id1 = 0
    save_image.id2 = 0
    save_image.id3 = 0

    # Subscribe to the image topics
    rospy.Subscriber('/front_camera/color/image_raw', Image, image_callback1)
    rospy.Subscriber('/mp_drawing', Image, image_callback2)
    rospy.Subscriber('/mp_points_image', Image, image_callback3)

    rospy.spin()
