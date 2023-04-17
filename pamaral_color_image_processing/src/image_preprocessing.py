#!/usr/bin/env python3

import cv2
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ImagePreprocessing:
    """This node should receive color images and apply the required preprocessing for the next nodes."""

    def __init__(self):
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher("/preprocessed_image", Image, queue_size=1)
        self.subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
    

    def image_callback(self, msg):
        # read as bgr8
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except:
            print("Error reading color image")
            return
        
        # crop to ROI
        img = img[17:474, 193:454]

        # convert to hsv
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # publish cropped hsv image
        self.publisher.publish(self.bridge.cv2_to_imgmsg(img))


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'image_preprocessing'
    rospy.init_node(default_node_name, anonymous=False)

    image_preprocessing = ImagePreprocessing()

    rospy.spin()


if __name__ == '__main__':
    main()
