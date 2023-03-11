#!/usr/bin/env python3

from cv_bridge import CvBridge
import cv2
import json
import numpy as np
import rospy
from sensor_msgs.msg import Image


class Perception_Block:
    """This node should receive raw data and apply relevant initial processing."""

    def __init__(self) -> None:
        self.path = "/home/miglou/catkin_ws/src/MRSI_Thesis/pamaral_perception_block/config/"

        f = open(self.path + "red.json")
        red = json.load(f)
        f.close()

        f = open(self.path + "green.json")
        green = json.load(f)
        f.close()

        self.red_mins = np.array([red['limits']['h']['min'], red['limits']['s']['min'], red['limits']['v']['min']])
        self.red_maxs = np.array([red['limits']['h']['max'], red['limits']['s']['max'], red['limits']['v']['max']])
        self.green_mins = np.array([green['limits']['h']['min'], green['limits']['s']['min'], green['limits']['v']['min']])
        self.green_maxs = np.array([green['limits']['h']['max'], green['limits']['s']['max'], green['limits']['v']['max']])

        self.bridge = CvBridge()
        self.red_mask_publisher = rospy.Publisher("/red_mask", Image, queue_size=1)
        self.green_mask_publisher = rospy.Publisher("/green_mask", Image, queue_size=1)
        self.cimage = None
        self.cimage_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.cimage_callback)
        self.dimage = None
        self.dimage_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, self.dimage_callback)
        
        self.showImage()
    

    def cimage_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.cimage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        except:
            print("Error reading color image")
        
        # process red and green mask
        self.red_mask = cv2.inRange(self.cimage, self.red_mins, self.red_maxs)
        self.green_mask = cv2.inRange(self.cimage, self.green_mins, self.green_maxs)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))

        self.red_mask = cv2.morphologyEx(self.red_mask, cv2.MORPH_CLOSE,kernel)
        self.green_mask = cv2.morphologyEx(self.green_mask, cv2.MORPH_CLOSE,kernel)

        self.red_mask_publisher.publish(self.bridge.cv2_to_imgmsg(self.red_mask, "8UC1"))
        self.green_mask_publisher.publish(self.bridge.cv2_to_imgmsg(self.green_mask, "8UC1"))


    def dimage_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.dimage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        except:
            print("Error reading depth image")
            return


    def showImage(self):
        while True:
            if self.cimage is not None and self.dimage is not None:
                cv2.imshow("Depth Image", cv2.cvtColor(self.dimage, cv2.COLOR_HSV2BGR))
                cv2.imshow("Color Image", cv2.cvtColor(self.cimage, cv2.COLOR_HSV2BGR))
                cv2.imshow("Red Mask", self.red_mask)
                cv2.imshow("Green Mask", self.green_mask)

                key = cv2.waitKey(100)

                if key == ord('q'):  # q for quit
                    print('You pressed q ... aborting')
                    break


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'perception_block'
    rospy.init_node(default_node_name, anonymous=False)

    perception_block = Perception_Block()

    rospy.spin()


if __name__ == '__main__':
    main()
