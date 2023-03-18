#!/usr/bin/env python3

from cv_bridge import CvBridge
import cv2
import rospy
from sensor_msgs.msg import Image


class Perception_Block:

    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.cimage = None
        self.cimage_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.cimage_callback)
        self.dimage = None
        self.dimage_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, self.dimage_callback)
        
        self.showImages()
    

    def cimage_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.cimage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        except:
            print("Error reading color image")


    def dimage_callback(self, msg):
        try:
            self.dimage = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        except:
            print("Error reading depth image")
            return


    def showImages(self):
        while True:
            if self.cimage is not None and self.dimage is not None:
                cv2.imshow("Depth Image", self.dimage)
                cv2.imshow("Color Image", cv2.cvtColor(self.cimage, cv2.COLOR_HSV2BGR))

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
