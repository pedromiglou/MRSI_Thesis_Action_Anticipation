#!/usr/bin/env python3

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point


class Perception_Block:
    """This node should receive raw data and apply relevant initial processing."""

    def __init__(self) -> None:
        self.pose_publisher = rospy.Publisher("/user_pose", PointStamped, queue_size=1)
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
                cimage, dimage = self.cimage.copy(), self.dimage.copy()

                max_d = np.max(dimage)

                dimage[dimage==0] = max_d

                c = np.argmin(dimage)

                if c is not None:
                    msg = PointStamped()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "/camera/depth/image_raw"
                    msg.point = Point(c%640,c/640,0)
                    self.pose_publisher.publish(msg)
                    
                    print(f'Coordinates:{c/640},{c%640}')

                cv2.imshow("Depth Image", dimage)
                cv2.imshow("Color Image", cv2.cvtColor(cimage, cv2.COLOR_HSV2BGR))

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
