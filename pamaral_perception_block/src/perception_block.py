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

        # invalid = np.max(self.dimage)
        # print(invalid)

        # highest_point = 0

        # c = None

        # for i in range(0, 480):
        #     for j in range(180, 640):
        #         if invalid > self.dimage[i, j] > highest_point:
        #             c = (i,j)
        #             highest_point = self.dimage[i, j]
        
        # if c is not None:
        #     msg = PointStamped()
        #     msg.header.stamp = rospy.Time.now()
        #     msg.header.frame_id = "/camera/depth/image_raw"
        #     msg.point = Point(c[1],c[0],0)
        #     self.pose_publisher.publish(msg)
            
        #     #cv2.putText(self.cimage, '+', (c[1], c[0]), cv2.FONT_ITALIC, 1, (0,0,255), 2, cv2.LINE_8)


        #     print(f'Coordinates:{c[1]},{c[0]}')


    def showImages(self):
        while True:
            if self.cimage is not None and self.dimage is not None:
                highest_point = 10000000000

                #invalid = np.argmin(self.dimage)

                c = None

                for i in range(0, 480):
                    for j in range(360, 640):
                        if self.dimage[i, j] != 0 and self.dimage[i, j] < highest_point:
                            c = (i,j)
                            highest_point = self.dimage[i, j]

                if c is not None:
                    msg = PointStamped()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "/camera/depth/image_raw"
                    msg.point = Point(c[1],c[0],0)
                    self.pose_publisher.publish(msg)
                    
                    #cv2.putText(self.cimage, '+', (c[1], c[0]), cv2.FONT_ITALIC, 1, (0,0,255), 2, cv2.LINE_8)

                    print(f'Coordinates:{c[1]},{c[0]}')

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
