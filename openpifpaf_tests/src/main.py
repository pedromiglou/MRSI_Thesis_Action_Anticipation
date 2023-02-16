#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Node:
    def __init__(self) -> 'Node':
        self.subscriber = rospy.Subscriber('/cimages', Image, self.callbackImageReceived)
        self.window_name = 'Output'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.bridge = CvBridge()
        self.cv_image = None


    def callbackImageReceived(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
    

    def showImage(self):
        while True: 
            if self.cv_image is not None:
                cv2.imshow(self.window_name, self.cv_image)
                
                key = cv2.waitKey(100)

                if key == ord('q'):  # q for quit
                    print('You pressed q ... aborting')
                    break


def main():
    rospy.init_node('openpifpaf', anonymous=False)
    node = Node()
    node.showImage()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()