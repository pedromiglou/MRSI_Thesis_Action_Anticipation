#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Node:
    def __init__(self) -> 'Node':
        self.publisher = rospy.Publisher('/cimages', Image, queue_size=0)
        self.capture = cv2.VideoCapture(0)
        bridge = CvBridge()
        window_name = 'camera'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            _, image = self.capture.read()  # get an image from the camera

            if image is None:
                print('Video is over, terminating.')
                break  # video is over
                
            image = cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2)))

            image_message = bridge.cv2_to_imgmsg(image, "rgb8")

            self.publisher.publish(image_message)

            cv2.imshow(window_name, image)
            key = cv2.waitKey(20)

            if key == ord('q'):  # q for quit
                print('You pressed q ... aborting')
                break


def main():
    rospy.init_node('camera', anonymous=False)
    node = Node()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------
    node.capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()