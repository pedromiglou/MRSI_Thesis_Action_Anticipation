#!/usr/bin/env python3

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
import argparse


class HumanLocater:
    """This node should receive depth images and detect the position of the human."""

    def __init__(self, debug) -> None:
        self.debug = debug

        self.pose_publisher = rospy.Publisher("/user_pose", PointStamped, queue_size=1)
        self.dimage_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, self.dimage_callback)

        if self.debug:
            self.bridge = CvBridge()
            self.pose = None
            self.dimage = None
            self.cimage = None
            self.cimage_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.cimage_callback)
            self.show_images()


    def dimage_callback(self, msg):
        try:
            dimage = self.bridge.imgmsg_to_cv2(msg, "passthrough")

            dimage = dimage.copy()

            dimage[dimage==0] = np.max(dimage)

            c = np.argmin(dimage)

            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "/camera/depth/image_raw"
            msg.point = Point(c%640,c/640,0)
            self.pose_publisher.publish(msg)
            
            if self.debug:
                self.dimage, self.pose = dimage, msg.point

        except:
            print("Error reading depth image")
            return
    

    def cimage_callback(self, msg):
        try:
            self.cimage = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except:
            print("Error reading color image")


    def show_images(self):
        while True:
            if self.cimage is not None and self.dimage is not None:
                cv2.putText(self.cimage, '+', (int(self.pose.x), int(self.pose.y)), cv2.FONT_ITALIC, 1, (255,0,0), 2, cv2.LINE_8)
                cv2.imshow("Depth Image", self.dimage)
                cv2.imshow("Human Location", self.cimage)

                key = cv2.waitKey(100)
                if key == ord('q'):  # q for quit
                    print('You pressed q ... aborting')
                    break


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'human_location_detector'
    rospy.init_node(default_node_name, anonymous=False)

    parser = argparse.ArgumentParser(description="Arguments for human location detection")
    parser.add_argument("-d", "--debug", action='store_true',
                    help="if present, then the user position is shown in a window")

    args, _ = parser.parse_known_args()

    human_location = HumanLocater(debug = args.debug)

    rospy.spin()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
