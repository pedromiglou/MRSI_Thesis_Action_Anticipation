#!/usr/bin/env python3

import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image


class HumanLocater:
    """This node should receive depth images and detect the position of the human."""

    def __init__(self):
        self.bridge = CvBridge()
        self.pose_publisher = rospy.Publisher("/user_pose", PointStamped, queue_size=1)
        self.dimage_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, self.dimage_callback)


    def dimage_callback(self, msg):
        try:
            dimage = self.bridge.imgmsg_to_cv2(msg, "passthrough")

            dimage = dimage.copy()[:,350:]

            dimage[dimage==0] = np.max(dimage)

            c = np.argmin(dimage)

            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "/camera/depth/image_raw"
            msg.point = Point(c%290+350,c/290,0)
            self.pose_publisher.publish(msg)

        except:
            print("Error reading depth image")
            return


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'human_locater'
    rospy.init_node(default_node_name, anonymous=False)

    human_locater = HumanLocater()

    rospy.spin()


if __name__ == '__main__':
    main()
