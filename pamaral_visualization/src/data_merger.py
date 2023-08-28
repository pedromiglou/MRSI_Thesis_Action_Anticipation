#!/usr/bin/env python3

import cv2
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image

from pamaral_perception_system.msg import CentroidList


class DataMerger:
    """This node should receive data relevant for visualization and process it to show in rviz."""

    def __init__(self):
        self.colors = {"red": (0, 0, 255), "dark_blue": (139, 0, 0), "light_blue": (232, 219, 164),
                       "green": (0, 255, 0), "yellow": (0, 255, 255), "orange": (0, 165, 255),
                       "violet": (182, 38, 155), "white": (255, 255, 255)}

        self.log_image_publisher = rospy.Publisher("/log_image", Image, queue_size=1)

        self.centroids = {"red": [], "dark_blue": [], "light_blue": [], "green": [],
                          "yellow": [], "orange": [], "violet": [], "white": []}
        self.centroids_subscriber = rospy.Subscriber("/table_centroids", CentroidList, self.centroids_callback)

        self.user_pose = None
        self.user_pose_subscriber = rospy.Subscriber("/user_pose", PointStamped, self.user_pose_callback)

        self.bridge = CvBridge()
        self.cimage = None
        self.cimage_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.cimage_callback)

    def centroids_callback(self, msg):
        self.centroids[msg.color] = msg.points

    def user_pose_callback(self, msg):
        self.user_pose = msg.point

    def cimage_callback(self, msg):
        try:
            self.cimage = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if self.cimage is not None:
                cimage = self.cimage

                for k, v in self.colors.items():
                    centroids = self.centroids[k]
                    for c in centroids:
                        cv2.putText(cimage, '+', (int(c.x), int(c.y)), cv2.FONT_ITALIC, 1, v, 2, cv2.LINE_8)

                if self.user_pose is not None:
                    cv2.putText(cimage, '+', (int(self.user_pose.x), int(self.user_pose.y)), cv2.FONT_ITALIC, 1,
                                (0, 0, 0), 2, cv2.LINE_8)

                self.log_image_publisher.publish(self.bridge.cv2_to_imgmsg(cimage, "bgr8"))

        except:
            print("Error reading color image")


def main():
    default_node_name = 'data_merger'
    rospy.init_node(default_node_name, anonymous=False)

    DataMerger()

    rospy.spin()


if __name__ == '__main__':
    main()
