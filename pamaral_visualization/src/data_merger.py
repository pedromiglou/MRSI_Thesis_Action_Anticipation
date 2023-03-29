#!/usr/bin/env python3

import cv2
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from pamaral_color_image_processing.msg import PointListStamped
from sensor_msgs.msg import Image


class DataMerger:
    """This node should receive data relevant for visualization and process it to show in rviz."""

    def __init__(self):
        self.bridge = CvBridge()
        self.cimage = None
        self.cimage_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.cimage_callback)

        self.red_centroids = None
        self.red_centroids_subscriber = rospy.Subscriber("/red_centroids", PointListStamped, self.red_centroids_callback)

        self.green_centroids = None
        self.green_centroids_subscriber = rospy.Subscriber("/green_centroids", PointListStamped, self.green_centroids_callback)

        self.user_pose = None
        self.user_pose_subscriber = rospy.Subscriber("/user_pose", PointStamped, self.user_pose_callback)

        self.log_image_publisher = rospy.Publisher("/log_image", Image, queue_size=1)
        
        self.publish()
    

    def cimage_callback(self, msg):
        try:
            self.cimage = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except:
            print("Error reading color image")
    

    def red_centroids_callback(self, msg):
        self.red_centroids = msg.points
    

    def green_centroids_callback(self, msg):
        self.green_centroids = msg.points
    

    def user_pose_callback(self, msg):
        self.user_pose = msg.point


    def publish(self):
        while True:
            if self.cimage is not None:
                cimage = self.cimage
                
                if self.red_centroids is not None:
                    red_centroids = self.red_centroids
                    for c in red_centroids:
                        cv2.putText(cimage, '+', (int(c.x), int(c.y)), cv2.FONT_ITALIC, 1, (0,0,255), 2, cv2.LINE_8)
                
                if self.green_centroids is not None:
                    green_centroids = self.green_centroids
                    for c in green_centroids:
                        cv2.putText(cimage, '+', (int(c.x), int(c.y)), cv2.FONT_ITALIC, 1, (0,255,0), 2, cv2.LINE_8)
                
                if self.user_pose is not None:
                    cv2.putText(cimage, '+', (int(self.user_pose.x), int(self.user_pose.y)), cv2.FONT_ITALIC, 1, (255,0,0), 2, cv2.LINE_8)

                self.log_image_publisher.publish(self.bridge.cv2_to_imgmsg(cimage, "bgr8"))


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'data_merger'
    rospy.init_node(default_node_name, anonymous=False)

    data_merger = DataMerger()

    rospy.spin()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
