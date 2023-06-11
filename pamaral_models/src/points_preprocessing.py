#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from pamaral_models.msg import PointList


class PointsPreprocessing:

    def __init__(self):
        self.bridge = CvBridge()
        self.mp_points_image_publisher = rospy.Publisher("mp_points_image", Image, queue_size=1)
        self.preprocessed_points_publisher = rospy.Publisher("preprocessed_points", PointList, queue_size=1)
        self.mp_points_sub = rospy.Subscriber("mp_points", PointList, self.mp_points_callback)

    def mp_points_callback(self, msg):
        points = [[p.x, p.y, p.z] for p in msg.points]
        points = np.array(points)

        centroid = np.average(points, axis=0)

        points[:,0] -= centroid[0]
        points[:,1] -= centroid[1]
        points[:,2] -= centroid[2]

        max_value = np.max(np.absolute(points))

        points = (points/max_value + 1)/2

        mp_points_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw the points on the image
        point_radius = 5
        point_color = 255  # White

        for point in points:
            cv2.circle(mp_points_image, [int(point[0]*640), int(point[1]*480)], point_radius, point_color, -1)
        
        self.mp_points_image_publisher.publish(self.bridge.cv2_to_imgmsg(mp_points_image, "bgr8"))

        msg = PointList()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "mp_points"
        msg.points = [Point(p[0], p[1], p[2]) for p in points]
        self.preprocessed_points_publisher.publish(msg)


def main():
    default_node_name = 'points_preprocessing'
    rospy.init_node(default_node_name, anonymous=False)

    PointsPreprocessing()

    rospy.spin()


if __name__ == '__main__':
    main()
