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
        self.preprocessed_points_publisher = rospy.Publisher("preprocessed_points", PointList, queue_size=300)
        self.right_hand_keypoints_sub = rospy.Subscriber("right_hand_keypoints", PointList, self.mp_points_callback)

    def mp_points_callback(self, msg):
        if len(msg.points)>0:
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
            point_radius = 8
            point_color = 255  # White

            lines = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                     [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
                     [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [13, 17]]
            
            # Define the polygon vertices
            polygon_points = np.array([points[0][:2], points[5][:2], points[9][:2], points[13][:2], points[17][:2]])

            polygon_points[:,0] *= 640
            polygon_points[:,1] *= 480

            polygon_points = polygon_points.astype(np.int32)

            # Reshape the points array into shape compatible with fillPoly
            polygon_points = polygon_points.reshape((-1, 1, 2))

            # Specify the color for the polygon (in BGR format)
            polygon_color = (0, 64, 0) # Dark green

            # Fill the polygon with the specified color
            cv2.fillPoly(mp_points_image, [polygon_points], polygon_color)
            
            for p1, p2 in lines:
                cv2.line(mp_points_image, [int(points[p1][0]*640), int(points[p1][1]*480)], [int(points[p2][0]*640), int(points[p2][1]*480)], [0, 255, 0], 2)

            for point in points:
                cv2.circle(mp_points_image, [int(point[0]*640), int(point[1]*480)], point_radius, [0,0,255], -1)
            
            self.mp_points_image_publisher.publish(self.bridge.cv2_to_imgmsg(mp_points_image, "bgr8"))
        
        else:
            points = []

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
