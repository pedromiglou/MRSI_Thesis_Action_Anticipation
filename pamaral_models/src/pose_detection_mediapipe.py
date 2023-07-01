#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from pamaral_models.msg import PointList


class PoseDetectionMediapipe:

    def __init__(self, input_topic):
        self.input_topic = input_topic

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.bridge = CvBridge()
        self.hands_centroids_publisher = rospy.Publisher("hands_centroids", PointList, queue_size=300)
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        rospy.loginfo("Pose Detection Ready")

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except Exception as e:
            rospy.logerr(e)
            return
        
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe pose model
        results = self.pose.process(image_rgb)

        points = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                points.append([landmark.x, landmark.y, landmark.z])
        
            left_hand = np.array([points[15], points[17], points[19], points[21]])
            right_hand = np.array([points[16], points[18], points[20], points[22]])
            
            points = [np.average(left_hand, axis=0), np.average(right_hand, axis=0)]
        
        # publish centroids and mask
        msg = PointList()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.input_topic
        msg.points = [Point(p[0], p[1], p[2]) for p in points]
        self.hands_centroids_publisher.publish(msg)


def main():
    default_node_name = 'pose_detection_mediapipe'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))

    PoseDetectionMediapipe(input_topic=input_topic)

    rospy.spin()


if __name__ == '__main__':
    main()
