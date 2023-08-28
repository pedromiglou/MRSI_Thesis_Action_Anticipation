#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set to empty string to force CPU usage

import actionlib
import cv2
import mediapipe as mp
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point

from pamaral_models.msg import PoseModelAction, PoseModelResult


class PoseModelMediapipe:

    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.bridge = CvBridge()

        self.server = actionlib.SimpleActionServer('pose_model', PoseModelAction, self.execute, False)
        self.server.start()

        rospy.loginfo("Pose Detection Ready")

    def execute(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")

        except Exception as e:
            rospy.logerr(e)
            return
        
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe pose model
        results = self.pose.process(image_rgb)

        points = []

        # If the pose was detected, extract the coordinates of each landmark
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                points.append(Point(landmark.x, landmark.y, landmark.z))
        
        # check if preempted
        if self.server.is_preempt_requested():
            self.server.set_preempted()
            return
        
        # return landmarks
        res = PoseModelResult(points=points)
        self.server.set_succeeded(res)


def main():
    default_node_name = 'pose_model_mediapipe'
    rospy.init_node(default_node_name, anonymous=False)

    PoseModelMediapipe()

    rospy.spin()


if __name__ == '__main__':
    main()
