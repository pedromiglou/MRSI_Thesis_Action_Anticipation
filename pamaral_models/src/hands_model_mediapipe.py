#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set to empty string to force CPU usage

import actionlib
import cv2
import mediapipe as mp
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pamaral_models.msg import HandsModelAction, HandsModelResult


class HandsModelMediapipe:

    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.bridge = CvBridge()

        self.mp_drawing_publisher = rospy.Publisher("mp_drawing", Image, queue_size=1)

        self.server = actionlib.SimpleActionServer('hands_model', HandsModelAction, self.execute, False)
        self.server.start()

        rospy.loginfo("Hand Model Ready")

    def execute(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")

        except Exception as e:
            rospy.logerr(e)
            return
        
        drawing = image.copy()

        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe hand model
        results = self.hands.process(image_rgb)

        points = []
        handednesses = []

        # If at least one hand was detected, extract the coordinates of each landmark
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                for landmark in hand_landmarks.landmark:
                    points.append(Point(landmark.x, landmark.y, landmark.z))
                
                if handedness.classification[0].label != 'Left':
                    handednesses.append(String('left'))
                else:
                    handednesses.append(String('right'))
                
                # Draw hand landmarks on the frame
                self.mp_drawing.draw_landmarks(drawing, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # check if preempted
        if self.server.is_preempt_requested():
            self.server.set_preempted()
            return
        
        # return landmarks
        res = HandsModelResult(points=points, handednesses=handednesses)
        self.server.set_succeeded(res)

        # Publish the frame with the hand landmarks
        self.mp_drawing_publisher.publish(self.bridge.cv2_to_imgmsg(drawing, "bgr8"))


def main():
    default_node_name = 'hands_model_mediapipe'
    rospy.init_node(default_node_name, anonymous=False)

    HandsModelMediapipe()

    rospy.spin()


if __name__ == '__main__':
    main()
