#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from pamaral_models.msg import PointList


class HandsDetectionMediapipe:

    def __init__(self, input_topic):
        self.input_topic = input_topic
        self.images = []

        # Initialize MediaPipe Hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.bridge = CvBridge()
        self.mp_drawing_publisher = rospy.Publisher("mp_drawing", Image, queue_size=1)
        self.mp_points_publisher = rospy.Publisher("mp_points", PointList, queue_size=300)
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        self.hands_centroids_sub = rospy.Subscriber("hands_centroids", PointList, self.hands_centroids_callback)
        rospy.loginfo("Hand Detection Ready")

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except Exception as e:
            rospy.logerr(e)
            return
        
        self.images.append(image)
    
    def hands_centroids_callback(self, msg):
        image = self.images[0]
        self.images = self.images[1:]

        points = []
        drawing = image.copy()

        if len(msg.points) > 0:

            temp = [[p.x, p.y, p.z] for p in msg.points]
            left_hand = np.array(temp[0])
            right_hand = np.array(temp[1])
            
            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe hand model
            results = self.hands.process(image_rgb)

            # If at least one hand was detected
            if results.multi_hand_landmarks:
                # Publish the points of the right hand
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract the x, y, z coordinates of each landmark
                    for landmark in hand_landmarks.landmark:
                        points.append([landmark.x, landmark.y, landmark.z])
                    
                    temp = np.array(points)

                    centroid = np.average(temp, axis=0)

                    right_hand_distance = np.linalg.norm(right_hand[:2] - centroid[:2])
                    left_hand_distance = np.linalg.norm(left_hand[:2] - centroid[:2])

                    if right_hand_distance < left_hand_distance:
                        # Draw right hand landmarks on the frame
                        self.mp_drawing.draw_landmarks(drawing, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        break

                    else:
                        points = []
        
        # publish centroids and mask
        msg = PointList()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.input_topic
        msg.points = [Point(p[0], p[1], p[2]) for p in points]
        self.mp_points_publisher.publish(msg)

        self.mp_drawing_publisher.publish(self.bridge.cv2_to_imgmsg(drawing, "bgr8"))


def main():
    default_node_name = 'hands_detection_mediapipe'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))

    HandsDetectionMediapipe(input_topic=input_topic)

    rospy.spin()


if __name__ == '__main__':
    main()
