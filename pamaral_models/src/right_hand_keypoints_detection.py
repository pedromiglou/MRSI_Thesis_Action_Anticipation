#!/usr/bin/env python3

import actionlib
import numpy as np
import rospy

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from pamaral_models.msg import PointList
from pamaral_models.msg import HandsModelAction, PoseModelAction, HandsModelGoal, PoseModelGoal

class RightHandKeypointsDetection:

    def __init__(self, input_topic):
        self.input_topic = input_topic
        self.images = []

        # Initialize Action Clients for MediaPipe Nodes
        self.hands_model_client = actionlib.SimpleActionClient('hands_model', HandsModelAction)
        self.pose_model_client = actionlib.SimpleActionClient('pose_model', PoseModelAction)
        self.hands_model_client.wait_for_server()
        self.pose_model_client.wait_for_server()

        self.right_hand_keypoints_publisher = rospy.Publisher("right_hand_keypoints", PointList, queue_size=300)
        self.hands_keypoints_publisher = rospy.Publisher("hands_keypoints", PointList, queue_size=300)
        self.pose_keypoints_publisher = rospy.Publisher("pose_keypoints", PointList, queue_size=300)
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)

    def image_callback(self, msg):
        # Send image to Mediapipe nodes
        self.hands_model_client.send_goal(HandsModelGoal(image=msg))
        self.pose_model_client.send_goal(PoseModelGoal(image=msg))

        # Wait for results
        self.hands_model_client.wait_for_result()
        self.pose_model_client.wait_for_result()

        # Get results
        hands_keypoints = self.hands_model_client.get_result().points
        pose_keypoints = self.pose_model_client.get_result().points

        self.hands_keypoints_publisher.publish(points = hands_keypoints)
        self.pose_keypoints_publisher.publish(points = pose_keypoints)

        points = []
        if len(hands_keypoints) > 0 and len(pose_keypoints) > 0:
            pose_keypoints = [[p.x, p.y, p.z] for p in pose_keypoints]
            hands_keypoints = [[p.x, p.y, p.z] for p in hands_keypoints]

            # obtain right and left hands centroids and valid radius
            left_hand = np.array([pose_keypoints[15], pose_keypoints[17], pose_keypoints[19], pose_keypoints[21]])
            right_hand = np.array([pose_keypoints[16], pose_keypoints[18], pose_keypoints[20], pose_keypoints[22]])

            left_hand_centroid = np.average(left_hand, axis=0)
            right_hand_centroid = np.average(right_hand, axis=0)

            left_hand_radius = 2*np.max(np.linalg.norm(left_hand_centroid - left_hand, axis=1))
            right_hand_radius = 2*np.max(np.linalg.norm(right_hand_centroid - right_hand, axis=1))

            # separate hands keypoints
            hands_keypoints = [hands_keypoints[i*21:(i+1)*21] for i in range(len(hands_keypoints)//21)]

            # check which hand is closer to the centroid of the right hand and if it is within the valid radius
            best_distance = 100000
            for hand in hands_keypoints:
                hand_centroid = np.average(np.array(hand), axis=0)

                right_hand_distance = np.linalg.norm(hand_centroid[:2] - right_hand_centroid[:2])
                left_hand_distance = np.linalg.norm(hand_centroid[:2] - left_hand_centroid[:2])

                if right_hand_distance < left_hand_distance:# and right_hand_distance < right_hand_radius:
                    if right_hand_distance < best_distance:
                        points = [Point(p[0], p[1], p[2]) for p in hand]
                        best_distance = right_hand_distance
        
        # publish the keypoints of the hand that is closer to the centroid of the right hand
        msg = PointList()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.input_topic
        msg.points = points
        self.right_hand_keypoints_publisher.publish(msg)


def main():
    default_node_name = 'right_hand_keypoints_detection'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))

    RightHandKeypointsDetection(input_topic=input_topic)

    rospy.spin()


if __name__ == '__main__':
    main()
