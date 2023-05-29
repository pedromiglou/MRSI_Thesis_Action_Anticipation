#!/usr/bin/env python3

import cv2
import rospy
import mediapipe as mp
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


BBOX_SIZE = [50, 50]


class HandsCropperMediapipe:

    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        self.bridge = CvBridge()
        self.hand_image_publisher = rospy.Publisher("hand_image", Image, queue_size=1)
        self.mp_drawing_publisher = rospy.Publisher("mp_drawing", Image, queue_size=1)
        self.subscriber = rospy.Subscriber("hand_usb_cam/image_raw", Image, self.image_callback)

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except:
            print("Error reading color image")
            return

        # Process the image with Mediapipe Hands
        results = self.hands.process(img_rgb)

        # Draw hand landmarks on the frame
        img_copy = img.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img_copy, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                break

        self.mp_drawing_publisher.publish(self.bridge.cv2_to_imgmsg(img_copy))

        if results.multi_hand_landmarks:
            hand_landmarks = [l for _,l in enumerate(results.multi_hand_landmarks)][0].landmark
            points_x = [0 if l.x < 0 else 1 if l.x > 1 else l.x for l in hand_landmarks]
            points_y = [0 if l.y < 0 else 1 if l.y > 1 else l.y for l in hand_landmarks]

            h, w, _ = img.shape

            c = [int(np.mean(points_x) * w), int(np.mean(points_y) * h)]
            
            x_min = 0 if c[0]-BBOX_SIZE[0]<0 else c[0]-BBOX_SIZE[0]
            x_max = w if c[0]+BBOX_SIZE[0]>w else c[0]+BBOX_SIZE[0]
            y_min = 0 if c[1]-BBOX_SIZE[1]<0 else c[1]-BBOX_SIZE[1]
            y_max = h if c[1]+BBOX_SIZE[1]>h else c[1]+BBOX_SIZE[1]

            hand_image = img[y_min:y_max, x_min:x_max]

            self.hand_image_publisher.publish(self.bridge.cv2_to_imgmsg(hand_image))

        """
        mp_data = {}
        mp_data["face_points"] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        mp_data["threshold_points"] = (11, 12, 24, 25) # shoulders and hips
        mp_data["left_hand_points"] = (16, 18, 20, 22)
        mp_data["right_hand_points"] = (15, 17, 19, 21)
        mp_data["mp_drawing"] = mp.solutions.drawing_utils
        mp_data["mp_drawing_styles"] = mp.solutions.drawing_styles
        mp_data["mp_pose"] = mp.solutions.pose
        mp_data["pose"] = mp_data["mp_pose"].Pose(static_image_mode=False,
                                                model_complexity=2,
                                                enable_segmentation=False,
                                                min_detection_confidence=0.7)
        
        left_bounding, right_bounding, hand_right, hand_left, mp_data["pose"] = find_hands(
                img, mp_data, x_lim=15, y_lim=15)

        if hand_right is not None:
            img = cv2.cvtColor(hand_right, cv2.COLOR_RGB2BGR)
            self.hand_image_publisher.publish(self.bridge.cv2_to_imgmsg(hand_right))
        """


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'hands_cropper_mediapipe'
    rospy.init_node(default_node_name, anonymous=False)

    HandsCropperMediapipe()

    rospy.spin()


if __name__ == '__main__':
    main()
