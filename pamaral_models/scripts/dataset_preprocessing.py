#!/usr/bin/env python3

import csv
import cv2
import mediapipe as mp
import numpy as np
import os
import rosbag

from cv_bridge import CvBridge


mp_hands = mp.solutions.hands
bridge = CvBridge()

folder_path = '/home/miglou/catkin_ws/src/MRSI_Thesis_Action_Anticipation/data/bag_files'

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Create the absolute path to the file
    bag_path = os.path.join(folder_path, filename)

    # Check if the file path is a file (not a directory)
    if os.path.isfile(bag_path) and bag_path.endswith(".bag"):
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        # Open the CSV file in append mode
        file_path = '/home/miglou/catkin_ws/src/MRSI_Thesis_Action_Anticipation/data/points/' + filename[:-4] + ".csv"
        file = open(file_path, 'a+', newline='')
        writer = csv.writer(file)

        # Open the bag file
        bag = rosbag.Bag(bag_path)

        # Iterate over the messages in the bag
        print(bag.get_message_count())

        """
        for topic, msg, t in bag.read_messages():
            image = bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe hand model
            results = hands.process(image_rgb)

            # If at least one hand was detected
            # if results.multi_hand_landmarks:
            #     # Publish the points of the right hand
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         points = []

            #         # Extract the x, y, z coordinates of each landmark
            #         for landmark in hand_landmarks.landmark:
            #             points.append([landmark.x, landmark.y, landmark.z])
                
            #         break

            #     points = np.array(points)

            #     centroid = np.average(points, axis=0)

            #     points[:,0] -= centroid[0]
            #     points[:,1] -= centroid[1]
            #     points[:,2] -= centroid[2]

            #     max_value = np.max(np.absolute(points))

            #     points = (points/max_value + 1)/2

                
            #     writer.writerow(points)

        """
        # Close the bag file
        bag.close()

        del hands
        del writer
        file.close()

        print("another one")