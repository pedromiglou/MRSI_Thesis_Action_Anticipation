"""
rotate around point 0 with angle between point 0 and 9
normalize using mins/maxs
"""

import csv
import os
import cv2
import mediapipe as mp
import numpy as np


# Set up MediaPipe hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Path to the folder containing the images
folder_path = '/home/miglou/catkin_ws/src/MRSI_Thesis_Action_Anticipation/data/images_dataset'

for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)  # Get the complete path of the item

    if os.path.isdir(item_path):
        # Get sorted list of image filenames
        image_files = sorted(os.listdir(item_path))

        # Initialize MediaPipe hand model
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.05)

        data = []

        # Iterate through images in the folder
        for filename in image_files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Load image
                image_path = os.path.join(item_path, filename)
                image = cv2.imread(image_path)
                
                # Convert BGR image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process image with MediaPipe hand model
                results = hands.process(image_rgb)
                
                # Check if hands were detected in the image
                test_image = np.zeros((480, 640))
                if results.multi_hand_landmarks:
                    first = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        points = []
                        if first:
                            # Iterate through the hand landmarks (key points)
                            for landmark in hand_landmarks.landmark:
                                # Extract the x, y, z coordinates of each landmark
                                point = [landmark.x, landmark.y, landmark.z]
                                # Do something with the coordinates
                                points.append(point)

                            # Draw hand landmarks on the image
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            points = np.array(points)

                            centroid = np.average(points, axis=0)

                            points[:,0] -= centroid[0]
                            points[:,1] -= centroid[1]
                            points[:,2] -= centroid[2]

                            max_value = np.max(np.absolute(points))

                            points = (points/max_value + 1)/2

                            # Draw the points on the image
                            point_radius = 3
                            point_color = 255  # White

                            for point in points:
                                cv2.circle(test_image, [int(point[0]*640), int(point[1]*480)], point_radius, point_color, -1)

                            data.append(points)

                            first = False
                        
                # Display the image with hand landmarks
                #cv2.imshow('Hand Landmarks', image)
                #cv2.imshow('Rotated Landmarks', test_image)
                #cv2.waitKey(0)

        # Clean up
        hands.close()
        #cv2.destroyAllWindows()

        # Path to the output CSV file
        csv_file = f'{item_path[:79]}/{item_path[79:]}.csv'

        # Open the CSV file in write mode
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Iterate through the list of lists and write each row to the CSV file
            for row in data:
                writer.writerow(row)