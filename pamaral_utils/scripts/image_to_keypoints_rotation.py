"""
rotate around point 0 with angle between point 0 and 9
normalize using mins/maxs
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(x1, y1, x2, y2):
    # Calculate differences in x and y coordinates
    dx = x2 - x1
    dy = y2 - y1

    # Check for special cases
    if dx == 0 and dy == 0:
        # Points are the same, no defined angle
        return None
    elif dx == 0:
        # Vertical line, angle is either π/2 or 3π/2
        angle = math.pi/2 if dy > 0 else 3*math.pi/2
    else:
        # Compute the angle using arctan and adjust based on quadrant
        angle = math.atan(dy / dx)
        if dx < 0:
            angle += math.pi
        elif dy < 0:
            angle += 2*math.pi

    # Convert angle to degrees
    angle_degrees = math.degrees(angle)

    return angle_degrees


# Set up MediaPipe hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Path to the folder containing the images
folder_path = '/home/miglou/catkin_ws/src/MRSI_Thesis_Action_Anticipation/data/images_dataset/05_06_2023_15:52:33'

# Get sorted list of image filenames
image_files = sorted(os.listdir(folder_path))

# Initialize MediaPipe hand model
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1)

data = []

# Iterate through images in the folder
for filename in image_files:
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe hand model
        results = hands.process(image_rgb)
        
        # Check if hands were detected in the image
        test_image = np.zeros((480, 640))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                first = True
                points = []
                if first:
                    # Iterate through the hand landmarks (key points)
                    for landmark in hand_landmarks.landmark:
                        # Extract the x, y, z coordinates of each landmark
                        point = [landmark.x, landmark.y, landmark.z]
                        # Do something with the coordinates
                        points.append(point)

                    first = False

                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                points = np.array(points)

                theta = - (calculate_angle(points[0,0], points[0,1], np.average(points[1:,0]), np.average(points[1:,1])))
                theta = 0

                points[:,0] -= points[0,0]
                points[:,1] -= points[0,1]
                points[:,2] -= points[0,2]

                # Construct the rotation matrix
                rotation_matrix = np.array([
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]
                ])
                
                # Apply the rotation to each point
                rotated_points = np.dot(points, rotation_matrix)

                rotated_points += 0.5

                # Draw the points on the image
                point_radius = 3
                point_color = 255  # White

                for point in rotated_points:
                    cv2.circle(test_image, [int(point[0]*480), int(point[1]*640)], point_radius, point_color, -1)

                data.append(points)
                
        # Display the image with hand landmarks
        cv2.imshow('Hand Landmarks', image)
        cv2.imshow('Rotated Landmarks', test_image)
        cv2.waitKey(0)

# Clean up
hands.close()
cv2.destroyAllWindows()


import csv

# Path to the output CSV file
csv_file = f'{folder_path[:79]}/{folder_path[79:]}.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Iterate through the list of lists and write each row to the CSV file
    for row in data:
        writer.writerow(row)