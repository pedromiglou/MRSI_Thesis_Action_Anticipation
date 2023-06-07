import os
import cv2
import mediapipe as mp

# Set up MediaPipe hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Path to the folder containing the images
folder_path = '/home/miglou/catkin_ws/src/MRSI_Thesis_Action_Anticipation/data/images_dataset/05_06_2023_15:52:33'

# Get sorted list of image filenames
image_files = sorted(os.listdir(folder_path))

# Initialize MediaPipe hand model
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

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
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                first = True
                points = []
                if first:
                    # Iterate through the hand landmarks (key points)
                    for landmark in hand_landmarks.landmark:
                        # Extract the x, y, z coordinates of each landmark
                        x, y, z = landmark.x, landmark.y, landmark.z
                        # Do something with the coordinates
                        points.append(x)
                        points.append(y)
                        points.append(z)

                    first = False

                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data.append(points)
                
        # Display the image with hand landmarks
        #cv2.imshow('Hand Landmarks', image)
        #cv2.waitKey(0)

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