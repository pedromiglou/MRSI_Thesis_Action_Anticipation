import copy
import numpy as np
import cv2

def find_hands(input_image, mp, x_lim, y_lim):
    """
    Finds the bounding boxes and images of the left and right hands in the input image.
    
    Args:
    input_image (numpy array): The input image as a numpy array.
    mp (dict): A dictionary containing information for the MediaPipe library.
    x_lim (int): The half-length of the bounding box in the x-axis direction.
    y_lim (int): The half-length of the bounding box in the y-axis direction.
    
    Returns:
    hand_left_bounding_box (list): A list of integers representing the bounding box of the left hand in the format [x_min, y_min, x_max, y_max].
    hand_right_bounding_box (list): A list of integers representing the bounding box of the right hand in the format [x_min, y_min, x_max, y_max].
    hand_right (numpy array): A numpy array representing the image of the right hand.
    hand_left (numpy array): A numpy array representing the image of the left hand.
    annotated_image (numpy array): A numpy array representing the annotated input image with bounding boxes drawn around the hands.
    pose (MediaPipe module): The MediaPipe Pose module.
    """

    pose = mp["pose"]
    mp_drawing = mp["mp_drawing"]
    mp_pose = mp["mp_pose"]
    mp_drawing_styles = mp["mp_drawing_styles"]
    left_hand_points = mp["left_hand_points"]
    right_hand_points = mp["right_hand_points"]

    hand_left_bounding_box = [0, 0, 0, 0]
    hand_right_bounding_box = [0, 0, 0, 0]

    h, w, _ = input_image.shape
    image = input_image

    # Process the image using the MediaPipe Pose module.
    results = pose.process(image)

    x_left_points = []
    x_right_points = []
    y_left_points = []
    y_right_points = []

    h, w, _ = image.shape

    hand_left = None
    hand_right = None
    hand_left_valid = True
    hand_right_valid = True
    color_r = (255, 0, 0)
    color_l = (255, 0, 0)

    # If pose landmarks were detected, extract the x and y coordinates of the left and right hand landmarks.
    if results.pose_landmarks:
        for id_landmark, landmark in enumerate(results.pose_landmarks.landmark):
            if id_landmark in left_hand_points:
                x_left_points.append(landmark.x)
                y_left_points.append(landmark.y)

            if id_landmark in right_hand_points:
                x_right_points.append(landmark.x)
                y_right_points.append(landmark.y)

        # Calculate the center points of the left and right hands.
        l_c = [int(np.mean(x_left_points) * w), int(np.mean(y_left_points) * h)]
        r_c = [int(np.mean(x_right_points) * w), int(np.mean(y_right_points) * h)]

        # Adjust the center points if they are out of bounds.
        if l_c[0] < x_lim:
            l_c[0] = x_lim
        if l_c[1] < y_lim:
            l_c[1] = y_lim
        if r_c[0] < x_lim:
            r_c[0] = x_lim
        if r_c[1] < y_lim:
            r_c[1] = y_lim
        
        # Acquire images from the region of interest.
        hand_left_bounding_box = [l_c[0]-x_lim, l_c[1]-y_lim, l_c[0]+x_lim, l_c[1]+y_lim]
        hand_right_bounding_box = [r_c[0]-x_lim, r_c[1]-y_lim, r_c[0]+x_lim, r_c[1]+y_lim]
        
        hand_left = input_image[l_c[1]-y_lim:l_c[1]+y_lim, l_c[0]-x_lim:l_c[0]+x_lim]

        hand_right = input_image[r_c[1]-y_lim:r_c[1]+y_lim, r_c[0]-x_lim:r_c[0]+x_lim]

        left_start_point = (l_c[0]-x_lim, l_c[1]-y_lim)
        left_end_point = (l_c[0]+x_lim, l_c[1]+y_lim)

        right_start_point = (r_c[0]-x_lim, r_c[1]-y_lim)
        right_end_point = (r_c[0]+x_lim, r_c[1]+y_lim)

        # Check validity of the location of the hands
        

        threshold_points = [results.pose_landmarks.landmark[id_land].y for id_land in mp["threshold_points"]]
        weights = [1, 1, 0.2, 0.2]
        weights = np.array(weights) / sum(weights)

        y_threshold = 0
        for i, weight in enumerate(weights):
            y_threshold += weight * threshold_points[i]

        y_threshold = y_threshold * h

        if y_threshold < (l_c[1]): # Check if left hand above threshold
            hand_left_valid = False

        if y_threshold < (r_c[1]): # Check if right hand above threshold
            hand_right_valid = False

        for face_id in mp["face_points"]: # Check if face is in image
            x = results.pose_landmarks.landmark[face_id].x * w
            y = results.pose_landmarks.landmark[face_id].y * h

            # check if face point is inside left bounding box
            if l_c[0]-x_lim < x < l_c[0] + x_lim and l_c[1]-y_lim < y < l_c[1] + y_lim:  
                hand_left_valid = False

            # check if face point is inside right bounding box
            if r_c[0]-x_lim < x < r_c[0] + x_lim and r_c[1]-y_lim < y < r_c[1] + y_lim:  
                hand_right_valid = False


        # Check if hands are superimposed
        left_box = {"x": (l_c[0]-x_lim, l_c[0]+x_lim),
                    "y": (l_c[1]-y_lim, l_c[1]+y_lim)}
        right_box = {"x": (r_c[0]-x_lim, r_c[0]+x_lim),
                    "y": (r_c[1]-y_lim, r_c[1]+y_lim)}

        if isOverlapping2D(left_box, right_box):
            hand_right_valid = False
            hand_left_valid = False
        
        hand_right_valid = True
        hand_left_valid = True

        if not hand_left_valid:
            color_l = (0, 0, 255)

        if not hand_right_valid:
            color_r = (0, 0, 255)

    if np.array(hand_left).shape != (2*x_lim, 2*y_lim, 3):
        hand_left = None

    if np.array(hand_right).shape != (2*x_lim, 2*y_lim, 3):
        hand_right = None

    if not hand_left_valid:
        hand_left = None

    if not hand_right_valid:
        hand_right = None

    return hand_left_bounding_box, hand_right_bounding_box, hand_right, hand_left, pose

def isOverlapping2D(box1, box2):

    x_cond = box1["x"][1] >= box2["x"][0] and box2["x"][1] >= box1["x"][0]
    y_cond = box1["y"][1] >= box2["y"][0] and box2["y"][1] >= box1["y"][0]

    return x_cond and y_cond
