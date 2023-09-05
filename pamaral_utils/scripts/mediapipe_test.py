import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        # Read frames from the video capture
        ret, frame = cap.read()
        if not ret:
            print("No frame captured. Exiting...")
            break

        # Convert the frame to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe Hands
        results = hands.process(image)

        # results attributes:
        # ['count', 'index', 'multi_hand_landmarks', 'multi_hand_world_landmarks', 'multi_handedness']

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                print(len(handedness.classification))
                if handedness.classification[0].label != 'Left':
                    mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()