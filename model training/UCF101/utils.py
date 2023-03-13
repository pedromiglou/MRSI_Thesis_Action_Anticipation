import numpy as np
import cv2


def dataset_loader(path):
    cap = cv2.VideoCapture(path)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return None
    
    video = []
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            return cv2.resize(frame, (160, 120))
            video.append(cv2.resize(frame, (160, 120)))
            
        # Break the loop
        else:
            break
 
    # When everything done, release the video capture object
    cap.release()

    return np.stack(video, axis=0)
