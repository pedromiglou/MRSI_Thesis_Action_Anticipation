# Importing required libraries
from skimage.segmentation import slic
from skimage.color import label2rgb
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("frame0004.jpg")

img = img[17:474, 193:454]

# Setting the plot size as 15, 15
plt.figure(figsize=(15, 15))

# Applying Simple Linear Iterative
# Clustering on the image
# - 50 segments & compactness = 10
astronaut_segments = slic(img,
                          n_segments=400,
                          compactness=10)
plt.subplot(1, 2, 1)

# Plotting the original image
plt.imshow(img)
plt.subplot(1, 2, 2)

# Converts a label image into
# an RGB color image for visualizing
# the labeled regions.
cv2.imshow("A", label2rgb(astronaut_segments,
                     img,
                     kind='avg'))

cv2.waitKey(0)

cv2.destroyAllWindows()