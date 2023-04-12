# orange: img[144:182,380:406,:]
# yellow: img[329:371,240:266,:]
# white: img[331:370,281:307,:]
# green: img[327:367,324:349,:]
# violet: img[337:376,367:380,:]

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("frame0006.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

a = 1

if not a:
    cv2.imshow("a", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

else:
    img = img[144:182,380:406,:]

    img0 = img[:,:,0]

    plt.hist(img0.ravel(),256,[0,256])

    plt.show()

    img1 = img[:,:,1]

    plt.hist(img1.ravel(),256,[0,256])

    plt.show()

    img2 = img[:,:,2]

    plt.hist(img2.ravel(),256,[0,256])

    plt.show()
