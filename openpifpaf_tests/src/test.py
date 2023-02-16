import numpy as np
import cv2
import io
import PIL
import requests
import torch
import openpifpaf
import matplotlib as plt

image = cv2.imread("image.png")

#pil_im = PIL.Image.open("image.png")
#im = np.asarray(pil_im)

#with openpifpaf.show.image_canvas(im) as ax:
#    pass

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
predictions, gt_anns, image_meta = predictor.numpy_image(image)

#annotation_painter = openpifpaf.show.AnnotationPainter()
#with openpifpaf.show.image_canvas(im) as ax:
#    annotation_painter.annotations(ax, predictions)

print(predictions[0].data)

for predict in predictions[0].data:
    image = cv2.circle(image, (int(predict[0]),int(predict[1])), 10, (255, 0, 0), 2)


cv2.imshow("A",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.imshow(ax)
#print(type(ax))