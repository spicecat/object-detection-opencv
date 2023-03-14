import cv2
from Yolov3 import Yolov3

img = Yolov3(get_img=lambda: cv2.imread('images/dog.jpg'))

cv2.imshow("Result", img.prediction())
cv2.waitKey(0)
cv2.destroyAllWindows()
