import cv2
from Yolov3 import Yolov3

cap = cv2.VideoCapture(0)
img = Yolov3(get_img=lambda: cap.read()[1])
while True:
    cv2.imshow("Result", img.prediction())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
