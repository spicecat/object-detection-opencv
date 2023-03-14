import cv2
import numpy as np


class Yolov3:
    def __init__(self, scale=0.00392, classes="yolov3.txt", config="yolov3.cfg", weights="yolov3.weights", get_img=None):
        self.scale = scale
        self.classes = [line.strip()
                        for line in open(classes, 'r').readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = cv2.dnn.readNet(weights, config)
        self.get_img = get_img
        self.height, self.width = get_img().shape[:2]

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1]
                         for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label+':'+str(confidence)[:4], (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def prediction(self):
        img = self.get_img()
        blob = cv2.dnn.blobFromImage(
            img, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            x, y, w, h = boxes[i]
            self.draw_prediction(img, class_ids[i], confidences[i], round(
                x), round(y), round(x+w), round(y+h))

        return img
