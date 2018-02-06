import numpy as np
import cv2



def get_detections(image):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")


    (h, w) = image.shape[:2]

    # [Shape of image is accessed by img.shape. It returns a tuple of number
    # of rows, columns and channels (if image is color):

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),  # convert frame into a blob
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)  # me blob eka nural network ekata deela eke detections gannawa
    detections = net.forward()

    return detections