import cv2 as cv
import cv2.aruco as aruco
import numpy as np

ARUCO_DICT = {
    "DICT_7x7_50" : cv.aruco.DICT_7X7_50
}

def aruco_display(corners, ids, rejected, image):
    if(len(corners) > 0):

        ids = ids.flatten()

        for(markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4,2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            cv.putText(image, str(markerID), (topLeft[0], topLeft[1] - 1), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            print("[Inference] Aruco marker ID: {}".format(markerID))

    return image

aruco_type = "DICT_7x7_50"
arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, img = cap.read()
    h, w, _ = img.shape
    width = 1000
    height = int(width*(h/w))
    img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
    image_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)


    corners, ids, rejected = detector.detectMarkers(img)

    detected_markers = aruco_display(corners, ids, rejected, img)

    cv.imshow("Image", detected_markers)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv.destroyAllWindows()
cap.release()
