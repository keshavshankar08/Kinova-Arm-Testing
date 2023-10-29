import cv2
import cv2.aruco as aruco
import numpy as np

def generate_aruco_marker(marker_id, marker_size, output_file):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = aruco.generateImageMarker(dictionary, marker_id, marker_size, marker_image, 1)

    cv2.imwrite(output_file, marker_image)

if __name__ == "__main__":
    for i in range(50):
        marker_id = i
        marker_size = 50
        output_file = "markers/marker" + str(i) + ".png"
        generate_aruco_marker(marker_id, marker_size, output_file)
