import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *
import time
import shutil
import os

# Dictionary of all aruco marker sizes
ARUCO_DICT = {
    "DICT_6X6_50": cv.aruco.DICT_6X6_50
}
'''
"DICT_4X4_50": cv.aruco.DICT_4X4_50,
"DICT_4X4_100": cv.aruco.DICT_4X4_100,
"DICT_4X4_250": cv.aruco.DICT_4X4_250,
"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
"DICT_5X5_50": cv.aruco.DICT_5X5_50,
"DICT_5X5_100": cv.aruco.DICT_5X5_100,
"DICT_5X5_250": cv.aruco.DICT_5X5_250,
"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
"DICT_6X6_50": cv.aruco.DICT_6X6_50,
"DICT_6X6_100": cv.aruco.DICT_6X6_100,
"DICT_6X6_250": cv.aruco.DICT_6X6_250,
"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
"DICT_7X7_50": cv.aruco.DICT_7X7_50,
"DICT_7X7_100": cv.aruco.DICT_7X7_100,
"DICT_7X7_250": cv.aruco.DICT_7X7_250,
"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
'''

# Main detection method
def pose_detection(marker_detector, image, intrinsic_coefs, distortion_coefs):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners, ids, rejected_points = marker_detector.detectMarkers(gray_image)
    if(len(corners) > 0):
        ids = ids.flatten()
        for(marker_corner, marker_id) in zip(corners, ids):
            corners = marker_corner.reshape((4,2))
            (top_left, top_right, bottom_right, bottom_left) = corners
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            center_position_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_position_y = int((top_left[1] + bottom_right[1]) / 2.0)
            print("Position: (" + str(center_position_x) + "," + str(center_position_y) + ")")
            rotation_vector, translation_vector, marker_points = cv.aruco.estimatePoseSingleMarkers(marker_corner, 0.03, intrinsic_coefs, distortion_coefs)
            cv.drawFrameAxes(image, intrinsic_coefs, distortion_coefs, rotation_vector, translation_vector, 0.02)
    return image

# Returns current color image with aruco tags marked
def marker_detection(corners, ids, rejected, image):
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
            #print("[Inference] Aruco marker ID: {}".format(markerID))
    return image

# Creates and saves a plot of current depth image
def createDepthPlot(depth_image, pointNumber):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    y, x = np.mgrid[:depth_image.shape[0], :depth_image.shape[1]]
    z = depth_image
    ax.scatter(z, x, y, c=z, cmap='viridis')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=40, roll=0)
    fig.savefig("CameraTesting/DepthPlots/DepthPlot" + str(pointNumber) + ".png")

# Aruco variables
arucoType = "DICT_6X6_50"
arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[arucoType])
arucoParams = cv.aruco.DetectorParameters()
arucoDetector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

# load in the calibration data
calib_data_path = "CameraTesting/CameraCalibration/calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

intrinsic_camera = cam_mat
distortion = dist_coef

# RealSense variables
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
pipe.start(cfg)

# Timer variables
timerStart = timerEnd = None

# Keeps track of number of data points recorded
numDataPoints = 0

# Clear past plots
if(os.path.exists("CameraTesting/DepthPlots")):
    shutil.rmtree("CameraTesting/DepthPlots")
if(os.path.exists("CameraTesting/MarkerPlots")):
    shutil.rmtree("CameraTesting/MarkerPlots")
if(os.path.exists("CameraTesting/DepthColorPlots")):
    shutil.rmtree("CameraTesting/DepthColorPlots")
if(os.path.exists("CameraTesting/TimingData.txt")):
    os.remove("CameraTesting/TimingData.txt")
os.mkdir("CameraTesting/DepthPlots")
os.mkdir("CameraTesting/MarkerPlots")
os.mkdir("CameraTesting/DepthColorPlots")
file = open("CameraTesting/TimingData.txt","w")

while True:
    # Get depth and color frames from camera
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    # Convert frames to images
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image_flipped = cv.flip(color_image, 1)
    #depth_image = cv.flip(color_image, 1)
    #color_image = cv.flip(color_image, 1)

    # Convert color image to other useful images
    gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    bgr_image = cv.cvtColor(color_image, cv.COLOR_RGB2BGR)
    binary_image = cv.threshold(bgr_image, 127, 255, cv.THRESH_BINARY)[1]

    # Convert depth image to other useful images
    depth_image_colorized = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha = 0.5), cv.COLORMAP_JET)
    
    # Display default image
    cv.imshow("Camera Feed", color_image_flipped)
    #cv.imshow("Depth Image", depth_image)
    #cv.imshow("Depth Image Colorized", depth_image_colorized)
    #cv.imshow("Binary Image", binary_image)

    # Get any key press
    key = cv.waitKey(1)

    # Start trial
    if(key == ord('s')):
        # Gets start time
        timerStart = time.time()
        numDataPoints += 1

    # Trial started process
    if(timerStart is not None):
        # Run aruco detection
        corners, ids, rejected = arucoDetector.detectMarkers(gray_image)

        # Create image with bounding boxes for markers
        detected_markers = pose_detection(arucoDetector, color_image, intrinsic_camera, distortion)
        #detected_markers = detected_markers(corners, ids, rejected, detected_markers)
        # Display marker detection result
        cv.imshow("Marker Detection", detected_markers)

        # If a tag is detected
        if(ids is not None):
            timerEnd = time.time()
            print("Aruco tag detected in: " + str(round((timerEnd - timerStart), 3)) + " seconds.")
            file.write("Trial " + str(numDataPoints) + ": " + str(round((timerEnd - timerStart), 3)) + " seconds.\n")
            #createDepthPlot(depth_image, numDataPoints)
            cv.imwrite("CameraTesting/MarkerPlots/MarkerPlot" + str(numDataPoints) + ".png", detected_markers)
            cv.imwrite("CameraTesting/DepthColorPlots/DepthColorPlot" + str(numDataPoints) + ".png", depth_image_colorized)
            timerStart = timerEnd = None
    
    if(key == ord('q')):
        break

cv.destroyAllWindows()
pipe.stop()
