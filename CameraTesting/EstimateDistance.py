import cv2 as cv
from cv2 import aruco
import numpy as np
import pyrealsense2 as rs

ARUCO_DICT = {
    "DICT_6X6_50": cv.aruco.DICT_6X6_50
}
'''
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000
'''

# load in the calibration data
calib_data_path = "CameraTesting/CameraCalibration/calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]


MARKER_SIZE = 3.6  # centimeters (measure your printed marker size)
obj_points = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                               [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                               [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                               [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]])

aruco_type = "DICT_6x6_50"
arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
pipe.start(cfg)

while True:
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    gray_frame = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    
    marker_corners, marker_IDs, reject = detector.detectMarkers(color_image)
    if(len(marker_corners) > 0):
        for i in range(len(marker_IDs)):
            img_points = marker_corners[i][0]

    if marker_corners:
        rVec, tVec, _ = cv.solvePnP(obj_points, img_points, cam_mat, dist_coef)
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                color_image, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Calculating the distance
            distance = np.sqrt((tVec[i][0][2] ** 2) + (tVec[i][0][0] ** 2) + (tVec[i][0][1] ** 2))
            # Draw the pose of the marker
            point = cv.drawFrameAxes(color_image, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                color_image,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                color_image,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            # print(ids, "  ", corners)
    cv.imshow("frame", color_image)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
pipe.stop()
cv.destroyAllWindows()