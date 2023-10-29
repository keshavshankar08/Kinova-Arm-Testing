import cv2 as cv
import os
import pyrealsense2 as rs
import numpy as np

Chess_Board_Dimensions = (9, 6)

n = 0  # image counter

# checks images dir is exist or not
image_path = "CameraTesting/CameraCalibration/images"

Dir_Check = os.path.isdir(image_path)

if not Dir_Check:  # if directory does not exist, a new one is created
    os.makedirs(image_path)
    print(f'"{image_path}" Directory is created')
else:
    print(f'"{image_path}" Directory already exists.')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret

# Intel RealSense Vars
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

    copyFrame = color_image
    gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(
        color_image, gray, criteria, Chess_Board_Dimensions
    )
    # print(ret)
    cv.putText(
        color_image,
        f"saved_img : {n}",
        (30, 40),
        cv.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.imshow("frame", color_image)
    # copyframe; without augmentation
    cv.imshow("copyFrame", copyFrame)

    key = cv.waitKey(1)

    if key == ord("q"):
        break
    if key == ord("s") and board_detected == True:
        # the checker board image gets stored
        cv.imwrite(f"{image_path}/image{n}.png", copyFrame)

        print(f"saved image number {n}")
        n += 1  # the image counter: incrementing
cv.destroyAllWindows()
pipe.stop()
print("Total saved Images:", n)