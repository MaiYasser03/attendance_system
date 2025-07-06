import cv2
import numpy as np
import logging
from utils.tts import speak

def calibrate_camera(cap):
    chessboard_size = (7, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    images_captured = 0
    max_images = 10

    speak("Starting camera calibration. Please show a chessboard pattern.")
    logging.info("Starting camera calibration")

    while images_captured < max_images:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame during calibration")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            images_captured += 1
            cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)

        try:
            cv2.imshow("Camera Calibration", frame)
        except cv2.error as e:
            logging.error(f"cv2.imshow failed during calibration: {str(e)}")
            cv2.imwrite(f"calibration_frame_{images_captured}.jpg", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    try:
        cv2.destroyWindow("Camera Calibration")
    except cv2.error:
        logging.warning("Could not destroy Camera Calibration window")

    cap.release()
    logging.info("Camera released after calibration")

    if images_captured > 0:
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        logging.info("Calibration completed")
        speak("Camera calibration completed")
        return mtx, dist
    else:
        logging.warning("Calibration failed, no images")
        speak("Calibration failed")
        return None, None
