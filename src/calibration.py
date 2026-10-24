"""
Step 1 - Camera calibration

Run once from the project folder:
    python src/calibration.py

Result:
    camera_cal/calib.npz                  camera matrix (mtx) + distortion (dist)
    output_images/calibration_corners.jpg detected corners on one image
    output_images/undistort_calibration.jpg   original | undistorted chessboard
    output_images/undistort_test1.jpg         original | undistorted road image
"""
import glob
from pathlib import Path

import cv2 as cv
import numpy as np

import config

# Termination criteria for the corner sub-pixel refinement
CRITERIA = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)


def calibrate(show=False):
    nx, ny = config.CHESSBOARD
    # Object points (0,0,0), (1,0,0) ... (8,5,0) - same for every image
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    obj_points, img_points = [], []
    corners_example = None

    paths = sorted(glob.glob(str(config.CAMERA_CAL_DIR / "calibration*.jpg")))
    for path in paths:
        img = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, (nx, ny), None)
        print(f"{'OK  ' if found else 'MISS'} {Path(path).name}")
        if not found:
            continue
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        obj_points.append(objp)
        img_points.append(corners)

        cv.drawChessboardCorners(img, (nx, ny), corners, found)
        if corners_example is None:
            corners_example = img
        if show:
            cv.imshow("chess board", img)
            cv.waitKey(200)
    if show:
        cv.destroyAllWindows()

    if not obj_points:
        raise SystemExit("No chessboard found - check camera_cal/ folder.")

    # Two images are 1281x721, so the size is set explicitly (not gray.shape)
    rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, config.CALIB_SIZE, None, None)

    # Mean re-projection error
    error = 0
    for i in range(len(obj_points)):
        projected, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error += cv.norm(img_points[i], projected, cv.NORM_L2) / len(projected)

    print(f"\nUsed {len(obj_points)}/{len(paths)} images")
    print(f"RMS error: {rms:.3f} px, mean error: {error / len(obj_points):.3f} px")
    print("Camera matrix:\n", np.round(mtx, 1))
    print("Distortion:", np.round(dist.ravel(), 4))

    np.savez(config.CALIB_FILE, mtx=mtx, dist=dist, size=np.array(config.CALIB_SIZE))
    print("Saved camera_cal/calib.npz")
    return mtx, dist, corners_example


def load_calibration():
    """Return (mtx, dist) for the 1280x720 camera."""
    if not config.CALIB_FILE.exists():
        raise SystemExit("camera_cal/calib.npz not found - run: python src/calibration.py")
    data = np.load(config.CALIB_FILE)
    return data["mtx"], data["dist"]


def scale_camera_matrix(mtx, image_size):
    """Camera matrix for a resized image (fx, cx scale with width; fy, cy with height).
    Distortion coefficients do not change with resizing."""
    sx = image_size[0] / config.CALIB_SIZE[0]
    sy = image_size[1] / config.CALIB_SIZE[1]
    m = mtx.copy()
    m[0, 0] *= sx
    m[0, 2] *= sx
    m[1, 1] *= sy
    m[1, 2] *= sy
    return m


def undistort(img, mtx, dist):
    """Undistort while keeping the same camera matrix.
    NOTE: not getOptimalNewCameraMatrix(alpha=1) - that changes the
    focal length and would move the perspective points."""
    return cv.undistort(img, mtx, dist, None, mtx)


if __name__ == "__main__":
    mtx, dist, corners_img = calibrate()
    config.OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)
    cv.imwrite(str(config.OUTPUT_IMAGES_DIR / "calibration_corners.jpg"), corners_img)

    for src_path, out_name in [
        (config.CAMERA_CAL_DIR / "calibration1.jpg", "undistort_calibration.jpg"),
        (config.TEST_IMAGES_DIR / "test1.jpg", "undistort_test1.jpg"),
    ]:
        img = cv.imread(str(src_path))
        if img is None:
            continue
        img = cv.resize(img, config.CALIB_SIZE)
        both = np.hstack((img, undistort(img, mtx, dist)))
        cv.imwrite(str(config.OUTPUT_IMAGES_DIR / out_name), both)
    print("Example images saved to output_images/")
