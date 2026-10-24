"""
Project settings: folders, chessboard size, camera profiles and perspective points.

The task data comes from two different cameras:

* "main"  - 1280x720 camera that has calibration images in camera_cal/.
            Videos: project_video03, challenge01-03.
            Images: test1-6, straight_lines1-2, and challange00101/111/136
            (these three are challenge01 frames scaled down to 960x540).
* "p1"    - 960x540 camera WITHOUT calibration images.
            Videos: project_video01, project_video02.
            Images: solid*, whiteCarLaneSwitch.
* "c2"    - 1280x720 camera of challenge02 (mounted lower, wider view,
            horizon at y~455), also without calibration images.

Everything that depends on the image size (src/dst points, camera matrix,
meters per pixel) is defined for the base size and scaled automatically.
"""
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent.parent
CAMERA_CAL_DIR = ROOT / "camera_cal"
CALIB_FILE = CAMERA_CAL_DIR / "calib.npz"
TEST_IMAGES_DIR = ROOT / "test_images"
TEST_VIDEOS_DIR = ROOT / "test_videos"
OUTPUT_IMAGES_DIR = ROOT / "output_images"
OUTPUT_VIDEOS_DIR = ROOT / "output_videos"

# ------------------------------------------------------------ calibration
CHESSBOARD = (9, 6)          # inner corners per row, per column
CALIB_SIZE = (1280, 720)     # size of the calibration images (w, h)

# --------------------------------------------------------- camera profiles
# src: trapezoid on the road (straight lane), order: top-left, bottom-left,
#      bottom-right, top-right.  dst: where those points go in the bird's-eye
#      image (a rectangle -> lane lines become vertical and parallel).
# lane_width_px: lane width in the warped image (= dst right x - dst left x).
# ym_per_pix: meters per pixel along the road in the warped image, estimated
#      from the dashed lane markings (one dash + gap = 12 m on US highways).
PROFILES = {
    "main": {
        "base_size": (1280, 720),
        "calibrated": True,
        "src": np.float32([[585, 455], [203, 720], [1127, 720], [695, 455]]),
        "dst": np.float32([[320, 0], [320, 720], [960, 720], [960, 0]]),
        "lane_width_px": 640,
        "ym_per_pix": 30 / 720,
    },
    "p1": {
        "base_size": (960, 540),
        "calibrated": False,
        "src": np.float32([[418, 345], [130, 540], [830, 540], [542, 345]]),
        "dst": np.float32([[240, 0], [240, 540], [720, 540], [720, 0]]),
        "lane_width_px": 480,
        "ym_per_pix": 27 / 540,
    },
    "c2": {
        "base_size": (1280, 720),
        "calibrated": False,
        "src": np.float32([[591, 500], [315, 680], [996, 680], [728, 500]]),
        "dst": np.float32([[320, 0], [320, 720], [960, 720], [960, 0]]),
        "lane_width_px": 640,
        "ym_per_pix": 36 / 720,
    },
}

LANE_WIDTH_M = 3.7           # standard highway lane width (US / EU ~3.5-3.75 m)


def pick_profile(path, image_size):
    """Choose the camera profile from the file name and the image size."""
    name = Path(path).name.lower()
    if name.startswith("challenge02"):
        return "c2"
    if name.startswith(("project_video01", "project_video02", "solid", "whitecar")):
        return "p1"
    if image_size == (960, 540) and not name.startswith("challange"):
        return "p1"
    return "main"


class Profile:
    """A camera profile scaled to one concrete image size."""

    def __init__(self, name, image_size):
        p = PROFILES[name]
        bw, bh = p["base_size"]
        w, h = image_size
        sx, sy = w / bw, h / bh
        scale = np.float32([sx, sy])

        self.name = name
        self.size = image_size
        self.calibrated = p["calibrated"]
        self.src = p["src"] * scale
        self.dst = p["dst"] * scale
        self.xm_per_pix = LANE_WIDTH_M / (p["lane_width_px"] * sx)
        self.ym_per_pix = p["ym_per_pix"] / sy
        self.lane_width_px = p["lane_width_px"] * sx
        # scale used for sliding-window sizes etc. (1.0 for 1280x720)
        self.pix_scale = w / 1280
