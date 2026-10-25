"""
Step 3 - Perspective transform to a bird's-eye view.
Source and destination points are defined in config.py for each camera profile.
The transform maps the road trapezoid to a rectangular top-down view.
"""
import cv2 as cv
import numpy as np


class Perspective:
    def __init__(self, src, dst):
        self.src = np.float32(src)
        self.dst = np.float32(dst)
        self.M = cv.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img):
        h, w = img.shape[:2]
        return cv.warpPerspective(img, self.M, (w, h), flags=cv.INTER_LINEAR)

    def unwarp(self, img):
        h, w = img.shape[:2]
        return cv.warpPerspective(img, self.Minv, (w, h), flags=cv.INTER_LINEAR)

    def warp_point(self, x, y):
        """Where does an image point land in the bird's-eye view?"""
        p = cv.perspectiveTransform(np.float32([[[x, y]]]), self.M)
        return float(p[0, 0, 0]), float(p[0, 0, 1])

    def draw_check(self, img):
        """Original with the src trapezoid | warped with the dst rectangle.
        On a straight road both lane lines must be vertical in the right image."""
        a = img.copy()
        b = self.warp(img)
        cv.polylines(a, [self.src.astype(np.int32)], True, (0, 0, 255), 2)
        cv.polylines(b, [self.dst.astype(np.int32)], True, (0, 0, 255), 2)
        return np.hstack((a, b))
