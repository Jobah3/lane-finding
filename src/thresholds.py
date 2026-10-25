"""
Step 2 - Binary image of the lane lines (colour + gradient thresholds).

Idea: a lane marking is a NARROW stripe that is BRIGHTER than the asphalt
next to it.  The top-hat filter keeps exactly such stripes and removes wide
bright areas (concrete patches, sky, dirt) and step edges (shadow borders,
tar seams).  Every mask below is 0/1 (uint8).
"""
import cv2 as cv
import numpy as np

RIDGE_KERNEL = 41            # wider than a lane line, in pixels (1280 px image)


def scale_to_255(img):
    img = np.absolute(img)
    m = img.max()
    if m == 0:
        return np.zeros(img.shape, np.uint8)
    return np.uint8(255 * img / m)


def in_range(channel, low, high):
    return ((channel >= low) & (channel <= high)).astype(np.uint8)


def ridge(gray, kernel_width):
    """Top-hat with a horizontal kernel: how much brighter a pixel is than
    the road around it (left/right)."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    return cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)


def sobel_x_mask(gray, low=30, high=255, ksize=3):
    """Strong horizontal brightness change (= near-vertical edges).
    We take the absolute value and scale by the maximum,
    so the thresholds work for bright and dark frames."""
    sx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=ksize)
    return in_range(scale_to_255(sx), low, high)


def yellow_mask(hsv, lab):
    """Yellow paint: hue 17-35 (OpenCV hue is 0-179), saturated, and a high
    B channel in LAB (128 = neutral, yellow > 145).  Brown dirt has hue < 17."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b = lab[:, :, 2]
    return ((h >= 17) & (h <= 35) & (s >= 75) & (v >= 60) & (b >= 142)).astype(np.uint8)


def white_mask(hls, ridge_l):
    """White paint: bright, not coloured, and brighter than the road next to it.
    The brightness limit adapts to the frame (bright concrete vs dark asphalt)."""
    l, s = hls[:, :, 1], hls[:, :, 2]
    road = l[l.shape[0] // 2:, :]                       # lower half = road
    limit = max(170, np.percentile(road, 96))
    return ((l >= limit) & (s < 120) & (ridge_l > 20)).astype(np.uint8)


def combined_binary(img_bgr, return_parts=False):
    k = max(15, int(RIDGE_KERNEL * img_bgr.shape[1] / 1280)) | 1
    blur = cv.GaussianBlur(img_bgr, (5, 5), 0)
    hls = cv.cvtColor(blur, cv.COLOR_BGR2HLS)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(blur, cv.COLOR_BGR2LAB)
    l = hls[:, :, 1]
    ridge_l = ridge(l, k)

    yellow = yellow_mask(hsv, lab)
    white = white_mask(hls, ridge_l)
    # Gradient: Sobel edges that belong to a narrow bright stripe
    stripe = cv.dilate(in_range(ridge_l, 25, 255), np.ones((1, 5), np.uint8))
    edges = sobel_x_mask(l, 30, 255) & stripe

    combined = yellow | white | edges
    if return_parts:
        return combined, {"yellow": yellow, "white": white, "edges": edges}
    return combined


def colour_debug(parts):
    """Partial masks as one colour image (BGR): yellow paint = yellow,
    white paint = blue, Sobel edges = green."""
    y, w, e = parts["yellow"], parts["white"], parts["edges"]
    return np.dstack((w * 255, (y | e) * 255, y * 255))
