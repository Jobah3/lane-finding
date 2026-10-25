"""
Steps 4 and 5 - Find lane pixels, fit polynomials, curvature and offset.

  * histogram of the lower half of the warped binary -> where the lines start
  * sliding windows from bottom to top -> pixels of each line
  * np.polyfit 2nd order: x = A*y^2 + B*y + C   (x as a function of y, because
    the lines are almost vertical in the bird's-eye view)
  * radius of curvature  R = (1 + (2Ay + B)^2)^1.5 / |2A|   in meters
  * vehicle offset = image centre - lane centre               in meters
"""
import cv2 as cv
import numpy as np


def histogram_bases(binary_warped, lane_width_px):
    #x positions where the left and right line start (bottom of the image).
    h, w = binary_warped.shape
    histogram = np.sum(binary_warped[h // 2:, :], axis=0).astype(np.float64)
    # Smooth so a dashed line or a double yellow line gives one clear peak
    histogram = np.convolve(histogram, np.ones(25) / 25, mode="same")
    mid = w // 2
    # Only look roughly where lines can be (ignores barriers / next lanes)
    lo = int(max(0, mid - 0.85 * lane_width_px))
    hi = int(min(w, mid + 0.85 * lane_width_px))
    left_base = lo + int(np.argmax(histogram[lo:mid]))
    right_base = mid + int(np.argmax(histogram[mid:hi]))
    return left_base, right_base, histogram


def sliding_windows(binary_warped, left_base, right_base,
                    nwindows=9, margin=100, minpix=50):
    """Collect line pixels with windows that move from bottom to top and
    re-centre on the mean x of the pixels they found."""
    h, w = binary_warped.shape
    out = np.dstack((binary_warped,) * 3) * 255
    window_h = h // nwindows
    nonzeroy, nonzerox = binary_warped.nonzero()

    centres = [left_base, right_base]
    steps = [0, 0]                       # last movement, used when a window is empty
    found = [[], []]
    for win in range(nwindows):
        y_low = h - (win + 1) * window_h
        y_high = h - win * window_h
        for side in (0, 1):
            x_low = centres[side] - margin
            x_high = centres[side] + margin
            cv.rectangle(out, (int(x_low), y_low), (int(x_high), y_high), (0, 255, 0), 2)
            good = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                    (nonzerox >= x_low) & (nonzerox < x_high)).nonzero()[0]
            found[side].append(good)
            if len(good) > minpix:
                new_centre = int(np.mean(nonzerox[good]))
                steps[side] = new_centre - centres[side]
                centres[side] = new_centre
            else:
                centres[side] += steps[side]     # keep following the curve

    left_idx = np.concatenate(found[0])
    right_idx = np.concatenate(found[1])
    return (nonzerox[left_idx], nonzeroy[left_idx],
            nonzerox[right_idx], nonzeroy[right_idx], out)


def search_around_poly(binary_warped, left_fit, right_fit, margin=80):
    """Video: search only near last frame's polynomials (faster, more stable)."""
    nonzeroy, nonzerox = binary_warped.nonzero()
    left_x = np.polyval(left_fit, nonzeroy)
    right_x = np.polyval(right_fit, nonzeroy)
    left_idx = np.abs(nonzerox - left_x) < margin
    right_idx = np.abs(nonzerox - right_x) < margin
    out = np.dstack((binary_warped,) * 3) * 255
    return (nonzerox[left_idx], nonzeroy[left_idx],
            nonzerox[right_idx], nonzeroy[right_idx], out)


def fit_poly(x, y, min_points=300):
    """2nd order polynomial x = A*y^2 + B*y + C, or None if too few pixels."""
    if len(x) < min_points or np.ptp(y) < 60:
        return None
    return np.polyfit(y, x, 2)


def curvature_m(fit, y_eval_px, xm_per_pix, ym_per_pix):
    """Radius of curvature in meters at y_eval (usually the bottom).
    Convert the pixel polynomial to meters:
        x_m = xm * x_px,  y_m = ym * y_px
        A_m = A * xm / ym^2,  B_m = B * xm / ym"""
    a = fit[0] * xm_per_pix / ym_per_pix ** 2
    b = fit[1] * xm_per_pix / ym_per_pix
    y = y_eval_px * ym_per_pix
    return (1 + (2 * a * y + b) ** 2) ** 1.5 / max(abs(2 * a), 1e-9)


def vehicle_offset_m(left_fit, right_fit, y_eval_px, camera_x_px, xm_per_pix):
    """Positive = vehicle is right of the lane centre.
    camera_x_px is where the image centre lands in the warped image (the camera
    is assumed to be mounted in the middle of the car)."""
    left_x = np.polyval(left_fit, y_eval_px)
    right_x = np.polyval(right_fit, y_eval_px)
    lane_centre = (left_x + right_x) / 2
    return (camera_x_px - lane_centre) * xm_per_pix


def draw_fit(out, fit, colour, thickness=4):
    h = out.shape[0]
    y = np.arange(h)
    x = np.polyval(fit, y)
    pts = np.int32(np.column_stack((x, y)))
    cv.polylines(out, [pts], False, colour, thickness)
