"""
The complete lane-finding pipeline for one image or one video frame:

  undistort -> binary (thresholds) -> bird's-eye warp -> find pixels
  -> polynomial fit -> sanity check + smoothing -> curvature / offset
  -> draw lane, warp back, write text

LaneFinder keeps state between video frames (previous fits), which allows
"search around previous polynomial", smoothing and recovering from bad frames.
For single images create a new LaneFinder(video=False) for every image.
"""
from collections import deque

import cv2 as cv
import numpy as np

import calibration
import config
import lane_detection as ld
import thresholds
from perspective import Perspective


class Line:
    """Remembers the last n good fits of one lane line.
    best = average of those fits (simple smoothing)."""

    def __init__(self, n):
        self.fits = deque(maxlen=n)

    def add(self, fit):
        self.fits.append(fit)

    def reset(self):
        self.fits.clear()

    @property
    def best(self):
        return np.mean(self.fits, axis=0) if self.fits else None


class LaneFinder:
    def __init__(self, image_size, profile_name="main", mtx=None, dist=None,
                 video=True, history=8, debug=False):
        self.size = image_size
        self.profile = config.Profile(profile_name, image_size)
        self.video = video
        self.debug = debug
        self.history = history

        self.mtx = self.dist = None
        if self.profile.calibrated and mtx is not None:
            self.mtx = calibration.scale_camera_matrix(mtx, image_size)
            self.dist = dist

        self.persp = Perspective(self.profile.src, self.profile.dst)
        w, h = image_size
        self.camera_x = self.persp.warp_point(w / 2, h)[0]
        self.left = Line(history)
        self.right = Line(history)
        self.bad_frames = 0
        self.stages = {}            # intermediate images (for run_images.py)
        self.status = ""            # what happened in the last frame
        self.last_radius = self.last_offset = float("nan")

    # ------------------------------------------------------------ helpers
    def _sane(self, lf, rf):
        """Do two fits look like one lane? Width plausible and roughly parallel."""
        if lf is None or rf is None:
            return False
        h = self.size[1]
        ys = np.array([h - 1, h * 0.5, 0])
        widths = np.polyval(rf, ys) - np.polyval(lf, ys)
        lw = self.profile.lane_width_px
        if np.any(widths < 0.65 * lw) or np.any(widths > 1.45 * lw):
            return False
        if np.ptp(widths) > 0.35 * lw:          # not parallel
            return False
        return True

    def _close_to_previous(self, lf, rf):
        """In video a line cannot jump far between two frames."""
        if self.left.best is None:
            return True
        h = self.size[1]
        lim = 0.25 * self.profile.lane_width_px
        for fit, best in ((lf, self.left.best), (rf, self.right.best)):
            for y in (h - 1, h / 2):
                if abs(np.polyval(fit, y) - np.polyval(best, y)) > lim:
                    return False
        return True

    def _detect(self, warped, use_previous):
        s = self.profile.pix_scale
        if use_previous:
            lx, ly, rx, ry, vis = ld.search_around_poly(
                warped, self.left.best, self.right.best, margin=int(80 * s))
        else:
            lb, rb, _ = ld.histogram_bases(warped, self.profile.lane_width_px)
            lx, ly, rx, ry, vis = ld.sliding_windows(
                warped, lb, rb, nwindows=9, margin=int(100 * s), minpix=int(50 * s))
        min_pts = int(200 * s * s)
        lf, rf = ld.fit_poly(lx, ly, min_pts), ld.fit_poly(rx, ry, min_pts)
        vis[ly, lx] = (0, 0, 255)                 # left pixels red
        vis[ry, rx] = (255, 0, 0)                 # right pixels blue
        return lf, rf, vis, (len(lx), len(rx))

    def _complete_single_line(self, lf, rf, counts):
        """Only one line found (e.g. single image, faded paint): build the other
        one by shifting it one lane width."""
        lw = self.profile.lane_width_px
        if lf is not None and rf is None:
            rf = lf.copy(); rf[2] += lw
        elif rf is not None and lf is None:
            lf = rf.copy(); lf[2] -= lw
        elif lf is not None and rf is not None and not self._sane(lf, rf):
            # keep the line with more pixels
            if counts[0] >= counts[1]:
                rf = lf.copy(); rf[2] += lw
            else:
                lf = rf.copy(); lf[2] -= lw
        return lf, rf

    # ------------------------------------------------------------- main
    def process(self, frame):
        und = frame if self.mtx is None else calibration.undistort(frame, self.mtx, self.dist)
        binary, parts = thresholds.combined_binary(und, return_parts=True)
        warped = self.persp.warp(binary)

        have_previous = self.video and self.left.best is not None
        lf, rf, vis, counts = self._detect(warped, use_previous=have_previous)
        ok = self._sane(lf, rf) and (not self.video or self._close_to_previous(lf, rf))
        if not ok and have_previous:
            # fall back to a full sliding-window search in the same frame
            lf, rf, vis, counts = self._detect(warped, use_previous=False)
            ok = self._sane(lf, rf) and self._close_to_previous(lf, rf)

        if ok:
            self.left.add(lf)
            self.right.add(rf)
            self.bad_frames = 0
            self.status = "ok"
        elif self.video and self.left.best is not None:
            self.bad_frames += 1                   # keep using the last good fits
            self.status = "kept previous"
            if self.bad_frames > self.history:
                self.left.reset(); self.right.reset()
                self.bad_frames = 0
                self.status = "reset"
        else:
            lf, rf = self._complete_single_line(lf, rf, counts)
            if lf is not None:
                self.left.add(lf); self.right.add(rf)
                self.status = "one line + lane width"
            else:
                self.status = "not found"

        left_fit, right_fit = self.left.best, self.right.best
        if left_fit is None:
            result = und.copy()
            cv.putText(result, "Lane not found", (30, 50), cv.FONT_HERSHEY_SIMPLEX,
                       1.2 * self.profile.pix_scale + 0.3, (0, 0, 255), 2)
        else:
            result = self._draw_lane(und, left_fit, right_fit)
            for fit, colour in ((left_fit, (0, 255, 255)), (right_fit, (0, 255, 255))):
                ld.draw_fit(vis, fit, colour, max(2, int(4 * self.profile.pix_scale)))
            self._write_text(result, left_fit, right_fit)

        self.stages = {
            "undistorted": und,
            "binary": thresholds.colour_debug(parts),
            "warped": self.persp.warp(und),
            "warped_binary": warped * 255,
            "detection": vis,
            "result": result,
        }
        if self.debug:
            result = self._debug_view(result, self.stages["binary"], vis)
        return result

    # ---------------------------------------------------------- drawing
    def _draw_lane(self, und, left_fit, right_fit):
        h, w = und.shape[:2]
        ploty = np.linspace(0, h - 1, h)
        left_x = np.polyval(left_fit, ploty)
        right_x = np.polyval(right_fit, ploty)

        lane = np.zeros_like(und)
        pts_left = np.column_stack((left_x, ploty))
        pts_right = np.flipud(np.column_stack((right_x, ploty)))
        cv.fillPoly(lane, [np.int32(np.vstack((pts_left, pts_right)))], (0, 255, 0))
        t = max(8, int(25 * self.profile.pix_scale))
        cv.polylines(lane, [np.int32(pts_left)], False, (0, 0, 255), t)
        cv.polylines(lane, [np.int32(pts_right)], False, (255, 0, 0), t)

        back = self.persp.unwarp(lane)            # Minv: bird's-eye -> camera view
        return cv.addWeighted(und, 1.0, back, 0.35, 0)

    def _write_text(self, img, left_fit, right_fit):
        p = self.profile
        y_eval = self.size[1] - 1
        r_left = ld.curvature_m(left_fit, y_eval, p.xm_per_pix, p.ym_per_pix)
        r_right = ld.curvature_m(right_fit, y_eval, p.xm_per_pix, p.ym_per_pix)
        radius = (r_left + r_right) / 2
        offset = ld.vehicle_offset_m(left_fit, right_fit, y_eval, self.camera_x, p.xm_per_pix)

        radius_txt = "straight road" if radius > 3000 else f"{radius:.0f} m"
        side = "right" if offset > 0 else "left"
        lines = [f"Radius of curvature: {radius_txt}",
                 f"Vehicle is {abs(offset):.2f} m {side} of center"]
        scale = 1.2 * p.pix_scale + 0.1
        for i, txt in enumerate(lines):
            y = int((50 + 50 * i) * p.pix_scale) + 10
            cv.putText(img, txt, (30, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6)
            cv.putText(img, txt, (30, y), cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        self.last_radius, self.last_offset = radius, offset

    def _debug_view(self, result, binary_colour, detection):
        """Result on top, binary mask and bird's-eye detection below (half size)."""
        h, w = result.shape[:2]
        half = (w // 2, h // 2)
        bottom = np.hstack((cv.resize(binary_colour, half), cv.resize(detection, half)))
        return np.vstack((result, bottom[:, :w]))
