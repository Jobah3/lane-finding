"""
Run the pipeline on a video:

    python src/run_video.py test_videos/project_video03.mp4
    python src/run_video.py test_videos/challenge01.mp4 --show
    python src/run_video.py test_videos/project_video01.mp4 --debug
    python src/run_video.py --all

Options:
    --show     show frames live (Esc = stop, Space = pause)
    --debug    add binary mask + bird's-eye detection under the result
    --all      process every video in test_videos/
Output: output_videos/<name>_out.mp4 (or <name>_debug.mp4)
"""
import argparse
import time
from pathlib import Path

import cv2 as cv

import calibration
import config
from pipeline import LaneFinder


def process_video(path, mtx, dist, show=False, debug=False):
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Cannot open {path}")
        return
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    profile = config.pick_profile(path, (w, h))

    finder = LaneFinder((w, h), profile, mtx, dist, video=True, debug=debug)

    config.OUTPUT_VIDEOS_DIR.mkdir(exist_ok=True)
    suffix = "_debug" if debug else "_out"
    out_path = config.OUTPUT_VIDEOS_DIR / f"{Path(path).stem}{suffix}.mp4"
    out_h = h + h // 2 if debug else h
    writer = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (w, out_h))

    print(f"{Path(path).name}: {w}x{h}, {total} frames, camera={profile}")
    start = time.time()
    n = 0
    status_count = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = finder.process(frame)
        writer.write(result)
        status_count[finder.status] = status_count.get(finder.status, 0) + 1
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{total} frames")

        if show:
            cv.imshow("Lane finding", result)
            key = cv.waitKey(1) & 0xFF
            if key == 27:                       # Esc
                break
            if key == 32:                       # Space = pause
                cv.waitKey(0)

    cap.release()
    writer.release()
    if show:
        cv.destroyAllWindows()
    print(f"  done: {n} frames in {time.time() - start:.1f} s -> {out_path}")
    # ok = new fit accepted; kept previous = bad frame, last good lane used
    print("  frames:", ", ".join(f"{k}: {v}" for k, v in status_count.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", help="path to the video")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--all", action="store_true", help="all videos in test_videos/")
    args = parser.parse_args()

    mtx, dist = calibration.load_calibration()
    if args.all:
        videos = sorted(config.TEST_VIDEOS_DIR.glob("*.mp4"))
    elif args.video:
        videos = [Path(args.video)]
    else:
        parser.error("give a video path or --all")
    for v in videos:
        process_video(v, mtx, dist, show=args.show, debug=args.debug)


if __name__ == "__main__":
    main()
