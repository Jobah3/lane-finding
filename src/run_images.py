"""
Run the pipeline on every image in test_images/ and save each step.

    python src/run_images.py                 all images
    python src/run_images.py test1.jpg       one image
    python src/run_images.py --show          also show the result windows

For every image <name> the following files go to output_images/<name>/:
    1_undistorted.jpg    2_binary.jpg (yellow = yellow paint, blue = white
    paint, green = Sobel edges)    3_warped.jpg    4_warped_binary.jpg
    5_detection.jpg (sliding windows + fitted polynomials)    6_result.jpg
Plus output_images/perspective_check.jpg (src/dst points on straight_lines1).
"""
import argparse

import cv2 as cv

import calibration
import config
from perspective import Perspective
from pipeline import LaneFinder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="file names in test_images/ (default: all)")
    parser.add_argument("--show", action="store_true", help="show results with cv.imshow")
    args = parser.parse_args()

    mtx, dist = calibration.load_calibration()
    out_root = config.OUTPUT_IMAGES_DIR
    out_root.mkdir(exist_ok=True)

    names = args.images or sorted(p.name for p in config.TEST_IMAGES_DIR.glob("*.jpg"))
    for name in names:
        path = config.TEST_IMAGES_DIR / name
        img = cv.imread(str(path))
        if img is None:
            print(f"Cannot read {path}")
            continue
        size = (img.shape[1], img.shape[0])
        profile = config.pick_profile(path, size)

        finder = LaneFinder(size, profile, mtx, dist, video=False)
        result = finder.process(img)

        out_dir = out_root / path.stem
        out_dir.mkdir(exist_ok=True)
        for i, key in enumerate(["undistorted", "binary", "warped",
                                 "warped_binary", "detection", "result"], start=1):
            cv.imwrite(str(out_dir / f"{i}_{key}.jpg"), finder.stages[key])
        print(f"{name:24s} camera={profile:4s} radius={finder.last_radius:8.0f} m  "
              f"offset={finder.last_offset:+.2f} m  ({finder.status})")

        if name == "straight_lines1.jpg":
            persp = Perspective(finder.profile.src, finder.profile.dst)
            check = persp.draw_check(finder.stages["undistorted"])
            cv.imwrite(str(out_root / "perspective_check.jpg"), check)

        if args.show:
            cv.imshow(name, result)
            cv.waitKey(0)
            cv.destroyAllWindows()
    print(f"Saved to {out_root}")


if __name__ == "__main__":
    main()
