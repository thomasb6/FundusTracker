#!/usr/bin/env python3
"""
sift_recover.py
---------------
Recover original-space annotations from a _sift.json annotation file.

Usage:
    python sift_recover.py --json my_image_sift.json \
                           --ref reference.png \
                           --target rotated.png \
                           [--output my_image_original.json]

How it works:
    1. Recomputes the SIFT homography M that maps target → reference
       (same algorithm as FundusTracker's align_images_sift_simple).
    2. Applies M⁻¹ to every shape coordinate in the JSON, mapping
       annotations from warped/SIFT space back to original image space.
    3. Writes the result as a JSON list identical in structure to the
       input — importable directly into FundusTracker Manual Segmentation.

Input JSON formats supported:
    - Plain list:   [{shape}, {shape}, ...]           (app _sift.json export)
    - Dict:         {"annotations": [...]}             (app Image Bank export)
    - Dict:         {"shapes_sift": [...]}             (app ZIP export)
    - Dict:         {"shapes": [...]}                  (legacy)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


# ── SIFT homography ────────────────────────────────────────────────────────────

def compute_homography(ref_path: str, target_path: str) -> np.ndarray:
    """Return 3×3 homography M such that warpPerspective(target, M) ≈ ref."""
    img_ref = cv2.imread(ref_path)
    img_target = cv2.imread(target_path)

    if img_ref is None:
        raise FileNotFoundError(f"Cannot read reference image: {ref_path}")
    if img_target is None:
        raise FileNotFoundError(f"Cannot read target image: {target_path}")

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_target, None)

    if des1 is None or des2 is None:
        raise RuntimeError("Not enough detail in one of the images for SIFT.")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        raise RuntimeError(
            f"Only {len(good)} good matches found — need at least 4. "
            "Check that both images share enough common content."
        )

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        raise RuntimeError("Homography computation failed (findHomography returned None).")

    n_inliers = int(mask.sum()) if mask is not None else len(good)
    print(f"  SIFT: {len(good)} good matches, {n_inliers} inliers")

    return M


# ── Coordinate transform ───────────────────────────────────────────────────────

def transform_shapes_to_original(shapes: list, M: np.ndarray) -> list:
    """Apply M⁻¹ to bring shapes from SIFT space back to original image space."""
    M_inv = np.linalg.inv(M)
    result = []

    for shape in shapes:
        s = shape.copy()

        if shape.get("type") == "path" and shape.get("path"):
            coords = re.findall(r"[-+]?\d*\.?\d+", shape["path"])
            if len(coords) >= 2:
                pts = np.array(
                    [[float(coords[i]), float(coords[i + 1]), 1.0]
                     for i in range(0, len(coords) - 1, 2)]
                )
                t = (M_inv @ pts.T).T
                t = t[:, :2] / t[:, 2:3]
                s["path"] = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in t) + " Z"

        elif shape.get("type") in ("rect", "circle"):
            corners = np.array([
                [shape["x0"], shape["y0"], 1.0],
                [shape["x1"], shape["y0"], 1.0],
                [shape["x0"], shape["y1"], 1.0],
                [shape["x1"], shape["y1"], 1.0],
            ])
            t = (M_inv @ corners.T).T
            t = t[:, :2] / t[:, 2:3]
            s["x0"] = float(t[:, 0].min())
            s["x1"] = float(t[:, 0].max())
            s["y0"] = float(t[:, 1].min())
            s["y1"] = float(t[:, 1].max())

        result.append(s)

    return result


# ── JSON I/O ───────────────────────────────────────────────────────────────────

def load_shapes(json_path: str) -> list:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("shapes_sift", "shapes", "annotations"):
            if data.get(key):
                return data[key]
    raise ValueError(
        f"Unrecognised JSON structure in {json_path}. "
        "Expected a list or a dict with 'shapes_sift', 'shapes', or 'annotations' key."
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert FundusTracker _sift.json annotations to original image space."
    )
    parser.add_argument("--json",   required=True, help="Path to the _sift.json annotation file")
    parser.add_argument("--ref",    required=True, help="Reference image used during SIFT alignment")
    parser.add_argument("--target", required=True, help="Target (rotated/moved) image that was aligned")
    parser.add_argument("--output", default=None,  help="Output JSON path (default: <input>_original.json)")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: annotation file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else json_path.with_name(
        json_path.stem.replace("_sift", "") + "_original.json"
    )

    print(f"Loading annotations from: {json_path}")
    shapes = load_shapes(str(json_path))
    print(f"  {len(shapes)} shape(s) found")

    print(f"Computing SIFT homography...")
    print(f"  ref:    {args.ref}")
    print(f"  target: {args.target}")
    M = compute_homography(args.ref, args.target)

    print("Transforming coordinates to original space...")
    shapes_original = transform_shapes_to_original(shapes, M)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shapes_original, f, indent=2)

    print(f"Done → {output_path}")


if __name__ == "__main__":
    main()
