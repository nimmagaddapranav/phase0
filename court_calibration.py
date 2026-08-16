#!/usr/bin/env python3
"""
Phase 0: Court Calibration (matches Court Annotator v5)

Takes keypoint annotations from the Court Annotator v5 tool and computes:
  - Homography matrix (pixel → court coordinates in meters)
  - Court mask (binary mask of playable area)
  - Service box regions
  - Court overlay visualization

Usage:
  python court_calibration.py --input court_calibration.json --video IMG_3524.mp4

Output:
  court_config.json  — all calibration data for downstream phases
  court_overlay.jpg  — visualization of projected court lines on frame
  court_mask.png     — binary mask of the court
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════
# BWF Standard Badminton Court Dimensions (meters)
# Origin: near-left corner of DOUBLES court
# X-axis: left → right (width)
# Y-axis: near baseline (0) → far baseline (13.4)
# ══════════════════════════════════════════════════════════════
COURT = {
    "length": 13.40,
    "width_doubles": 6.10,
    "width_singles": 5.18,
    "singles_offset": 0.46,        # (6.10 - 5.18) / 2
    "net": 6.70,                   # net y-position (13.40 / 2)
    "short_svc_near": 4.72,       # short service line near (6.70 - 1.98)
    "short_svc_far": 8.68,        # short service line far  (6.70 + 1.98)
    "long_svc_near": 0.76,        # long service line near  (doubles back boundary)
    "long_svc_far": 12.64,        # long service line far   (13.40 - 0.76)
    "center_x": 3.05,             # center line x (6.10 / 2)
}

# ══════════════════════════════════════════════════════════════
# All 32 annotatable points — MUST MATCH court_annotator_v5.jsx
# ══════════════════════════════════════════════════════════════
C = COURT
ALL_COURT_POINTS = {
    # Baselines & Corners
    "near_left":           np.array([0.0,            0.0]),
    "near_right":          np.array([C["width_doubles"], 0.0]),
    "near_center":         np.array([C["center_x"],  0.0]),
    "far_left":            np.array([0.0,            C["length"]]),
    "far_right":           np.array([C["width_doubles"], C["length"]]),
    "far_center":          np.array([C["center_x"],  C["length"]]),

    # Net Poles (physical landmarks)
    "net_pole_left":       np.array([0.0,            C["net"]]),
    "net_pole_right":      np.array([C["width_doubles"], C["net"]]),

    # Long Service Line — Near (Y=0.76)
    "long_near_left":      np.array([0.0,            C["long_svc_near"]]),
    "long_near_right":     np.array([C["width_doubles"], C["long_svc_near"]]),
    "long_near_center":    np.array([C["center_x"],  C["long_svc_near"]]),
    "long_near_singles_l": np.array([C["singles_offset"], C["long_svc_near"]]),
    "long_near_singles_r": np.array([C["width_doubles"] - C["singles_offset"], C["long_svc_near"]]),

    # Short Service Line — Near (Y=4.72)
    "short_near_left":     np.array([0.0,            C["short_svc_near"]]),
    "short_near_right":    np.array([C["width_doubles"], C["short_svc_near"]]),
    "short_near_center":   np.array([C["center_x"],  C["short_svc_near"]]),
    "short_near_singles_l": np.array([C["singles_offset"], C["short_svc_near"]]),
    "short_near_singles_r": np.array([C["width_doubles"] - C["singles_offset"], C["short_svc_near"]]),

    # Short Service Line — Far (Y=8.68)
    "short_far_left":      np.array([0.0,            C["short_svc_far"]]),
    "short_far_right":     np.array([C["width_doubles"], C["short_svc_far"]]),
    "short_far_center":    np.array([C["center_x"],  C["short_svc_far"]]),
    "short_far_singles_l": np.array([C["singles_offset"], C["short_svc_far"]]),
    "short_far_singles_r": np.array([C["width_doubles"] - C["singles_offset"], C["short_svc_far"]]),

    # Long Service Line — Far (Y=12.64)
    "long_far_left":       np.array([0.0,            C["long_svc_far"]]),
    "long_far_right":      np.array([C["width_doubles"], C["long_svc_far"]]),
    "long_far_center":     np.array([C["center_x"],  C["long_svc_far"]]),
    "long_far_singles_l":  np.array([C["singles_offset"], C["long_svc_far"]]),
    "long_far_singles_r":  np.array([C["width_doubles"] - C["singles_offset"], C["long_svc_far"]]),

    # Singles × Baselines
    "near_singles_l":      np.array([C["singles_offset"], 0.0]),
    "near_singles_r":      np.array([C["width_doubles"] - C["singles_offset"], 0.0]),
    "far_singles_l":       np.array([C["singles_offset"], C["length"]]),
    "far_singles_r":       np.array([C["width_doubles"] - C["singles_offset"], C["length"]]),
}


def compute_homography(pixel_points, court_points):
    """
    Compute homography from pixel coordinates to court coordinates.
    Returns H (pixel→court) and H_inv (court→pixel).
    """
    src_pts = []
    dst_pts = []

    for name in pixel_points:
        if name in court_points:
            src_pts.append(pixel_points[name])
            dst_pts.append(court_points[name])

    if len(src_pts) < 4:
        print(f"ERROR: Need at least 4 matched points, got {len(src_pts)}")
        sys.exit(1)

    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)

    print(f"  Computing homography from {len(src_pts)} point correspondences...")

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    H_inv, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Validate reprojection error
    errors = []
    for name in pixel_points:
        if name not in court_points:
            continue
        px_pt = pixel_points[name]
        px_h = np.array([px_pt[0], px_pt[1], 1.0])
        ct_h = H @ px_h
        ct = ct_h[:2] / ct_h[2]
        expected = court_points[name]
        err = np.linalg.norm(ct - expected)
        errors.append(err)
        print(f"    {name}: pixel({px_pt[0]:.0f},{px_pt[1]:.0f}) -> "
              f"court({ct[0]:.2f},{ct[1]:.2f})m, "
              f"expected({expected[0]:.2f},{expected[1]:.2f})m, "
              f"error={err:.4f}m")

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"  Reprojection error: mean={mean_err:.4f}m, max={max_err:.4f}m")

    if mean_err > 0.5:
        print("  WARNING: High reprojection error! Check your keypoint annotations.")

    return H, H_inv


def court_to_pixel(H_inv, cx, cy):
    """Transform court meters to pixel coordinates."""
    pt = np.array([cx, cy, 1.0])
    px = H_inv @ pt
    return px[0] / px[2], px[1] / px[2]


def pixel_to_court(H, px, py):
    """Transform pixel coordinates to court meters."""
    pt = np.array([px, py, 1.0])
    ct = H @ pt
    return ct[0] / ct[2], ct[1] / ct[2]


def generate_court_mask(H_inv, image_shape):
    """Generate binary mask of the doubles court in pixel space."""
    h, w = image_shape[:2]
    wd = COURT["width_doubles"]
    ln = COURT["length"]

    corners_court = [[0, 0], [wd, 0], [wd, ln], [0, ln]]
    corners_pixel = []
    for cx, cy in corners_court:
        px, py = court_to_pixel(H_inv, cx, cy)
        corners_pixel.append([int(px), int(py)])

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(corners_pixel, dtype=np.int32), 255)
    return mask


def generate_service_boxes(H_inv):
    """Generate pixel coordinates for the 4 service boxes."""
    cx = COURT["center_x"]
    wd = COURT["width_doubles"]
    net = COURT["net"]
    ss_n = COURT["short_svc_near"]
    ss_f = COURT["short_svc_far"]

    boxes = {
        "near_even": {
            "desc": "Near court, even (right side)",
            "court_coords": [[cx, ss_n], [wd, ss_n], [wd, net], [cx, net]],
        },
        "near_odd": {
            "desc": "Near court, odd (left side)",
            "court_coords": [[0, ss_n], [cx, ss_n], [cx, net], [0, net]],
        },
        "far_even": {
            "desc": "Far court, even (left side from camera)",
            "court_coords": [[0, net], [cx, net], [cx, ss_f], [0, ss_f]],
        },
        "far_odd": {
            "desc": "Far court, odd (right side from camera)",
            "court_coords": [[cx, net], [wd, net], [wd, ss_f], [cx, ss_f]],
        },
    }

    for box_name, box in boxes.items():
        pixel_coords = []
        for cc in box["court_coords"]:
            px, py = court_to_pixel(H_inv, cc[0], cc[1])
            pixel_coords.append([int(px), int(py)])
        box["pixel_coords"] = pixel_coords

    return boxes


def draw_court_overlay(frame, H_inv, pixel_points):
    """Draw all court lines, service boxes, and annotations on frame."""
    overlay = frame.copy()

    def draw_line(cx1, cy1, cx2, cy2, color, thickness=2):
        px1, py1 = court_to_pixel(H_inv, cx1, cy1)
        px2, py2 = court_to_pixel(H_inv, cx2, cy2)
        cv2.line(overlay, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness)

    wd = COURT["width_doubles"]
    so = COURT["singles_offset"]
    ln = COURT["length"]
    net = COURT["net"]
    ss_n = COURT["short_svc_near"]
    ss_f = COURT["short_svc_far"]
    ls_n = COURT["long_svc_near"]
    ls_f = COURT["long_svc_far"]
    cx = COURT["center_x"]

    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    RED = (0, 0, 255)
    ORANGE = (0, 165, 255)
    PURPLE = (255, 100, 255)

    # Doubles boundary (green)
    draw_line(0, 0, wd, 0, GREEN, 2)       # near baseline
    draw_line(0, ln, wd, ln, GREEN, 2)      # far baseline
    draw_line(0, 0, 0, ln, GREEN, 2)        # left sideline
    draw_line(wd, 0, wd, ln, GREEN, 2)      # right sideline

    # Singles sidelines (purple, thinner)
    draw_line(so, 0, so, ln, PURPLE, 1)
    draw_line(wd - so, 0, wd - so, ln, PURPLE, 1)

    # Long service lines (orange)
    draw_line(0, ls_n, wd, ls_n, ORANGE, 1)
    draw_line(0, ls_f, wd, ls_f, ORANGE, 1)

    # Short service lines (cyan)
    draw_line(0, ss_n, wd, ss_n, CYAN, 2)
    draw_line(0, ss_f, wd, ss_f, CYAN, 2)

    # Center line — TWO separate segments (not connected across net)
    draw_line(cx, 0, cx, ss_n, YELLOW, 1)      # near: baseline → short svc
    draw_line(cx, ss_f, cx, ln, YELLOW, 1)      # far:  short svc → baseline

    # Net (red, thick)
    draw_line(0, net, wd, net, RED, 3)

    # Service box labels
    for label, court_pos in [
        ("NEAR-EVEN", (cx + (wd - cx) / 2, (ss_n + net) / 2)),
        ("NEAR-ODD", (cx / 2, (ss_n + net) / 2)),
        ("FAR-EVEN", (cx / 2, (net + ss_f) / 2)),
        ("FAR-ODD", (cx + (wd - cx) / 2, (net + ss_f) / 2)),
    ]:
        px, py = court_to_pixel(H_inv, court_pos[0], court_pos[1])
        cv2.putText(overlay, label, (int(px) - 40, int(py)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw annotated keypoints
    for name, coords in pixel_points.items():
        px, py = int(coords[0]), int(coords[1])
        cv2.circle(overlay, (px, py), 8, (0, 255, 255), -1)
        cv2.circle(overlay, (px, py), 10, (0, 255, 255), 2)
        short = name.replace("_", " ")[:20]
        cv2.putText(overlay, short, (px + 12, py - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Blend
    result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
    return result


def extract_frame(video_path, frame_idx=100):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Court Calibration")
    parser.add_argument("--input", type=str, default=None,
                        help="JSON from Court Annotator v5")
    parser.add_argument("--video", type=str, required=True,
                        help="Match video path")
    parser.add_argument("--frame-idx", type=int, default=100,
                        help="Frame index for visualization")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as video)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.video))
    os.makedirs(output_dir, exist_ok=True)

    # ── Load keypoints ──
    if not args.input:
        print("ERROR: --input required (JSON from Court Annotator v5)")
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    pixel_points = {}
    for name, coords in data["pixel_points"].items():
        pixel_points[name] = np.array(coords, dtype=np.float64)

    print(f"Loaded {len(pixel_points)} keypoints from {args.input}")
    print(f"Points: {', '.join(pixel_points.keys())}")

    # Match pixel points to known court coordinates
    court_points = {}
    unmatched = []
    for name in pixel_points:
        if name in ALL_COURT_POINTS:
            court_points[name] = ALL_COURT_POINTS[name]
        else:
            unmatched.append(name)
            print(f"  WARNING: Unknown point ID '{name}' - skipping")

    if unmatched:
        print(f"  Valid IDs: {', '.join(sorted(ALL_COURT_POINTS.keys()))}")

    # ── Compute homography ──
    print(f"\n[1/4] Computing homography...")
    H, H_inv = compute_homography(pixel_points, court_points)

    # ── Extract frame ──
    print(f"\n[2/4] Extracting frame {args.frame_idx}...")
    frame = extract_frame(args.video, args.frame_idx)
    h, w = frame.shape[:2]
    print(f"  Frame: {w}x{h}")

    # ── Court mask ──
    print(f"\n[3/4] Generating court mask...")
    mask = generate_court_mask(H_inv, frame.shape)
    mask_path = os.path.join(output_dir, "court_mask.png")
    cv2.imwrite(mask_path, mask)
    court_pct = np.sum(mask > 0) / (h * w) * 100
    print(f"  Saved: {mask_path} (court covers {court_pct:.1f}% of frame)")

    # ── Service boxes ──
    service_boxes = generate_service_boxes(H_inv)

    # ── Overlay ──
    print(f"\n[4/4] Drawing court overlay...")
    overlay = draw_court_overlay(frame, H_inv, pixel_points)
    overlay_path = os.path.join(output_dir, "court_overlay.jpg")
    cv2.imwrite(overlay_path, overlay)
    print(f"  Saved: {overlay_path}")

    # ── Test transforms ──
    print(f"\n  Transform tests:")
    tests = [
        ("Court center (net)", COURT["center_x"], COURT["net"]),
        ("Near-odd svc box center", COURT["center_x"] / 2,
         (COURT["short_svc_near"] + COURT["net"]) / 2),
        ("Far-odd svc box center",
         COURT["center_x"] + COURT["center_x"] / 2,
         (COURT["net"] + COURT["short_svc_far"]) / 2),
    ]
    for label, cxx, cyy in tests:
        px, py = court_to_pixel(H_inv, cxx, cyy)
        print(f"    {label}: court({cxx:.2f}, {cyy:.2f})m -> pixel({px:.0f}, {py:.0f})")

    # ── Save config ──
    config = {
        "video": os.path.abspath(args.video),
        "frame_size": {"width": w, "height": h},
        "frame_idx_used": args.frame_idx,
        "homography_pixel_to_court": H.tolist(),
        "homography_court_to_pixel": H_inv.tolist(),
        "pixel_points": {k: v.tolist() for k, v in pixel_points.items()},
        "court_dimensions": COURT,
        "service_boxes": {
            name: {
                "description": box["desc"],
                "court_coords": box["court_coords"],
                "pixel_coords": box["pixel_coords"],
            }
            for name, box in service_boxes.items()
        },
        "files": {
            "court_mask": os.path.abspath(mask_path),
            "court_overlay": os.path.abspath(overlay_path),
        },
    }

    config_path = os.path.join(output_dir, "court_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved: {config_path}")

    print(f"\n{'='*55}")
    print(f"  Phase 0 complete!")
    print(f"  court_config.json  - use in all downstream phases")
    print(f"  court_overlay.jpg  - REVIEW THIS for accuracy")
    print(f"  court_mask.png     - binary court mask")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
