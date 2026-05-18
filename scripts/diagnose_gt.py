"""
Diagnostic tool: read one frame from a GT .npy file, print keypoint
coordinates with indices, visualize the skeleton with labels, and save
the figure to outputs/diagnostics/.

Usage:
    python scripts/diagnose_gt.py --gt-file <path> --frame 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

H36M17_KEYPOINT_NAMES = [
    "pelvis",      # 0
    "r_hip",       # 1
    "r_knee",      # 2
    "r_ankle",     # 3
    "l_hip",       # 4
    "l_knee",      # 5
    "l_ankle",     # 6
    "spine",       # 7
    "thorax",      # 8
    "neck",        # 9
    "head",        # 10
    "l_shoulder",  # 11
    "l_elbow",     # 12
    "l_wrist",     # 13
    "r_shoulder",  # 14
    "r_elbow",     # 15
    "r_wrist",     # 16
]

H36M17_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3),       # right leg
    (0, 4), (4, 5), (5, 6),       # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine → head
    (8, 11), (11, 12), (12, 13),  # left arm
    (8, 14), (14, 15), (15, 16),  # right arm
)


def _valid(pt: np.ndarray) -> bool:
    return bool(np.isfinite(pt).all())


def main():
    parser = argparse.ArgumentParser(description="Diagnose a single GT frame")
    parser.add_argument("--gt-file", required=True, help="Path to a GT .npy file (e.g. E01_S01_A01.npy)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    args = parser.parse_args()

    gt = np.load(args.gt_file)
    print(f"GT file shape: {gt.shape}")
    if gt.shape[1:] not in ((17, 2), (17, 3)):
        print(f"WARNING: unexpected keypoint shape {gt.shape[1:]}, expected (17, 2) or (17, 3)")

    frame = args.frame
    if frame >= gt.shape[0]:
        print(f"Frame {frame} out of range (max {gt.shape[0] - 1})")
        return

    kpts = gt[frame, :, :2].copy()  # (17, 2)

    print(f"\n--- Frame {frame} Keypoints ---")
    for i in range(kpts.shape[0]):
        name = H36M17_KEYPOINT_NAMES[i] if i < len(H36M17_KEYPOINT_NAMES) else f"joint_{i}"
        x, y = kpts[i]
        status = "valid" if _valid(kpts[i]) else "invalid"
        print(f"  [{i:2d}] {name:12s}  x={x: 9.6f}  y={y: 9.6f}  ({status})")

    # --- visualization ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"Frame {frame} — {Path(args.gt_file).name}")

    # draw bones
    for start, end in H36M17_BONE_EDGES:
        if start < kpts.shape[0] and end < kpts.shape[0]:
            p1, p2 = kpts[start], kpts[end]
            if _valid(p1) and _valid(p2):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue", linewidth=1.5, alpha=0.6)

    # draw keypoints
    for i in range(kpts.shape[0]):
        x, y = kpts[i]
        if _valid(kpts[i]):
            name = H36M17_KEYPOINT_NAMES[i] if i < len(H36M17_KEYPOINT_NAMES) else f"J{i}"
            ax.scatter(x, y, c="red", s=40, zorder=5)
            ax.annotate(
                f"{i}:{name}\n({x:.3f},{y:.3f})",
                (x, y),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=7,
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.gt_file).stem
    out_path = out_dir / f"{stem}_frame{frame:04d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved visualization to {out_path}")


if __name__ == "__main__":
    main()
