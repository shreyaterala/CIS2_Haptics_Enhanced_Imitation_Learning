#!/usr/bin/env python3
# view_episode.py
#
# View HDF5 recorded episode data with image playback and joint positions.
#
# Usage:
#   python view_episode.py --episode ./data/ambf_dataset_new/episode_000.hdf5
#   python view_episode.py --episode ./data/ambf_dataset_new/episode_000.hdf5 --cam wrist
#   python view_episode.py --episode ./data/ambf_dataset_new/episode_000.hdf5 --all_cams
#   python view_episode.py --episode ./data/ambf_dataset_new/episode_000.hdf5 --info

import os
import sys
import argparse
import cv2
import h5py
import numpy as np


JOINT_NAMES = ['yaw', 'pitch', 'insertion', 'roll', 'tpitch', 'tyaw', 'jaw']


# ============================================================
# Load
# ============================================================

def load_episode(path):
    with h5py.File(path, "r") as f:
        qpos   = f["observations/qpos"][:].astype(np.float32)
        action = f["action"][:].astype(np.float32)
        attrs  = dict(f.attrs)

        images = {}
        for cam in ["left", "right", "wrist"]:
            key = f"observations/images/{cam}"
            if key in f:
                images[cam] = f[key][:]

    return {
        "qpos":   qpos,
        "action": action,
        "images": images,
        "fps":    float(attrs.get("fps", 20.0)),
        "steps":  qpos.shape[0],
        "attrs":  attrs,
    }


# ============================================================
# Print info
# ============================================================

def print_info(path, data):
    print(f"\n{'='*55}")
    print(f"File: {path}")
    print(f"{'='*55}")

    attrs = data["attrs"]
    print(f"\n--- Recording Parameters ---")
    for key in ["episode_idx", "fps", "captured_steps", "capture_ratio",
                "recorded_at_iso", "joint_dim"]:
        if key in attrs:
            print(f"  {key:25s}: {attrs[key]}")

    print(f"\n--- Data Shape ---")
    print(f"  qpos   : {data['qpos'].shape}")
    print(f"  action : {data['action'].shape}")
    for cam, imgs in data["images"].items():
        print(f"  {cam:7s}: {imgs.shape}  dtype={imgs.dtype}")

    print(f"\n--- qpos Statistics ---")
    qpos  = data["qpos"]
    names = JOINT_NAMES[:qpos.shape[1]]
    print(f"  {'joint':10s}  {'min':>8}  {'max':>8}  {'mean':>8}  {'motion':>8}")
    for i, name in enumerate(names):
        col    = qpos[:, i]
        motion = np.abs(np.diff(col)).mean()
        print(f"  {name:10s}  {col.min():>+8.4f}  {col.max():>+8.4f}"
              f"  {col.mean():>+8.4f}  {motion:>8.5f}")

    print(f"\n--- action vs qpos[t+1] error (should be 0) ---")
    expected = np.vstack([qpos[1:], qpos[-1:]])
    diff     = np.abs(data["action"] - expected)
    print(f"  max error : {diff.max():.8f}")
    print(f"  mean error: {diff.mean():.8f}")
    if diff.max() < 1e-5:
        print("  OK: action == qpos[t+1]")
    else:
        print("  WARNING: action does not match qpos[t+1]")

    print()


# ============================================================
# Visualization
# ============================================================

def draw_frame(imgs_dict, cam_order, qpos_row, action_row, step, total, fps):
    H, W   = 240, 320
    INFO_W = 280

    panels = []
    for cam in cam_order:
        img = imgs_dict.get(cam)
        if img is not None:
            img = cv2.resize(img, (W, H))
        else:
            img = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(img, f"No {cam}", (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 1)
        cv2.putText(img, cam.upper(), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        panels.append(img)

    cam_strip = np.hstack(panels)

    info = np.zeros((H, INFO_W, 3), dtype=np.uint8)

    # Progress bar
    prog = int((step / max(total-1, 1)) * (INFO_W - 20))
    cv2.rectangle(info, (10, 8),  (INFO_W-10, 20), (60,60,60),   -1)
    cv2.rectangle(info, (10, 8),  (10+prog,   20), (0,200,80),   -1)
    cv2.putText(info, f"step {step+1}/{total}  {fps:.0f}fps",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)

    # Joint values
    cv2.putText(info, "  joint     qpos    action",
                (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

    names = JOINT_NAMES[:qpos_row.shape[0]]
    for i, name in enumerate(names):
        y         = 70 + i * 22
        q         = qpos_row[i]
        a         = action_row[i] if i < action_row.shape[0] else 0.0
        err       = abs(a - q)
        intensity = min(int(err / 0.1 * 255), 255)
        color     = (80, 255-intensity, 255-intensity)
        cv2.putText(info, f"{name:7s}  {q:+.3f}  {a:+.3f}",
                    (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

    return np.hstack([cam_strip, info])


def play(data, cam_order, start_step, speed):
    steps  = data["steps"]
    fps    = data["fps"] * speed
    dt_ms  = max(1, int(1000.0 / fps))

    win = "Episode Viewer  [space=pause  a/d=step  ,/.=10steps  s=screenshot  q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 320 * len(cam_order) + 280, 240)

    i      = start_step
    paused = False

    while True:
        imgs_this = {cam: data["images"][cam][i]
                     for cam in cam_order if cam in data["images"]}

        frame = draw_frame(
            imgs_dict  = imgs_this,
            cam_order  = cam_order,
            qpos_row   = data["qpos"][i],
            action_row = data["action"][i],
            step       = i,
            total      = steps,
            fps        = data["fps"],
        )

        cv2.imshow(win, frame)
        key = cv2.waitKey(1 if paused else dt_ms) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            fname = f"frame_{i:04d}.png"
            cv2.imwrite(fname, frame)
            print(f"Screenshot saved: {fname}")
        elif key in (81, ord('a')):
            i      = max(0, i - 1)
            paused = True
        elif key in (83, ord('d')):
            i      = min(steps-1, i + 1)
            paused = True
        elif key == ord(','):
            i      = max(0, i - 10)
            paused = True
        elif key == ord('.'):
            i      = min(steps-1, i + 10)
            paused = True

        if not paused:
            i += 1
            if i >= steps:
                i = 0

    cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="View HDF5 recorded episode data")
    p.add_argument("--episode",  required=True, help="Path to .hdf5 file")
    p.add_argument("--cam",      default="wrist",
                   choices=["left", "right", "wrist"],
                   help="Camera to display (default: wrist)")
    p.add_argument("--all_cams", action="store_true",
                   help="Display all three cameras side by side")
    p.add_argument("--info",     action="store_true",
                   help="Print data statistics only, no video playback")
    p.add_argument("--start",    type=int,   default=0,
                   help="Start from this step (default: 0)")
    p.add_argument("--speed",    type=float, default=1.0,
                   help="Playback speed multiplier (default: 1.0)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.episode):
        print(f"ERROR: file not found: {args.episode}")
        sys.exit(1)

    print(f"Loading: {args.episode} ...", flush=True)
    data = load_episode(args.episode)

    print_info(args.episode, data)

    if args.info:
        return

    if args.all_cams:
        cam_order = [c for c in ["left", "right", "wrist"]
                     if c in data["images"]]
    else:
        if args.cam not in data["images"]:
            available = list(data["images"].keys())
            print(f"Warning: camera '{args.cam}' not found, available: {available}")
            args.cam = available[0]
        cam_order = [args.cam]

    print(f"Playing camera(s): {cam_order}")
    print("Keys: space=pause  a/d=prev/next frame  ,/.=jump 10 frames  s=screenshot  q=quit\n")

    play(data, cam_order, args.start, args.speed)


if __name__ == "__main__":
    main()