#!/usr/bin/env python3
# set_start_pose.py
#
# Record the current PSM position as a fixed start pose, then move the PSM
# back to that pose before each recording session.
#
# Usage:
#   # Save current PSM position as start pose
#   python set_start_pose.py --save
#
#   # Move PSM to the saved start pose
#   python set_start_pose.py --goto
#
#   # Show the saved start pose
#   python set_start_pose.py --show
#
#   # Use a custom file path (default: ./start_pose.json)
#   python set_start_pose.py --save --pose_file ./my_start_pose.json

import os
import sys
import time
import json
import argparse
import numpy as np

POSE_FILE_DEFAULT = "./start_pose.json"
WATCHDOG_SAFE_HZ  = 20.0


# ============================================================
# Connect to simulator
# ============================================================

def connect_psm(psm_name: str):
    try:
        from surgical_robotics_challenge.simulation_manager import SimulationManager
        from surgical_robotics_challenge.psm_arm import PSM
    except ImportError:
        print("ERROR: surgical_robotics_challenge library not found")
        print("Install with: python -m pip install -e ~/surgical_robotics_challenge/scripts/")
        sys.exit(1)

    print(f"Connecting to simulator...", flush=True)
    sim = SimulationManager('start_pose_tool')
    time.sleep(0.5)

    print(f"Loading {psm_name}...", flush=True)
    psm = PSM(sim, psm_name, add_joint_errors=False)
    time.sleep(0.5)

    if not psm.is_present():
        print(f"ERROR: {psm_name} not found - check AMBF is running correctly")
        sys.exit(1)

    print(f"{psm_name} ready\n")
    return psm


# ============================================================
# Save current position
# ============================================================

def save_pose(psm_name: str, pose_file: str):
    psm = connect_psm(psm_name)

    jp = psm.measured_jp()
    if jp is None:
        print("ERROR: could not read current joint positions")
        sys.exit(1)

    jp = list(jp)

    try:
        jaw = float(psm.get_jaw_angle())
    except Exception:
        jaw = 0.0
        print("Warning: could not read jaw angle, defaulting to 0.0")

    pose = {
        "psm_name":    psm_name,
        "servo_jp":    jp,
        "jaw":         jaw,
        "joint_names": [
            "yaw", "pitch", "insertion",
            "tool_roll", "tool_pitch", "tool_yaw"
        ],
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(pose_file, "w") as f:
        json.dump(pose, f, indent=2)

    print(f"Start pose saved to: {pose_file}")
    print(f"\n  servo_jp : {[round(v, 5) for v in jp]}")
    print(f"  jaw      : {jaw:.5f}")
    print(f"\n  Joint breakdown:")
    for i, (name, val) in enumerate(zip(pose["joint_names"], jp)):
        print(f"    j{i} {name:20s}: {val:+.5f}")


# ============================================================
# Move to saved position
# ============================================================

def goto_pose(psm_name: str, pose_file: str, speed: float, steps: int):
    if not os.path.isfile(pose_file):
        print(f"ERROR: pose file not found: {pose_file}")
        print(f"Run first: python set_start_pose.py --save")
        sys.exit(1)

    with open(pose_file, "r") as f:
        pose = json.load(f)

    target_jp  = pose["servo_jp"]
    target_jaw = pose["jaw"]
    saved_psm  = pose.get("psm_name", psm_name)

    if saved_psm != psm_name:
        print(f"Warning: pose was saved for {saved_psm}, currently using {psm_name}")

    print(f"Target start pose (from {pose_file}):")
    print(f"  servo_jp : {[round(v, 5) for v in target_jp]}")
    print(f"  jaw      : {target_jaw:.5f}")
    print(f"  saved at : {pose.get('saved_at', 'unknown')}")

    psm = connect_psm(psm_name)

    current_jp = list(psm.measured_jp() or target_jp)

    # Warmup: hold current position to keep Watch Dog alive
    print("Warmup 1s...", flush=True)
    dt = 1.0 / WATCHDOG_SAFE_HZ
    t0 = time.time()
    while time.time() - t0 < 1.0:
        psm.servo_jp(current_jp)
        psm.set_jaw_angle(0.0)
        time.sleep(dt)

    # Smooth interpolation to target
    print(f"Smooth move to start pose ({steps} steps @ {speed:.0f} Hz)...", flush=True)
    dt_move     = 1.0 / speed
    target_arr  = np.array(target_jp,  dtype=np.float32)
    current_arr = np.array(current_jp, dtype=np.float32)

    for i in range(steps + 1):
        t        = i / steps
        t_smooth = t * t * (3 - 2 * t)   # ease in/out
        interp   = (current_arr + t_smooth * (target_arr - current_arr)).tolist()
        psm.servo_jp(interp)
        psm.set_jaw_angle(target_jaw)
        time.sleep(dt_move)

    # Send a few extra commands to ensure arrival
    for _ in range(10):
        psm.servo_jp(target_jp)
        psm.set_jaw_angle(target_jaw)
        time.sleep(dt_move)

    print(f"PSM is at start pose - ready to record")


# ============================================================
# Show saved position
# ============================================================

def show_pose(pose_file: str):
    if not os.path.isfile(pose_file):
        print(f"Pose file not found: {pose_file}")
        print(f"Run first: python set_start_pose.py --save")
        return

    with open(pose_file, "r") as f:
        pose = json.load(f)

    print(f"=== Saved start pose ({pose_file}) ===")
    print(f"  PSM      : {pose.get('psm_name', 'unknown')}")
    print(f"  saved at : {pose.get('saved_at', 'unknown')}")
    print(f"  jaw      : {pose['jaw']:.5f}")
    print(f"\n  servo_jp:")
    names = pose.get("joint_names",
                     ["yaw","pitch","insertion","roll","tool_pitch","tool_yaw"])
    for i, (name, val) in enumerate(zip(names, pose["servo_jp"])):
        print(f"    j{i} {name:20s}: {val:+.5f}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Save and restore PSM start pose for consistent recording"
    )
    p.add_argument("--psm",       default="psm1")
    p.add_argument("--pose_file", default=POSE_FILE_DEFAULT,
                   help=f"Pose file path (default: {POSE_FILE_DEFAULT})")

    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--save", action="store_true",
                        help="Save current PSM position as start pose")
    action.add_argument("--goto", action="store_true",
                        help="Move PSM to saved start pose")
    action.add_argument("--show", action="store_true",
                        help="Print saved start pose")

    p.add_argument("--speed", type=float, default=10.0,
                   help="Move speed in Hz for --goto (lower = safer, default: 10)")
    p.add_argument("--steps", type=int, default=100,
                   help="Interpolation steps for --goto (default: 100)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.show:
        show_pose(args.pose_file)
    elif args.save:
        save_pose(args.psm, args.pose_file)
    elif args.goto:
        goto_pose(args.psm, args.pose_file, args.speed, args.steps)


if __name__ == "__main__":
    main()