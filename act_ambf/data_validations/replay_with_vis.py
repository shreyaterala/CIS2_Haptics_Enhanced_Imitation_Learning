#!/usr/bin/env python3
# replay_with_vis.py  (v2 - no unit conversion)
#
# Replay HDF5 episode with side-by-side visualization:
#   Left  : recorded image + recorded qpos
#   Right : live camera image + live qpos
#   Bottom: per-joint error bar chart
#
# Usage:
#   python replay_with_vis.py --episode ./data/ambf_dataset_new/episode_000.hdf5

import os
import sys
import time
import argparse

import cv2
import h5py
import numpy as np

WATCHDOG_HZ = 20.0
JOINT_NAMES = ['yaw', 'pitch', 'insert', 'roll', 'tpitch', 'tyaw', 'jaw']


# ============================================================
# Load HDF5
# ============================================================

def load_episode(path):
    with h5py.File(path, "r") as f:
        qpos   = f["observations/qpos"][:].astype(np.float32)
        images = f["observations/images/left"][:]
        attrs  = dict(f.attrs)

    return {
        "qpos":   qpos,
        "images": images,
        "fps":    float(attrs.get("fps", 20.0)),
        "steps":  qpos.shape[0],
    }


def prepare_jp(qpos_row) -> list:
    """Take first 6 motion joints. No unit conversion needed (SimToSI=1.0)."""
    return qpos_row[:6].tolist()


def prepare_jaw(qpos_row) -> float:
    return float(qpos_row[6]) if qpos_row.shape[0] > 6 else 0.0


# ============================================================
# Build visualization frame
# ============================================================

def draw_panel(rec_img, live_img, rec_qpos, live_jp, step, total):
    H, W    = 240, 320
    BAR_H   = 120
    PANEL_W = W * 2
    PANEL_H = H + BAR_H

    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    # Images
    canvas[:H, :W] = cv2.resize(rec_img,  (W, H))
    canvas[:H, W:] = cv2.resize(live_img, (W, H))

    # Titles
    cv2.putText(canvas, f"RECORDED  step={step}/{total}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)
    cv2.putText(canvas, "LIVE",
                (W+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)

    # Joint values
    rec_jp  = rec_qpos[:6]
    rec_jaw = float(rec_qpos[6]) if len(rec_qpos) > 6 else 0.0

    for i, name in enumerate(JOINT_NAMES[:6]):
        y = 40 + i * 16
        cv2.putText(canvas, f"{name}:{rec_jp[i]:+.3f}",
                    (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,255,200), 1)
        cv2.putText(canvas, f"{name}:{live_jp[i]:+.3f}",
                    (W+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,255), 1)

    cv2.putText(canvas, f"jaw:{rec_jaw:+.3f}",
                (5, 40 + 6*16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,255,200), 1)

    # Error bar chart
    bar_y = H
    cv2.rectangle(canvas, (0, bar_y), (PANEL_W, PANEL_H), (30,30,30), -1)
    cv2.putText(canvas, "ERROR (live - recorded)",
                (5, bar_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    errors  = np.array(live_jp) - np.array(rec_jp)
    max_err = 0.3
    bar_w   = (PANEL_W - 20) // 6

    for i, (name, err) in enumerate(zip(JOINT_NAMES[:6], errors)):
        x0       = 10 + i * bar_w
        cx       = x0 + bar_w // 2
        baseline = bar_y + BAR_H - 20
        bar_len  = min(int(abs(err) / max_err * (BAR_H - 40)), BAR_H - 40)
        color    = (0, 80, 255) if err > 0 else (255, 80, 0)

        if err > 0:
            cv2.rectangle(canvas, (x0+2, baseline-bar_len),
                          (x0+bar_w-2, baseline), color, -1)
        else:
            cv2.rectangle(canvas, (x0+2, baseline),
                          (x0+bar_w-2, baseline+bar_len), color, -1)

        cv2.line(canvas, (x0, baseline), (x0+bar_w, baseline), (150,150,150), 1)
        cv2.putText(canvas, name,
                    (cx-15, bar_y+BAR_H-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200,200,200), 1)
        cv2.putText(canvas, f"{err:+.2f}",
                    (x0, baseline-bar_len-3 if err > 0 else baseline+bar_len+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,100), 1)

    cv2.line(canvas, (W, 0), (W, H), (150,150,150), 1)
    cv2.line(canvas, (0, H), (PANEL_W, H), (150,150,150), 1)

    return canvas


# ============================================================
# Main
# ============================================================

def run(data, fps, psm_name):

    try:
        from surgical_robotics_challenge.simulation_manager import SimulationManager
        from surgical_robotics_challenge.psm_arm import PSM
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        sys.exit(1)

    steps = data["steps"]
    dt    = 1.0 / fps

    # ROS2 node for live camera feed
    rclpy.init()

    class ImgNode(Node):
        def __init__(self):
            super().__init__('replay_vis')
            self.bridge = CvBridge()
            self.latest = None
            self.create_subscription(
                Image, '/ambf/env/cameras/cameraL/ImageData', self._cb, 10)
        def _cb(self, msg):
            self.latest = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    img_node = ImgNode()

    # Connect to simulator
    print("Connecting to simulator...", flush=True)
    sim = SimulationManager('replay_vis_client')
    time.sleep(0.3)

    psm = PSM(sim, psm_name, add_joint_errors=False)
    time.sleep(0.3)

    if not psm.is_present():
        print(f"ERROR: {psm_name} not found")
        rclpy.shutdown()
        sys.exit(1)

    print(f"{psm_name} ready")

    # Warmup: continuously send commands to keep Watch Dog alive
    current_jp  = list(psm.measured_jp() or [0.0]*6)
    current_jaw = 0.0
    print("Warmup 1s...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 1.0:
        psm.servo_jp(current_jp)
        psm.set_jaw_angle(current_jaw)
        rclpy.spin_once(img_node, timeout_sec=0.01)
        time.sleep(1.0 / WATCHDOG_HZ)

    # Smooth move to start pose
    target_jp  = prepare_jp(data["qpos"][0])
    target_jaw = prepare_jaw(data["qpos"][0])
    print("Moving to start pose...", flush=True)
    t_arr = np.array(target_jp)
    c_arr = np.array(current_jp)
    for i in range(101):
        t_s    = (i/100)**2 * (3 - 2*(i/100))
        interp = (c_arr + t_s * (t_arr - c_arr)).tolist()
        psm.servo_jp(interp)
        psm.set_jaw_angle(target_jaw)
        rclpy.spin_once(img_node, timeout_sec=0.005)
        time.sleep(1.0 / WATCHDOG_HZ)
    # Smooth move done - enter main loop immediately, no print/sleep here

    # Main replay loop
    cv2.namedWindow("Replay Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Replay Monitor", 640, 360)

    wall_start = time.time()
    paused     = False

    for i in range(steps):

        # Step 0: send command first, then print to avoid Watch Dog timeout
        if i == 0:
            jp  = prepare_jp(data["qpos"][0])
            jaw = prepare_jaw(data["qpos"][0])
            psm.servo_jp(jp)
            psm.set_jaw_angle(jaw)
            print(f"\nStart pose reached - replaying {steps} steps @ {fps:.1f} Hz ...\n",
                  flush=True)
            wall_start = time.time()
            continue

        target_t = wall_start + i * dt
        now      = time.time()
        if target_t > now:
            time.sleep(target_t - now)

        jp  = prepare_jp(data["qpos"][i])
        jaw = prepare_jaw(data["qpos"][i])

        psm.servo_jp(jp)
        psm.set_jaw_angle(jaw)
        rclpy.spin_once(img_node, timeout_sec=0.002)

        live_jp  = list(psm.measured_jp() or jp)
        live_img = img_node.latest
        if live_img is None:
            live_img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(live_img, "No image", (100, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)

        frame = draw_panel(
            rec_img  = data["images"][i],
            live_img = live_img,
            rec_qpos = data["qpos"][i],
            live_jp  = live_jp,
            step     = i,
            total    = steps,
        )

        cv2.imshow("Replay Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("User quit")
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")

        while paused:
            psm.servo_jp(jp)
            psm.set_jaw_angle(jaw)
            rclpy.spin_once(img_node, timeout_sec=0.01)
            time.sleep(0.05)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                paused = False
            elif key == ord('q'):
                paused = False

        if (i + 1) % max(1, steps // 10) == 0:
            pct = (i + 1) / steps * 100
            print(f"  {pct:5.1f}%  step {i+1}/{steps}", flush=True)

    cv2.destroyAllWindows()
    print(f"\nReplay complete")
    rclpy.shutdown()


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Replay HDF5 episode with live vs recorded visualization"
    )
    p.add_argument("--episode", required=True, help="Path to .hdf5 file")
    p.add_argument("--psm",     default="psm1")
    p.add_argument("--fps",     type=float, default=None,
                   help="Replay FPS (default: read from file)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.episode):
        print(f"ERROR: file not found: {args.episode}")
        sys.exit(1)

    print(f"Loading: {args.episode}")
    data = load_episode(args.episode)
    fps  = args.fps or data["fps"]

    print(f"  steps: {data['steps']}  fps: {fps:.1f}")

    run(data, fps, args.psm)


if __name__ == "__main__":
    main()