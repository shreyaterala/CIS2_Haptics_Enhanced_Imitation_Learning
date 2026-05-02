#!/usr/bin/env python3
# run_ambf_act.py
#
# Run ACT policy on AMBF PSM using psm_arm.py servo_jp() + set_jaw_angle().
# Same control path as MTM teleoperation - cannot be overridden by ROS topics.
#
# Usage:
#   python run_ambf_act.py \
#       --ckpt_dir ./checkpoints/ambf_move \
#       --task_name ambf_task_move \
#       --state_dim 7 \
#       --chunk_size 50 \
#       --start_pose ./start_pose.json \
#       --fps 20

import os
import time
import json
import pickle
import argparse
from collections import deque

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from policy import ACTPolicy, CNNMLPPolicy


# ============================================================
# Constants
# ============================================================

WATCHDOG_SAFE_HZ = 20.0


# ============================================================
# Policy helpers
# ============================================================

def make_policy(policy_class, policy_config, ckpt_path, device):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError(f"Unsupported policy_class: {policy_class}")

    state_dict  = torch.load(ckpt_path, map_location=device)
    load_status = policy.load_state_dict(state_dict)
    print(load_status)
    policy.to(device)
    policy.eval()
    return policy


def build_policy_config_from_args(args, camera_names, state_dim):
    lr_backbone = 1e-5
    backbone    = "resnet18"
    if args.policy_class == "ACT":
        return {
            "lr":              args.lr,
            "num_queries":     args.chunk_size,
            "kl_weight":       args.kl_weight,
            "hidden_dim":      args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone":     lr_backbone,
            "backbone":        backbone,
            "enc_layers":      4,
            "dec_layers":      7,
            "nheads":          8,
            "camera_names":    camera_names,
            "state_dim":       state_dim,
        }
    elif args.policy_class == "CNNMLP":
        return {
            "lr":           args.lr,
            "lr_backbone":  lr_backbone,
            "backbone":     backbone,
            "num_queries":  1,
            "camera_names": camera_names,
            "state_dim":    state_dim,
        }
    raise NotImplementedError(f"Unsupported policy_class: {args.policy_class}")


# ============================================================
# Temporal aggregation
# ============================================================

class ChunkAggregator:
    def __init__(self, decay_k=0.01):
        self.decay_k     = decay_k
        self.predictions = deque()

    def reset(self):
        self.predictions.clear()

    def add_chunk(self, start_step, chunk_np):
        self.predictions.append((int(start_step),
                                  np.asarray(chunk_np, dtype=np.float32)))

    def get_action(self, current_step):
        valid = [chunk[current_step - start]
                 for start, chunk in self.predictions
                 if 0 <= current_step - start < len(chunk)]
        if not valid:
            return None
        valid   = np.stack(valid, axis=0)
        weights = np.exp(-self.decay_k * np.arange(len(valid), dtype=np.float32))
        weights = weights / (weights.sum() + 1e-8)
        return (valid * weights[:, None]).sum(axis=0).astype(np.float32)


# ============================================================
# Image subscriber node (ROS2 only for cameras)
# ============================================================

class ImageNode(Node):
    def __init__(self, camera_topic_map: dict, encoding: str = "bgr8"):
        super().__init__("act_image_subscriber")
        self.bridge    = CvBridge()
        self.encoding  = encoding
        self.images    = {cam: None for cam in camera_topic_map}
        self.img_times = {cam: None for cam in camera_topic_map}

        for cam, topic in camera_topic_map.items():
            self.create_subscription(
                Image, topic, self._make_cb(cam), 10)
            self.get_logger().info(f"Subscribed: {topic}")

    def _make_cb(self, cam):
        def cb(msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, self.encoding)
                self.images[cam]    = img
                self.img_times[cam] = time.time()
            except Exception as e:
                self.get_logger().error(f"Image convert failed [{cam}]: {e}")
        return cb


# ============================================================
# PSM helpers
# ============================================================

def hold_position(psm, jp: list, jaw: float, duration: float):
    """Continuously send commands to prevent Watch Dog timeout."""
    dt = 1.0 / WATCHDOG_SAFE_HZ
    t0 = time.time()
    while time.time() - t0 < duration:
        psm.servo_jp(jp)
        psm.set_jaw_angle(jaw)
        time.sleep(dt)


def smooth_goto(psm, target_jp: list, target_jaw: float,
                current_jp: list, steps: int = 100):
    """
    Ease in/out interpolation to target pose.
    Continuously sends commands throughout to prevent Watch Dog timeout.
    Returns immediately - caller must not sleep after this.
    """
    t_arr = np.array(target_jp,  dtype=np.float32)
    c_arr = np.array(current_jp, dtype=np.float32)
    dt    = 1.0 / WATCHDOG_SAFE_HZ

    print(f"Moving to start pose ({steps} steps)...", flush=True)
    for i in range(steps + 1):
        t        = i / steps
        t_smooth = t * t * (3.0 - 2.0 * t)
        interp   = (c_arr + t_smooth * (t_arr - c_arr)).tolist()
        psm.servo_jp(interp)
        psm.set_jaw_angle(target_jaw)
        time.sleep(dt)
    # No print/sleep here - return immediately


def load_start_pose(start_pose_path: str):
    """Load start pose from set_start_pose.py saved JSON file."""
    if not os.path.isfile(start_pose_path):
        raise FileNotFoundError(
            f"Start pose file not found: {start_pose_path}\n"
            f"Create one with: python set_start_pose.py --save"
        )
    with open(start_pose_path) as f:
        pose = json.load(f)
    jp  = pose["servo_jp"]
    jaw = pose["jaw"]
    print(f"Start pose from: {start_pose_path} (saved at {pose.get('saved_at','unknown')})")
    print(f"  jp : {[round(v,5) for v in jp]}")
    print(f"  jaw: {jaw:.5f}")
    return jp, jaw


# ============================================================
# Main runner
# ============================================================

def run(args):

    # ── Load stats and config ─────────────────────────────────
    stats_path  = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    config_path = os.path.join(args.ckpt_dir, "train_config.pkl")

    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"Missing dataset stats: {stats_path}")

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    if os.path.isfile(config_path):
        print(f"Loading train_config from: {config_path}")
        with open(config_path, "rb") as f:
            train_config = pickle.load(f)
        policy_class  = train_config["policy_class"]
        policy_config = train_config["policy_config"]
        camera_names  = list(train_config["camera_names"])
        state_dim     = int(train_config["state_dim"])
        temporal_agg  = bool(args.temporal_agg or
                             train_config.get("temporal_agg", False))
    else:
        print("train_config.pkl not found - rebuilding from CLI args.")

        if args.task_name is not None:
            from constants import SIM_TASK_CONFIGS
            if args.task_name not in SIM_TASK_CONFIGS:
                raise ValueError(
                    f"task_name '{args.task_name}' not found in constants.py. "
                    f"Available: {list(SIM_TASK_CONFIGS.keys())}"
                )
            task_cfg     = SIM_TASK_CONFIGS[args.task_name]
            camera_names = list(task_cfg["camera_names"])
            print(f"camera_names from constants [{args.task_name}]: {camera_names}")
            if "resize_w" in task_cfg and args.resize_w == 320:
                args.resize_w = int(task_cfg["resize_w"])
            if "resize_h" in task_cfg and args.resize_h == 240:
                args.resize_h = int(task_cfg["resize_h"])
        else:
            camera_names = [c.strip() for c in args.camera_names.split(",")]

        state_dim     = int(args.state_dim)
        temporal_agg  = bool(args.temporal_agg)
        policy_class  = args.policy_class
        policy_config = build_policy_config_from_args(args, camera_names, state_dim)

    if args.joint_dim is not None:
        state_dim = int(args.joint_dim)

    num_queries = int(policy_config.get("num_queries", 1))

    if args.run_chunk_size is not None:
        if args.run_chunk_size > num_queries:
            raise ValueError(
                f"--run_chunk_size ({args.run_chunk_size}) > "
                f"training chunk_size ({num_queries})"
            )
        effective_chunk = int(args.run_chunk_size)
    else:
        effective_chunk = num_queries

    query_frequency = 1 if temporal_agg else effective_chunk

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    print(f"Device       : {device}")
    print(f"state_dim    : {state_dim}")
    print(f"camera_names : {camera_names}")
    print(f"num_queries  : {num_queries}  effective_chunk: {effective_chunk}")
    print(f"temporal_agg : {temporal_agg}  fps: {args.fps}")
    print(f"resize       : {args.resize_w}x{args.resize_h}")

    # ── Load policy ───────────────────────────────────────────
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    policy = make_policy(policy_class, policy_config, ckpt_path, device)

    # ── Normalization ─────────────────────────────────────────
    qpos_mean   = np.asarray(stats["qpos_mean"],   dtype=np.float32).reshape(-1)
    qpos_std    = np.asarray(stats["qpos_std"],    dtype=np.float32).reshape(-1)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32).reshape(-1)
    action_std  = np.asarray(stats["action_std"],  dtype=np.float32).reshape(-1)

    if len(qpos_mean) != state_dim:
        print(f"Warning: state_dim={state_dim} but qpos_mean len={len(qpos_mean)}. "
              f"Using stats dimension.")
        state_dim = len(qpos_mean)

    # ── Resolve start pose ────────────────────────────────────
    # Priority: --start_pose file > qpos_mean fallback
    if args.start_pose is not None:
        start_jp, start_jaw = load_start_pose(args.start_pose)
    else:
        start_jp  = qpos_mean[:6].tolist()
        start_jaw = float(qpos_mean[6]) if len(qpos_mean) > 6 else 0.0
        print(f"Start pose from qpos_mean (no --start_pose given)")
        print(f"  jp : {[round(v,5) for v in start_jp]}")
        print(f"  jaw: {start_jaw:.5f}")

    # ── Camera topic map ──────────────────────────────────────
    default_topics = {
        "left":  args.left_image_topic,
        "right": args.right_image_topic,
        "wrist": args.wrist_image_topic,
    }
    camera_topic_map = {}
    for cam in camera_names:
        if cam not in default_topics or not default_topics[cam]:
            raise ValueError(f"No topic for camera '{cam}'")
        camera_topic_map[cam] = default_topics[cam]

    # ── Connect to AMBF via psm_arm.py ───────────────────────
    try:
        from surgical_robotics_challenge.simulation_manager import SimulationManager
        from surgical_robotics_challenge.psm_arm import PSM
    except ImportError:
        raise ImportError(
            "surgical_robotics_challenge not found.\n"
            "Install: python -m pip install -e "
            "~/surgical_robotics_challenge/scripts/"
        )

    print("\nConnecting to simulator...", flush=True)
    sim = SimulationManager("act_runner")
    time.sleep(0.5)

    psm = PSM(sim, args.psm, add_joint_errors=False)
    time.sleep(0.5)

    if not psm.is_present():
        raise RuntimeError(f"PSM '{args.psm}' not found in simulator")

    print(f"PSM '{args.psm}' ready")

    current_jp  = list(psm.measured_jp() or [0.0] * 6)
    current_jaw = 0.0

    # ── Start ROS2 for camera images ─────────────────────────
    if not rclpy.ok():
        rclpy.init()
    img_node = ImageNode(camera_topic_map, args.image_encoding)

    # ── Resize settings ───────────────────────────────────────
    resize = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize = (int(args.resize_w), int(args.resize_h))

    # ── Warmup: spin images + hold PSM position ───────────────
    print(f"\nWarmup {args.warmup_sec:.1f}s...", flush=True)
    dt_wd = 1.0 / WATCHDOG_SAFE_HZ
    t0    = time.time()
    while time.time() - t0 < args.warmup_sec:
        rclpy.spin_once(img_node, timeout_sec=0.002)
        psm.servo_jp(current_jp)
        psm.set_jaw_angle(current_jaw)
        time.sleep(dt_wd)
    print("Warmup done")

    # ── Smooth move to start pose ─────────────────────────────
    smooth_goto(psm, start_jp, start_jaw, current_jp, steps=args.goto_steps)
    current_jp  = start_jp
    current_jaw = start_jaw
    # smooth_goto returns immediately - enter control loop with no delay

    # ── Control loop ──────────────────────────────────────────
    aggregator       = ChunkAggregator(decay_k=args.temporal_decay)
    last_chunk       = None
    last_chunk_start = 0
    step_idx         = 0
    dt               = 1.0 / args.fps

    # Send step 0 before printing to prevent Watch Dog timeout
    psm.servo_jp(current_jp)
    psm.set_jaw_angle(current_jaw)
    print(f"\nStart pose reached - running ACT @ {args.fps:.1f} Hz. Ctrl+C to stop.\n",
          flush=True)

    try:
        while True:
            t_step_start = time.time()

            # ── Spin cameras ──────────────────────────────────
            rclpy.spin_once(img_node, timeout_sec=0.002)

            images_ok = True
            for cam in camera_names:
                t_img = img_node.img_times.get(cam)
                if t_img is None or (time.time() - t_img) > args.max_image_age:
                    if args.debug:
                        print(f"[step {step_idx}] stale/missing: {cam}")
                    images_ok = False
                    break

            if not images_ok:
                psm.servo_jp(current_jp)
                psm.set_jaw_angle(current_jaw)
                time.sleep(0.01)
                continue

            # ── Build image tensor ────────────────────────────
            img_list = []
            for cam in camera_names:
                img = img_node.images[cam].copy()
                if resize is not None:
                    img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img_list.append(img)

            image_np = np.stack(img_list, axis=0)
            image_t  = torch.from_numpy(image_np).float().unsqueeze(0).to(device)

            # ── Get PSM state ─────────────────────────────────
            jp_raw = psm.measured_jp()
            if jp_raw is None:
                if args.debug:
                    print(f"[step {step_idx}] measured_jp returned None")
                psm.servo_jp(current_jp)
                psm.set_jaw_angle(current_jaw)
                time.sleep(0.01)
                continue

            # measured_jp() returns 6 joints only
            # append current_jaw as 7th dimension to match recording format
            qpos_full = np.array(list(jp_raw) + [float(current_jaw)], dtype=np.float32)
            qpos_np   = qpos_full[:state_dim]
            qpos_np   = (qpos_np - qpos_mean) / qpos_std
            qpos_t    = torch.from_numpy(qpos_np).float().unsqueeze(0).to(device)

            # ── Policy inference ──────────────────────────────
            with torch.inference_mode():
                if policy_class == "ACT":
                    if temporal_agg:
                        all_actions = policy(qpos_t, image_t)
                        all_actions = all_actions.squeeze(0).cpu().numpy()
                        aggregator.add_chunk(step_idx, all_actions)
                        raw_action = aggregator.get_action(step_idx)
                        if raw_action is None:
                            psm.servo_jp(current_jp)
                            psm.set_jaw_angle(current_jaw)
                            time.sleep(0.01)
                            continue
                    else:
                        if step_idx % query_frequency == 0:
                            all_actions      = policy(qpos_t, image_t)
                            last_chunk       = all_actions.squeeze(0).cpu().numpy()
                            last_chunk_start = step_idx

                        if last_chunk is None:
                            psm.servo_jp(current_jp)
                            psm.set_jaw_angle(current_jaw)
                            time.sleep(0.01)
                            continue

                        offset = step_idx - last_chunk_start
                        if offset >= len(last_chunk):
                            psm.servo_jp(current_jp)
                            psm.set_jaw_angle(current_jaw)
                            time.sleep(0.01)
                            continue

                        raw_action = last_chunk[offset]

                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos_t, image_t).squeeze(0).cpu().numpy()
                else:
                    raise NotImplementedError

            # ── Denormalize and send ──────────────────────────
            action_np = raw_action * action_std + action_mean
            if args.clip_actions:
                action_np = np.clip(action_np,
                                    -args.action_clip_value,
                                    args.action_clip_value)

            jp_cmd  = action_np[:6].tolist()
            jaw_cmd = float(action_np[6]) if len(action_np) > 6 else 0.0

            psm.servo_jp(jp_cmd)
            psm.set_jaw_angle(jaw_cmd)

            current_jp  = jp_cmd
            current_jaw = jaw_cmd

            if args.debug:
                jp_str = "  ".join(f"{v:+.4f}" for v in jp_cmd)
                print(f"step={step_idx:5d} | jp=[{jp_str}]  jaw={jaw_cmd:+.4f}")

            step_idx += 1

            elapsed = time.time() - t_step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print(f"\nStopped after {step_idx} steps.")
    finally:
        img_node.destroy_node()
        rclpy.shutdown()
        del psm
        del sim
        print("Shutdown complete.")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Run ACT policy on AMBF using psm_arm.servo_jp()"
    )

    # Checkpoint
    p.add_argument("--ckpt_dir",  required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")

    # Policy config
    p.add_argument("--policy_class",    default="ACT", choices=["ACT", "CNNMLP"])
    p.add_argument("--state_dim",       type=int,   default=7)
    p.add_argument("--task_name",       default=None,
                   help="Task name from constants.py. Sets camera_names automatically.")
    p.add_argument("--camera_names",    default="left,wrist",
                   help="Ignored if --task_name is given.")
    p.add_argument("--chunk_size",      type=int,   default=50)
    p.add_argument("--hidden_dim",      type=int,   default=512)
    p.add_argument("--dim_feedforward", type=int,   default=3200)
    p.add_argument("--kl_weight",       type=float, default=10.0)
    p.add_argument("--lr",              type=float, default=1e-5)

    # Runtime
    p.add_argument("--psm",        default="psm1")
    p.add_argument("--fps",        type=float, default=20.0)
    p.add_argument("--joint_dim",  type=int,   default=None)
    p.add_argument("--cpu",        action="store_true")
    p.add_argument("--warmup_sec", type=float, default=2.0)

    # Start pose
    p.add_argument("--start_pose", default=None,
                   help="Path to start_pose.json from set_start_pose.py. "
                        "If not given, uses qpos_mean as start pose.")
    p.add_argument("--goto_steps", type=int, default=100,
                   help="Interpolation steps for smooth move to start pose (default: 100)")

    # Camera topics
    p.add_argument("--left_image_topic",
                   default="/ambf/env/cameras/cameraL/ImageData")
    p.add_argument("--right_image_topic",
                   default="/ambf/env/cameras/cameraR/ImageData")
    p.add_argument("--wrist_image_topic",
                   default="/ambf/env/cameras/Wrist1Frame/ImageData")
    p.add_argument("--image_encoding", default="bgr8", choices=["bgr8", "rgb8"])

    # Image resize (must match recording)
    p.add_argument("--resize_w", type=int, default=320)
    p.add_argument("--resize_h", type=int, default=240)

    # ACT options
    p.add_argument("--temporal_agg",      action="store_true")
    p.add_argument("--temporal_decay",    type=float, default=0.01)
    p.add_argument("--run_chunk_size",    type=int,   default=None)

    # Safety
    p.add_argument("--max_image_age",     type=float, default=0.5)
    p.add_argument("--clip_actions",      action="store_true")
    p.add_argument("--action_clip_value", type=float, default=10.0)

    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()