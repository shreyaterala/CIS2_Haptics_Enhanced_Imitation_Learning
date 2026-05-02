#!/usr/bin/env python3
# run_dvrk_act.py
#
# Run an ACT policy on a real dVRK PSM over ROS2.
#
# NOTE: This variant does NOT do any start-pose move. After warmup, ACT
# takes control immediately from wherever the PSM currently is. Position
# the arm manually (or via teleop) before launching this script.
#
# Data semantics (must match record_dvrk_episodes.py):
#   State  (qpos)  : /{PSM}/measured_js      (6 arm joint positions)
#                    /{PSM}/jaw/measured_js  (1 jaw joint position)
#                    qpos   = [arm | jaw]
#   Action         : published to
#                    /{PSM}/servo_jp         (6 arm joint positions)
#                    /{PSM}/jaw/servo_jp     (1 jaw joint position)
#                    action = [cmd_arm | cmd_jaw]
#   Cameras (default topics, match recorder):
#     left     -> /console1/left/image_raw
#     right    -> /console1/right/image_raw
#     wrist_1  -> /wrist/cam1/image_raw
#     wrist_2  -> /wrist/cam2/image_raw
#
# Usage:
#   python run_dvrk_act.py \
#       --ckpt_dir ./checkpoints/dvrk_task \
#       --task_name dvrk_task \
#       --psm PSM2 \
#       --state_dim 7 \
#       --chunk_size 50 \
#       --fps 10
#
# Prerequisites:
#   1. dvrk_system is running.
#   2. Camera stacks are running (console + wrist).
#   3. Target PSM is powered, enabled, homed, and already at a sensible
#      starting configuration (move it there via teleop or manually).
#   4. MTM teleoperation is OFF for this PSM.

import os
import time
import pickle
import argparse
from collections import deque

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

from policy import ACTPolicy, CNNMLPPolicy


# ============================================================
# Constants
# ============================================================

WATCHDOG_SAFE_HZ = 30.0


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
# Combined dVRK I/O node (state + images + command publishing)
# ============================================================

class DVRKRunner(Node):
    def __init__(self,
                 psm_name,
                 camera_topic_map,
                 arm_joints=6,
                 jaw_joints=1,
                 image_encoding="bgr8"):
        super().__init__("dvrk_act_runner")

        self.bridge         = CvBridge()
        self.arm_joints     = int(arm_joints)
        self.jaw_joints     = int(jaw_joints)
        self.image_encoding = image_encoding

        self.arm_meas_topic = f"/{psm_name}/measured_js"
        self.jaw_meas_topic = f"/{psm_name}/jaw/measured_js"
        self.arm_cmd_topic  = f"/{psm_name}/servo_jp"
        self.jaw_cmd_topic  = f"/{psm_name}/jaw/servo_jp"

        self.arm_pos, self.arm_vel = None, None
        self.jaw_pos, self.jaw_vel = None, None
        self.arm_time, self.jaw_time = None, None

        self.create_subscription(JointState, self.arm_meas_topic, self._arm_cb, 50)
        self.create_subscription(JointState, self.jaw_meas_topic, self._jaw_cb, 50)
        self.get_logger().info(f"Subscribed: {self.arm_meas_topic}")
        self.get_logger().info(f"Subscribed: {self.jaw_meas_topic}")

        self.arm_pub = self.create_publisher(JointState, self.arm_cmd_topic, 10)
        self.jaw_pub = self.create_publisher(JointState, self.jaw_cmd_topic, 10)
        self.get_logger().info(f"Publishing: {self.arm_cmd_topic}")
        self.get_logger().info(f"Publishing: {self.jaw_cmd_topic}")

        self.images    = {cam: None for cam in camera_topic_map}
        self.img_times = {cam: None for cam in camera_topic_map}
        for cam, topic in camera_topic_map.items():
            self.create_subscription(Image, topic, self._make_img_cb(cam), 10)
            self.get_logger().info(f"Subscribed: {topic}  ({cam})")

    def _arm_cb(self, msg):
        pos = np.asarray(msg.position, dtype=np.float32)
        if len(pos) < self.arm_joints:
            self.get_logger().warn(
                f"arm measured_js has {len(pos)} positions, need {self.arm_joints}")
            return
        vel = np.asarray(msg.velocity, dtype=np.float32) if msg.velocity else \
              np.zeros(self.arm_joints, dtype=np.float32)
        if len(vel) < self.arm_joints:
            vel = np.pad(vel, (0, self.arm_joints - len(vel)))
        self.arm_pos  = pos[:self.arm_joints].copy()
        self.arm_vel  = vel[:self.arm_joints].copy()
        self.arm_time = time.time()

    def _jaw_cb(self, msg):
        pos = np.asarray(msg.position, dtype=np.float32)
        if len(pos) < self.jaw_joints:
            self.get_logger().warn(
                f"jaw measured_js has {len(pos)} positions, need {self.jaw_joints}")
            return
        vel = np.asarray(msg.velocity, dtype=np.float32) if msg.velocity else \
              np.zeros(self.jaw_joints, dtype=np.float32)
        if len(vel) < self.jaw_joints:
            vel = np.pad(vel, (0, self.jaw_joints - len(vel)))
        self.jaw_pos  = pos[:self.jaw_joints].copy()
        self.jaw_vel  = vel[:self.jaw_joints].copy()
        self.jaw_time = time.time()

    def _make_img_cb(self, cam):
        def cb(msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, self.image_encoding)
                self.images[cam]    = img
                self.img_times[cam] = time.time()
            except Exception as e:
                self.get_logger().error(f"Image convert failed [{cam}]: {e}")
        return cb

    def measured_jp(self):
        if self.arm_pos is None or self.jaw_pos is None:
            return None, None
        return self.arm_pos.copy(), self.jaw_pos.copy()

    def state_age(self):
        if self.arm_time is None or self.jaw_time is None:
            return float("inf")
        return time.time() - min(self.arm_time, self.jaw_time)

    def servo_jp(self, jp):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(v) for v in jp[:self.arm_joints]]
        self.arm_pub.publish(msg)

    def set_jaw_angle(self, jaw):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if np.isscalar(jaw):
            msg.position = [float(jaw)]
        else:
            msg.position = [float(v) for v in list(jaw)[:self.jaw_joints]]
        self.jaw_pub.publish(msg)


# ============================================================
# Motion helpers
# ============================================================

def hold_position(node, jp, jaw, duration):
    """Stream constant commands to satisfy dVRK watchdog."""
    dt = 1.0 / WATCHDOG_SAFE_HZ
    t0 = time.time()
    while time.time() - t0 < duration:
        node.servo_jp(jp)
        node.set_jaw_angle(jaw)
        rclpy.spin_once(node, timeout_sec=0.001)
        time.sleep(dt)


# ============================================================
# Main runner
# ============================================================

def run(args):

    # -- Load stats + training config --------------------------------------
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
    print(f"PSM          : {args.psm}")
    print(f"state_dim    : {state_dim}   (arm={args.arm_joints} + jaw={args.jaw_joints})")
    print(f"camera_names : {camera_names}")
    print(f"num_queries  : {num_queries}  effective_chunk: {effective_chunk}")
    print(f"temporal_agg : {temporal_agg}  fps: {args.fps}")
    print(f"resize       : {args.resize_w}x{args.resize_h}")

    if args.arm_joints + args.jaw_joints != state_dim:
        print(f"[WARN] arm_joints({args.arm_joints}) + jaw_joints({args.jaw_joints}) "
              f"!= state_dim({state_dim}). qpos assembly may not match training.")

    # -- Load policy -------------------------------------------------------
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    policy = make_policy(policy_class, policy_config, ckpt_path, device)

    # -- Normalization stats ----------------------------------------------
    qpos_mean   = np.asarray(stats["qpos_mean"],   dtype=np.float32).reshape(-1)
    qpos_std    = np.asarray(stats["qpos_std"],    dtype=np.float32).reshape(-1)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32).reshape(-1)
    action_std  = np.asarray(stats["action_std"],  dtype=np.float32).reshape(-1)

    if len(qpos_mean) != state_dim:
        print(f"[WARN] state_dim={state_dim} but qpos_mean len={len(qpos_mean)}. "
              f"Using stats dimension.")
        state_dim = len(qpos_mean)

    # -- Camera topic map --------------------------------------------------
    default_topics = {
        "left":    args.left_image_topic,
        "right":   args.right_image_topic,
        "wrist_1": args.wrist1_image_topic,
        "wrist_2": args.wrist2_image_topic,
    }
    camera_topic_map = {}
    for cam in camera_names:
        if cam not in default_topics or not default_topics[cam]:
            raise ValueError(
                f"No topic for camera '{cam}'. "
                f"Recognized: {list(default_topics.keys())}"
            )
        camera_topic_map[cam] = default_topics[cam]

    # -- Start ROS2 --------------------------------------------------------
    if not rclpy.ok():
        rclpy.init()

    node = DVRKRunner(
        psm_name         = args.psm,
        camera_topic_map = camera_topic_map,
        arm_joints       = args.arm_joints,
        jaw_joints       = args.jaw_joints,
        image_encoding   = args.image_encoding,
    )

    # -- Wait for first state + first image of every camera --------------
    print(f"\nWaiting for state + cameras (max {args.state_timeout:.1f}s)...",
          flush=True)
    t0 = time.time()
    ready = False
    while time.time() - t0 < args.state_timeout:
        rclpy.spin_once(node, timeout_sec=0.01)
        have_state  = (node.arm_pos is not None and node.jaw_pos is not None)
        have_images = all(node.images[c] is not None for c in camera_names)
        if have_state and have_images:
            ready = True
            break
    if not ready:
        missing = []
        if node.arm_pos is None: missing.append(node.arm_meas_topic)
        if node.jaw_pos is None: missing.append(node.jaw_meas_topic)
        missing += [camera_topic_map[c] for c in camera_names
                    if node.images[c] is None]
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError(
            f"Timed out waiting for incoming data. Missing topics:\n"
            + "\n".join(f"  {t}" for t in missing)
            + "\nCheck that dvrk_system, console cameras, and wrist cameras "
              "are all publishing.")
    print("First state + camera frames received.")

    # -- Pick up wherever the PSM currently is ---------------------------
    arm_pos, jaw_pos = node.measured_jp()
    current_jp  = arm_pos.tolist()
    current_jaw = float(jaw_pos[0]) if len(jaw_pos) >= 1 else 0.0
    print(f"Current measured pose (ACT will start from here):")
    print(f"  jp : {[round(v,5) for v in current_jp]}")
    print(f"  jaw: {current_jaw:.5f}")

    # -- Resize setup ------------------------------------------------------
    resize = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize = (int(args.resize_w), int(args.resize_h))

    # -- Warmup: stream current measured pose -----------------------------
    print(f"\nWarmup {args.warmup_sec:.1f}s (holding current pose)...", flush=True)
    hold_position(node, current_jp, current_jaw, args.warmup_sec)
    print("Warmup done")

    # -- Control loop ------------------------------------------------------
    aggregator       = ChunkAggregator(decay_k=args.temporal_decay)
    last_chunk       = None
    last_chunk_start = 0
    step_idx         = 0
    dt               = 1.0 / args.fps

    node.servo_jp(current_jp)
    node.set_jaw_angle(current_jaw)
    print(f"\nRunning ACT @ {args.fps:.1f} Hz. Ctrl+C to stop.\n", flush=True)

    try:
        while True:
            t_step_start = time.time()

            rclpy.spin_once(node, timeout_sec=0.002)

            if node.state_age() > args.max_state_age:
                if args.debug:
                    print(f"[step {step_idx}] stale state, holding")
                node.servo_jp(current_jp)
                node.set_jaw_angle(current_jaw)
                time.sleep(0.01)
                continue

            images_ok = True
            for cam in camera_names:
                t_img = node.img_times.get(cam)
                if t_img is None or (time.time() - t_img) > args.max_image_age:
                    if args.debug:
                        print(f"[step {step_idx}] stale/missing: {cam}")
                    images_ok = False
                    break
            if not images_ok:
                node.servo_jp(current_jp)
                node.set_jaw_angle(current_jaw)
                time.sleep(0.01)
                continue

            img_list = []
            for cam in camera_names:
                img = node.images[cam].copy()
                if resize is not None:
                    img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img_list.append(img)
            image_np = np.stack(img_list, axis=0)
            image_t  = torch.from_numpy(image_np).float().unsqueeze(0).to(device)

            arm_pos, jaw_pos = node.measured_jp()
            if arm_pos is None or jaw_pos is None:
                node.servo_jp(current_jp)
                node.set_jaw_angle(current_jaw)
                time.sleep(0.01)
                continue

            qpos_full = np.concatenate([arm_pos, jaw_pos], axis=0).astype(np.float32)
            qpos_np   = qpos_full[:state_dim]
            qpos_np   = (qpos_np - qpos_mean) / qpos_std
            qpos_t    = torch.from_numpy(qpos_np).float().unsqueeze(0).to(device)

            with torch.inference_mode():
                if policy_class == "ACT":
                    if temporal_agg:
                        all_actions = policy(qpos_t, image_t)
                        all_actions = all_actions.squeeze(0).cpu().numpy()
                        aggregator.add_chunk(step_idx, all_actions)
                        raw_action = aggregator.get_action(step_idx)
                        if raw_action is None:
                            node.servo_jp(current_jp)
                            node.set_jaw_angle(current_jaw)
                            time.sleep(0.01)
                            continue
                    else:
                        if step_idx % query_frequency == 0:
                            all_actions      = policy(qpos_t, image_t)
                            last_chunk       = all_actions.squeeze(0).cpu().numpy()
                            last_chunk_start = step_idx

                        if last_chunk is None:
                            node.servo_jp(current_jp)
                            node.set_jaw_angle(current_jaw)
                            time.sleep(0.01)
                            continue

                        offset = step_idx - last_chunk_start
                        if offset >= len(last_chunk):
                            node.servo_jp(current_jp)
                            node.set_jaw_angle(current_jaw)
                            time.sleep(0.01)
                            continue

                        raw_action = last_chunk[offset]

                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos_t, image_t).squeeze(0).cpu().numpy()
                else:
                    raise NotImplementedError

            action_np = raw_action * action_std + action_mean
            if args.clip_actions:
                action_np = np.clip(action_np,
                                    -args.action_clip_value,
                                    args.action_clip_value)

            jp_cmd  = action_np[:args.arm_joints].tolist()
            jaw_cmd = float(action_np[args.arm_joints]) \
                      if len(action_np) > args.arm_joints else 0.0

            if args.max_delta_jp > 0.0:
                cur   = np.array(current_jp, dtype=np.float32)
                cmd   = np.array(jp_cmd,     dtype=np.float32)
                delta = cmd - cur
                norm  = float(np.linalg.norm(delta))
                if norm > args.max_delta_jp:
                    cmd    = cur + delta * (args.max_delta_jp / norm)
                    jp_cmd = cmd.tolist()
                    if args.debug:
                        print(f"[step {step_idx}] Δjp clipped "
                              f"{norm:.4f} -> {args.max_delta_jp:.4f}")

            node.servo_jp(jp_cmd)
            node.set_jaw_angle(jaw_cmd)

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
        try:
            hold_position(node, current_jp, current_jaw, duration=0.25)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        print("Shutdown complete.")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Run an ACT policy on a real dVRK PSM (ROS2). "
                    "No start-pose move: ACT picks up from the current "
                    "measured pose after warmup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--ckpt_dir",  required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")

    p.add_argument("--policy_class",    default="ACT", choices=["ACT", "CNNMLP"])
    p.add_argument("--state_dim",       type=int,   default=7)
    p.add_argument("--task_name",       default=None,
                   help="Task name from constants.py. Sets camera_names automatically.")
    p.add_argument("--camera_names",    default="left,wrist_1",
                   help="Used if --task_name is not given. "
                        "Recognized names: left, right, wrist_1, wrist_2.")
    p.add_argument("--chunk_size",      type=int,   default=50)
    p.add_argument("--hidden_dim",      type=int,   default=512)
    p.add_argument("--dim_feedforward", type=int,   default=3200)
    p.add_argument("--kl_weight",       type=float, default=10.0)
    p.add_argument("--lr",              type=float, default=1e-5)

    p.add_argument("--psm",        default="PSM2", choices=["PSM1", "PSM2"])
    p.add_argument("--arm_joints", type=int,   default=6)
    p.add_argument("--jaw_joints", type=int,   default=1)
    p.add_argument("--fps",        type=float, default=10.0)
    p.add_argument("--joint_dim",  type=int,   default=None)
    p.add_argument("--cpu",        action="store_true")
    p.add_argument("--warmup_sec",    type=float, default=2.0)
    p.add_argument("--state_timeout", type=float, default=10.0)

    p.add_argument("--left_image_topic",   default="/console1/left/image_raw")
    p.add_argument("--right_image_topic",  default="/console1/right/image_raw")
    p.add_argument("--wrist1_image_topic", default="/wrist/cam1/image_raw")
    p.add_argument("--wrist2_image_topic", default="/wrist/cam2/image_raw")
    p.add_argument("--image_encoding",     default="bgr8", choices=["bgr8", "rgb8"])

    p.add_argument("--resize_w", type=int, default=320)
    p.add_argument("--resize_h", type=int, default=240)

    p.add_argument("--temporal_agg",   action="store_true")
    p.add_argument("--temporal_decay", type=float, default=0.01)
    p.add_argument("--run_chunk_size", type=int,   default=None)

    p.add_argument("--max_image_age",     type=float, default=0.5)
    p.add_argument("--max_state_age",     type=float, default=0.2)
    p.add_argument("--clip_actions",      action="store_true")
    p.add_argument("--action_clip_value", type=float, default=10.0)
    p.add_argument("--max_delta_jp",      type=float, default=0.0,
                   help="If > 0, cap per-step ||Δjp|| (radians). 0 disables.")

    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()