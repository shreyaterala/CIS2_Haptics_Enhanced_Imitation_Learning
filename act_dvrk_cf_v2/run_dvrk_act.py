#!/usr/bin/env python3
# run_dvrk_act_v2.py
#
# Run an ACT policy on a real dVRK PSM over ROS2.
#
# Backward-compatible with both legacy (no force) and new (with force) ckpts.
# CF runtime baseline (B_run) is auto-collected during warmup to compensate
# for sensor zero drift between training and inference (path-1 calibration).
#
# Usage:
#   python run_dvrk_act_v2.py --ckpt_dir ./checkpoints/X --task_name X \
#       --psm PSM2 --state_dim 7 --chunk_size 50 --fps 10
#
#   Use --no_cf_zero to disable the warmup baseline (NOT RECOMMENDED).
#
#   Add --record_force to log /measured_cf during the ACT control loop and
#   save the raw data + a force/torque plot into a timestamped folder under
#   --force_output_dir when the run ends (Ctrl+C, --max_steps, or error).
#
#   Add --restart_after_sec 15 to auto-restart the policy every 15 seconds
#   (useful for letting ACT 'try again' if the first attempt fails). Combine
#   with --start_from_episode so the robot returns to the demonstrated start
#   pose between restarts. Use --max_restarts to cap the number of retries.

import os
import sys
import glob
import time
import pickle
import argparse
import threading
from collections import deque

import cv2
import h5py
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped

import crtk

from policy import ACTPolicy, CNNMLPPolicy


WATCHDOG_SAFE_HZ = 30.0


# ============================================================
# CRTK arm wrapper
# ============================================================

class ArmCustom:
    class _Jaw:
        def __init__(self, ral, operating_state_instance):
            self._utils = crtk.utils(
                class_instance=self, ral=ral,
                operating_state_instance=operating_state_instance,
            )
            self._utils.add_move_jp()
            self._utils.add_servo_jp()
            self._utils.add_servo_jf()
            self._utils.add_measured_js()

    def __init__(self, ral):
        self._ral = ral
        self._utils = crtk.utils(self, ral)
        self._utils.add_operating_state()
        self._utils.add_servo_jp()
        self._utils.add_move_jp()
        self._utils.add_measured_js()
        self._utils.add_setpoint_js()
        self.jaw = self._Jaw(ral.create_child("/jaw"), operating_state_instance=self)

    def ral(self):
        return self._ral


def safe_check_connections(ral, timeout_sec=5.0):
    if not hasattr(ral, "check_connections"):
        return
    try:
        ral.check_connections(timeout_sec)
    except TypeError:
        try:
            ral.check_connections(timeout=timeout_sec)
        except TypeError:
            ral.check_connections()


def safe_measured_jp(arm_or_jaw):
    result = arm_or_jaw.measured_jp()
    if isinstance(result, np.ndarray):
        return np.asarray(result, dtype=np.float32)
    if isinstance(result, (tuple, list)):
        if len(result) == 0:
            raise RuntimeError("measured_jp() returned empty tuple")
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


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


def build_policy_config_from_args(args, camera_names, state_dim, cf_dim=0):
    lr_backbone = 1e-5
    backbone    = "resnet18"
    if args.policy_class == "ACT":
        return {
            "lr": args.lr, "num_queries": args.chunk_size,
            "kl_weight": args.kl_weight, "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": lr_backbone, "backbone": backbone,
            "enc_layers": 4, "dec_layers": 7, "nheads": 8,
            "camera_names": camera_names,
            "state_dim": state_dim, "cf_dim": cf_dim,
        }
    elif args.policy_class == "CNNMLP":
        return {
            "lr": args.lr, "lr_backbone": lr_backbone, "backbone": backbone,
            "num_queries": 1, "camera_names": camera_names,
            "state_dim": state_dim, "cf_dim": cf_dim,
        }
    raise NotImplementedError(f"Unsupported policy_class: {args.policy_class}")


# ============================================================
# Temporal aggregation
# ============================================================

class ChunkAggregator:
    def __init__(self, decay_k=0.01):
        self.decay_k = decay_k
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
        valid = np.stack(valid, axis=0)
        weights = np.exp(-self.decay_k * np.arange(len(valid), dtype=np.float32))
        weights = weights / (weights.sum() + 1e-8)
        return (valid * weights[:, None]).sum(axis=0).astype(np.float32)


# ============================================================
# Sensor ROS2 node
# ============================================================

class SensorNode(Node):
    def __init__(self, camera_topic_map, image_encoding="bgr8", cf_topic=None,
                 force_record_topic=None):
        super().__init__("dvrk_act_sensors")
        self.bridge = CvBridge()
        self.image_encoding = image_encoding
        self.images    = {cam: None for cam in camera_topic_map}
        self.img_times = {cam: None for cam in camera_topic_map}
        for cam, topic in camera_topic_map.items():
            self.create_subscription(Image, topic, self._make_img_cb(cam), 10)
            self.get_logger().info(f"Subscribed: {topic}  ({cam})")
        self.cf_topic  = cf_topic
        self.latest_cf = None
        self.cf_time   = None
        if cf_topic is not None:
            self.create_subscription(WrenchStamped, cf_topic, self._cf_cb, 50)
            self.get_logger().info(f"Subscribed: {cf_topic}  (measured_cf)")

        # ── Force recording (independent of ACT CF input) ──────────────
        # Records every WrenchStamped message arriving on `force_record_topic`
        # into a buffer when self._force_recording_active is True.
        # If `force_record_topic` matches `cf_topic` we piggyback on _cf_cb
        # to avoid duplicate subscriptions.
        self.force_record_topic = force_record_topic
        self.force_record_buffer = []   # list of (t_wall_sec, np.array(6,) float32)
        self._force_recording_active = False
        self._force_share_cf = False
        if force_record_topic is not None and str(force_record_topic).lower() != "none":
            if force_record_topic == cf_topic:
                # Same topic ACT already listens to -> share callback
                self._force_share_cf = True
                self.get_logger().info(
                    f"Force recording will share CF subscription on {force_record_topic}"
                )
            else:
                self.create_subscription(
                    WrenchStamped, force_record_topic, self._force_record_cb, 50
                )
                self.get_logger().info(
                    f"Subscribed: {force_record_topic}  (force recording)"
                )

    def _make_img_cb(self, cam):
        def cb(msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, self.image_encoding)
                self.images[cam] = img
                self.img_times[cam] = time.time()
            except Exception as e:
                self.get_logger().error(f"Image convert failed [{cam}]: {e}")
        return cb

    def _cf_cb(self, msg):
        try:
            now = time.time()
            f = np.array([
                msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            ], dtype=np.float32)
            self.latest_cf = f
            self.cf_time = now
            # Piggyback recording when sharing the cf topic
            if self._force_share_cf and self._force_recording_active:
                # list.append is atomic under the GIL — no lock needed
                self.force_record_buffer.append((now, f.copy()))
        except Exception as e:
            self.get_logger().error(f"CF convert failed: {e}")

    def _force_record_cb(self, msg):
        """Dedicated callback used only when force_record_topic != cf_topic."""
        if not self._force_recording_active:
            return
        try:
            f = np.array([
                msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            ], dtype=np.float32)
            self.force_record_buffer.append((time.time(), f))
        except Exception as e:
            self.get_logger().error(f"Force record convert failed: {e}")

    def start_force_recording(self):
        """Clear buffer and begin appending incoming wrench samples."""
        self.force_record_buffer = []
        self._force_recording_active = True

    def stop_force_recording(self):
        """Stop appending and return a snapshot of the buffer."""
        self._force_recording_active = False
        # Snapshot: shallow copy of the list (tuples are immutable, np arrays are
        # already owned, so this is safe even with concurrent appends being halted).
        return list(self.force_record_buffer)


def confirm(prompt, skip=False):
    if skip:
        print(f"[AUTO] {prompt}  (skipped)")
        return
    input(f"---> {prompt}")


def load_start_pose_from_episode(path, arm_joints=6, jaw_joints=1):
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.hdf5')) + glob.glob(os.path.join(path, '*.h5'))
        if not files:
            raise FileNotFoundError(f"No .hdf5 or .h5 files found in directory {path}")
    else:
        raise FileNotFoundError(path)

    all_qpos0 = []
    for f_path in files:
        try:
            with h5py.File(f_path, "r") as f:
                qpos0 = np.asarray(f["observations/qpos"][0], dtype=np.float32)
                all_qpos0.append(qpos0)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    if not all_qpos0:
        raise ValueError(f"Could not read qpos from any files in {path}")

    avg_qpos0 = np.mean(all_qpos0, axis=0)
    return avg_qpos0[:arm_joints], avg_qpos0[arm_joints:arm_joints + jaw_joints]


def save_force_recording(records, output_dir, run_name=None, source_topic=None):
    """
    Save force recording data and plots to a timestamped folder.

    Parameters
    ----------
    records : list of (timestamp_sec, np.ndarray shape (6,))
        Buffer collected during ACT inference: [Fx, Fy, Fz, Tx, Ty, Tz].
    output_dir : str
        Parent directory; a timestamped subfolder is created inside it.
    run_name : str or None
        Optional identifier (e.g. task_name) prepended to the folder name.
    source_topic : str or None
        ROS topic the data came from (recorded into the npz metadata + plot).

    Returns
    -------
    str or None : path to the created folder, or None if nothing was saved.
    """
    if not records:
        print("[force-record] No force samples collected; nothing to save.")
        print("               (Was --force_record_topic publishing? "
              "Was --record_force passed?)")
        return None

    # Lazy-import matplotlib so the dependency only matters when actually used.
    try:
        import matplotlib
        matplotlib.use("Agg")              # headless-safe backend
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("[force-record] matplotlib not installed — saving raw data only.")

    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{run_name}_" if run_name else ""
    folder = os.path.join(output_dir, f"{tag}force_{stamp}")
    os.makedirs(folder, exist_ok=True)

    times  = np.array([r[0] for r in records], dtype=np.float64)
    forces = np.stack([r[1] for r in records], axis=0).astype(np.float32)  # (N, 6)
    t_rel  = times - times[0]
    duration = float(t_rel[-1]) if len(t_rel) > 0 else 0.0
    n = len(records)
    rate_hz = (n - 1) / duration if duration > 0 else 0.0

    # ── Raw data ──────────────────────────────────────────────────────
    npz_path = os.path.join(folder, "force_data.npz")
    np.savez_compressed(
        npz_path,
        timestamps_abs=times,
        timestamps_rel=t_rel,
        wrench=forces,
        labels=np.array(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]),
        source_topic=np.array(source_topic if source_topic else ""),
        run_name=np.array(run_name if run_name else ""),
    )

    csv_path = os.path.join(folder, "force_data.csv")
    csv_data = np.concatenate([t_rel.reshape(-1, 1),
                               times.reshape(-1, 1),
                               forces], axis=1)
    np.savetxt(csv_path, csv_data, delimiter=",",
               header="t_rel_s,t_abs_s,Fx,Fy,Fz,Tx,Ty,Tz", comments="")

    # ── Plots ─────────────────────────────────────────────────────────
    if plt is not None:
        suptitle_extra = f"  ({source_topic})" if source_topic else ""

        # Per-axis force / torque
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        for i, lab in enumerate(["Fx", "Fy", "Fz"]):
            axes[0].plot(t_rel, forces[:, i], label=lab, linewidth=1.0)
        axes[0].set_ylabel("Force [N]")
        axes[0].set_title(f"End-effector wrench during ACT inference{suptitle_extra}")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        for i, lab in enumerate(["Tx", "Ty", "Tz"]):
            axes[1].plot(t_rel, forces[:, 3 + i], label=lab, linewidth=1.0)
        axes[1].set_ylabel("Torque [N·m]")
        axes[1].set_xlabel("Time [s]")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(folder, "force_plot.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        # Magnitude
        fmag = np.linalg.norm(forces[:, :3], axis=1)
        tmag = np.linalg.norm(forces[:, 3:], axis=1)
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        l1, = ax2.plot(t_rel, fmag, label="|F|", linewidth=1.3, color="tab:blue")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("|F| [N]", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.grid(True, alpha=0.3)
        ax2_t = ax2.twinx()
        l2, = ax2_t.plot(t_rel, tmag, label="|τ|", linewidth=1.0,
                         color="tab:orange", alpha=0.85)
        ax2_t.set_ylabel("|τ| [N·m]", color="tab:orange")
        ax2_t.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_title(f"Force / torque magnitude{suptitle_extra}")
        ax2.legend([l1, l2], ["|F|", "|τ|"], loc="upper right")
        fig2.tight_layout()
        mag_path = os.path.join(folder, "force_magnitude.png")
        fig2.savefig(mag_path, dpi=150)
        plt.close(fig2)

    # ── Summary ───────────────────────────────────────────────────────
    summary_path = os.path.join(folder, "summary.txt")
    fmin = forces.min(axis=0); fmax = forces.max(axis=0); fmean = forces.mean(axis=0)
    with open(summary_path, "w") as fh:
        fh.write(f"ACT force recording\n")
        fh.write(f"run_name     : {run_name}\n")
        fh.write(f"source_topic : {source_topic}\n")
        fh.write(f"saved_at     : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"samples      : {n}\n")
        fh.write(f"duration_s   : {duration:.3f}\n")
        fh.write(f"avg_rate_hz  : {rate_hz:.2f}\n\n")
        fh.write("              Fx        Fy        Fz        Tx        Ty        Tz\n")
        fh.write(f"min     : " + "  ".join(f"{v:+8.4f}" for v in fmin)  + "\n")
        fh.write(f"max     : " + "  ".join(f"{v:+8.4f}" for v in fmax)  + "\n")
        fh.write(f"mean    : " + "  ".join(f"{v:+8.4f}" for v in fmean) + "\n")

    print(f"[force-record] Saved {n} samples ({duration:.2f}s, "
          f"~{rate_hz:.1f} Hz) to:")
    print(f"               {folder}")
    print(f"               - force_data.npz")
    print(f"               - force_data.csv")
    if plt is not None:
        print(f"               - force_plot.png")
        print(f"               - force_magnitude.png")
    print(f"               - summary.txt")
    return folder


# ============================================================
# Main runner
# ============================================================

def run_act(ral, args):

    stats_path  = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    config_path = os.path.join(args.ckpt_dir, "train_config.pkl")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"Missing dataset stats: {stats_path}")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    has_cf_in_stats = ("cf_mean" in stats) and ("cf_std" in stats)
    use_cf = has_cf_in_stats and (not args.no_cf)
    if args.no_cf and has_cf_in_stats:
        print("[WARN] --no_cf set but stats contains cf_mean. ckpt load WILL FAIL "
              "if the model was trained with cf.")

    if os.path.isfile(config_path):
        print(f"Loading train_config from: {config_path}")
        with open(config_path, "rb") as f:
            train_config = pickle.load(f)
        policy_class  = train_config["policy_class"]
        policy_config = train_config["policy_config"]
        camera_names  = list(train_config["camera_names"])
        state_dim     = int(train_config["state_dim"])
        temporal_agg  = bool(args.temporal_agg or train_config.get("temporal_agg", False))
        train_cf_dim  = int(policy_config.get("cf_dim", 0))
    else:
        print("train_config.pkl not found - rebuilding from CLI args.")
        if args.task_name is not None:
            from constants import SIM_TASK_CONFIGS
            if args.task_name not in SIM_TASK_CONFIGS:
                raise ValueError(f"task_name '{args.task_name}' not found.")
            task_cfg = SIM_TASK_CONFIGS[args.task_name]
            camera_names = list(task_cfg["camera_names"])
            print(f"camera_names: {camera_names}")
            if "resize_w" in task_cfg and args.resize_w == 320:
                args.resize_w = int(task_cfg["resize_w"])
            if "resize_h" in task_cfg and args.resize_h == 240:
                args.resize_h = int(task_cfg["resize_h"])
        else:
            camera_names = [c.strip() for c in args.camera_names.split(",")]
        state_dim     = int(args.state_dim)
        temporal_agg  = bool(args.temporal_agg)
        policy_class  = args.policy_class
        train_cf_dim  = 0
        policy_config = None

    if args.joint_dim is not None:
        state_dim = int(args.joint_dim)

    if use_cf:
        cf_mean_arr = np.asarray(stats["cf_mean"], dtype=np.float32).reshape(-1)
        cf_std_arr  = np.asarray(stats["cf_std"],  dtype=np.float32).reshape(-1)
        stats_cf_dim = len(cf_mean_arr)
        if args.cf_dim is not None:
            cf_dim = int(args.cf_dim)
        elif train_cf_dim > 0:
            cf_dim = train_cf_dim
        else:
            cf_dim = stats_cf_dim
        if cf_dim != stats_cf_dim:
            print(f"[WARN] cf_dim={cf_dim} differs from stats cf_mean length ({stats_cf_dim}).")
            cf_dim = min(cf_dim, stats_cf_dim)
        cf_mean_arr = cf_mean_arr[:cf_dim]
        cf_std_arr  = cf_std_arr[:cf_dim]
    else:
        cf_dim = 0
        cf_mean_arr = None
        cf_std_arr  = None

    cf_clip_val = float(stats.get("cf_clip", 0.0)) if use_cf else 0.0
    if cf_clip_val < 0:
        cf_clip_val = 0.0

    if policy_config is None:
        policy_config = build_policy_config_from_args(args, camera_names, state_dim, cf_dim=cf_dim)
    num_queries = int(policy_config.get("num_queries", 1))

    if args.run_chunk_size is not None:
        if args.run_chunk_size > num_queries:
            raise ValueError(f"--run_chunk_size > training chunk_size")
        effective_chunk = int(args.run_chunk_size)
    else:
        effective_chunk = num_queries
    query_frequency = 1 if temporal_agg else effective_chunk

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # ── CF runtime baseline (collected during warmup) ─────────────────
    do_cf_zero = use_cf and (not args.no_cf_zero)
    cf_runtime_bias = np.zeros(max(cf_dim, 1), dtype=np.float32)

    print("=" * 60)
    print(f"Mode         : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Device       : {device}")
    print(f"PSM          : {args.psm}")
    print(f"state_dim    : {state_dim}   (arm={args.arm_joints} + jaw={args.jaw_joints})")
    if use_cf:
        clip_str = f"±{cf_clip_val}" if cf_clip_val > 0 else "disabled"
        zero_str = "ON (path-1 baseline collected during warmup)" if do_cf_zero else "OFF (--no_cf_zero)"
        print(f"CF mode      : ON   (cf_dim={cf_dim}, topic={args.measured_cf_topic}, clip={clip_str})")
        print(f"CF zeroing   : {zero_str}")
    elif args.no_cf:
        print(f"CF mode      : OFF  (forced by --no_cf)")
    else:
        print(f"CF mode      : OFF  (legacy ckpt)")
    print(f"Network in   : {state_dim + cf_dim}D state vector")
    print(f"camera_names : {camera_names}")
    print(f"num_queries  : {num_queries}  effective_chunk: {effective_chunk}")
    print(f"temporal_agg : {temporal_agg}  fps: {args.fps}")
    print(f"resize       : {args.resize_w}x{args.resize_h}")
    print(f"max_steps    : {'unlimited' if args.max_steps <= 0 else args.max_steps}")
    if args.restart_after_sec > 0:
        cap = "∞" if args.max_restarts < 0 else str(args.max_restarts)
        print(f"auto-restart : every {args.restart_after_sec:.1f}s  "
              f"(max restarts: {cap})")
    else:
        print(f"auto-restart : OFF")
    print("=" * 60)

    if args.arm_joints + args.jaw_joints != state_dim:
        print(f"[WARN] arm_joints + jaw_joints != state_dim.")

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    policy = make_policy(policy_class, policy_config, ckpt_path, device)

    qpos_mean   = np.asarray(stats["qpos_mean"],   dtype=np.float32).reshape(-1)
    qpos_std    = np.asarray(stats["qpos_std"],    dtype=np.float32).reshape(-1)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32).reshape(-1)
    action_std  = np.asarray(stats["action_std"],  dtype=np.float32).reshape(-1)

    if len(qpos_mean) != state_dim:
        print(f"[WARN] state_dim={state_dim} but stats len={len(qpos_mean)}.")
        state_dim = len(qpos_mean)

    default_topics = {
        "left":    args.left_image_topic,
        "right":   args.right_image_topic,
        "wrist_1": args.wrist1_image_topic,
        "wrist_2": args.wrist2_image_topic,
    }
    camera_topic_map = {}
    for cam in camera_names:
        if cam not in default_topics or not default_topics[cam]:
            raise ValueError(f"No topic for camera '{cam}'.")
        camera_topic_map[cam] = default_topics[cam]

    sensor_node = SensorNode(
        camera_topic_map,
        image_encoding=args.image_encoding,
        cf_topic=(args.measured_cf_topic if use_cf else None),
        force_record_topic=(args.force_record_topic if args.record_force else None),
    )
    sensor_exec = rclpy.executors.SingleThreadedExecutor()
    sensor_exec.add_node(sensor_node)
    sensor_thread = threading.Thread(target=sensor_exec.spin, daemon=True)
    sensor_thread.start()

    if args.dry_run:
        print("\n[DRY RUN] Skipping CRTK arm setup.")
        arm = None
    else:
        print("\n-- Building CRTK arm interface ...")
        arm = ArmCustom(ral.create_child(args.psm))
        try:
            safe_check_connections(ral, 5.0)
        except Exception as e:
            print(f"-- Warning: connection check failed: {e}")
        print("-- Enabling arm ...")
        if not arm.enable(10):
            sys.exit("-- enable() timed out.")
        print("-- Homing arm ...")
        if not arm.home(10):
            sys.exit("-- home() timed out.")
        print("-- Arm is ENABLED + HOMED.")
        confirm("Verify the tool is installed. Press Enter when ready.",
                skip=args.skip_confirmations)
        if not args.skip_jaw_init:
            confirm("Press Enter to OPEN the gripper (+40 deg)", skip=args.skip_confirmations)
            arm.jaw.move_jp(np.array([40.0 * np.pi / 180.0])).wait()
            confirm("Press Enter to CLOSE the gripper (-20 deg)", skip=args.skip_confirmations)
            arm.jaw.move_jp(np.array([-20.0 * np.pi / 180.0])).wait()

    print(f"\nWaiting for camera frames (max {args.state_timeout:.1f}s)...", flush=True)
    t0 = time.time()
    while time.time() - t0 < args.state_timeout:
        if all(sensor_node.images[c] is not None for c in camera_names):
            break
        time.sleep(0.05)
    missing = [c for c in camera_names if sensor_node.images[c] is None]
    if missing:
        raise RuntimeError(f"Timed out waiting for cameras: {missing}.")
    print("Cameras ready.")

    if use_cf:
        print(f"Waiting for first /measured_cf frame (max {args.state_timeout:.1f}s)...", flush=True)
        t0 = time.time()
        while time.time() - t0 < args.state_timeout:
            if sensor_node.latest_cf is not None:
                break
            time.sleep(0.05)
        if sensor_node.latest_cf is None:
            raise RuntimeError(f"Timed out waiting for /measured_cf on '{args.measured_cf_topic}'.")
        cf0 = sensor_node.latest_cf
        print(f"  cf[0] (full 6-D) = [Fx={cf0[0]:+.3f} Fy={cf0[1]:+.3f} Fz={cf0[2]:+.3f} "
              f"| Tx={cf0[3]:+.4f} Ty={cf0[4]:+.4f} Tz={cf0[5]:+.4f}]")
        print(f"  Will use first {cf_dim} dim(s).")
        cf0_z = (cf0[:cf_dim] - cf_mean_arr) / cf_std_arr
        print(f"  cf[0] z-scored (BEFORE B_run correction) = {[round(float(v), 2) for v in cf0_z]}")
        if not do_cf_zero and cf_clip_val > 0 and np.any(np.abs(cf0_z) > cf_clip_val):
            print(f"  ⚠️  --no_cf_zero is set AND some channels exceed clip. "
                  f"Strongly consider removing --no_cf_zero.")

    target_start_arm = None
    target_start_jaw = None
    if not args.dry_run and args.start_from_episode:
        print(f"\nLoading start pose from: {args.start_from_episode}")
        start_arm, start_jaw = load_start_pose_from_episode(
            args.start_from_episode, args.arm_joints, args.jaw_joints
        )
        print(f"  arm_jp[0]: {np.round(start_arm, 4).tolist()}")
        print(f"  jaw_jp[0]: {float(start_jaw[0]):+.4f} rad")
        confirm("Press Enter to MOVE to this start pose", skip=args.skip_confirmations)
        arm.move_jp(start_arm).wait()
        arm.jaw.move_jp(start_jaw).wait()
        print("-- Reached start pose.")
        target_start_arm = np.asarray(start_arm, dtype=np.float32)
        target_start_jaw = np.asarray(start_jaw, dtype=np.float32)

    # Warn if user enabled restart but didn't give a start pose
    if args.restart_after_sec > 0 and not args.dry_run and target_start_arm is None:
        print(f"\n[WARN] --restart_after_sec is set but --start_from_episode is not. "
              f"On restart the robot will NOT return to a known pose; only the "
              f"policy's internal chunk/aggregator state will be reset. "
              f"Strongly recommend passing --start_from_episode.")

    if args.dry_run:
        current_jp  = [0.0] * args.arm_joints
        current_jaw = 0.0
    elif target_start_arm is not None:
        current_jp  = target_start_arm.tolist()
        current_jaw = float(target_start_jaw[0])
    else:
        time.sleep(0.5)
        try:
            jp_now = safe_measured_jp(arm)
            jaw_now = safe_measured_jp(arm.jaw)
            current_jp  = jp_now[:args.arm_joints].tolist()
            current_jaw = float(jaw_now[0]) if len(jaw_now) >= 1 else 0.0
        except Exception as e:
            print(f"[WARN] Could not read measured_jp: {e}.")
            current_jp  = [0.0] * args.arm_joints
            current_jaw = 0.0

    print(f"\nCurrent pose: jp={[round(v,5) for v in current_jp]}  jaw={current_jaw:.5f}")

    resize = (int(args.resize_w), int(args.resize_h)) if args.resize_w > 0 and args.resize_h > 0 else None

    confirm(f"Press Enter to start ACT inference at {args.fps:.1f} Hz "
            f"({'DRY RUN' if args.dry_run else 'LIVE'})", skip=args.skip_confirmations)

    # ── Warmup ───────────────────────────────────────────────────────────
    if not args.dry_run:
        if target_start_arm is not None:
            current_jp  = target_start_arm.tolist()
            current_jaw = float(target_start_jaw[0])
            print(f"\nWarmup will use --start_from_episode pose.")
        else:
            try:
                jp_now = safe_measured_jp(arm)
                jaw_now = safe_measured_jp(arm.jaw)
                current_jp  = jp_now[:args.arm_joints].tolist()
                current_jaw = float(jaw_now[0]) if len(jaw_now) >= 1 else 0.0
                print(f"\nWarmup will use current measured pose.")
            except Exception as e:
                print(f"[WARN] Could not re-read measured_jp: {e}")

        try:
            jp_actual = safe_measured_jp(arm)
            actual_arm = np.asarray(jp_actual[:args.arm_joints], dtype=np.float32)
            target_arm_arr = np.asarray(current_jp, dtype=np.float32)
            diff = float(np.max(np.abs(actual_arm - target_arm_arr)))
            print(f"  | measured - target |_inf = {diff:.4f} rad ({np.degrees(diff):.2f} deg)")
            if diff > 0.05:
                print(f"  ⚠️  Large gap, re-running move_jp ...")
                arm.move_jp(target_arm_arr).wait()
        except Exception as e:
            print(f"[WARN] Sanity check failed: {e}")

        print(f"\nWarmup step 1: move_jp to lock target pose ...", flush=True)
        try:
            arm.move_jp(np.array(current_jp, dtype=np.float32)).wait()
            arm.jaw.move_jp(np.array([current_jaw], dtype=np.float32)).wait()
        except Exception as e:
            print(f"[ERROR] move_jp during warmup failed: {e}")
            raise

        if do_cf_zero:
            print(f"Warmup step 2: streaming servo_jp for {args.warmup_sec:.1f}s "
                  f"(also collecting CF baseline B_run) ...", flush=True)
        else:
            print(f"Warmup step 2: streaming servo_jp for {args.warmup_sec:.1f}s ...", flush=True)

        cf_baseline_samples = []
        cf_settle_sec = min(args.cf_zero_settle_sec, max(0.0, args.warmup_sec - 0.5))

        t0 = time.time()
        warmup_dt = 1.0 / WATCHDOG_SAFE_HZ
        while time.time() - t0 < args.warmup_sec:
            arm.servo_jp(np.array(current_jp, dtype=np.float32))
            arm.jaw.servo_jp(np.array([current_jaw], dtype=np.float32))
            if (do_cf_zero
                    and (time.time() - t0) > cf_settle_sec
                    and sensor_node.latest_cf is not None):
                cf_baseline_samples.append(sensor_node.latest_cf[:cf_dim].copy())
            time.sleep(warmup_dt)

        # Compute B_run
        if do_cf_zero:
            min_samples = max(5, int(args.fps * 0.5))
            if len(cf_baseline_samples) >= min_samples:
                cf_runtime_bias = np.mean(np.stack(cf_baseline_samples), axis=0).astype(np.float32)
                z_before = (cf_runtime_bias - cf_mean_arr) / cf_std_arr
                z_after  = ((cf_runtime_bias - cf_runtime_bias) - cf_mean_arr) / cf_std_arr
                print(f"\n[cf-zero] Collected {len(cf_baseline_samples)} cf samples during warmup.")
                print(f"[cf-zero] B_run = {cf_runtime_bias.tolist()}")
                print(f"[cf-zero] z-score BEFORE B_run correction (raw - mean)/std:")
                print(f"          [{', '.join(f'{v:+.2f}' for v in z_before)}]")
                print(f"[cf-zero] z-score AFTER  B_run correction at rest:")
                print(f"          [{', '.join(f'{v:+.2f}' for v in z_after)}]")
                if np.all(np.abs(z_after) <= 1.5):
                    print(f"[cf-zero] ✅ Calibration looks good (residual within ±1.5 sigma).")
                else:
                    # z_after = -cf_mean/cf_std; large value just means training data
                    # had non-zero cf_mean (i.e., NOT zero-aligned at filter time).
                    # That is fine — the runtime sample after correction still aligns
                    # to the training distribution (because both subtract their own bias).
                    print(f"[cf-zero] Note: z_after equals -cf_mean/cf_std.")
                    print(f"          If your training data was NOT zero-aligned at filter time,")
                    print(f"          cf_mean is the training-time bias center. After B_run correction,")
                    print(f"          the runtime distribution is aligned to the training distribution.")
            else:
                print(f"\n[cf-zero] WARNING: only got {len(cf_baseline_samples)} cf samples "
                      f"(need >={min_samples}). NOT applying B_run correction.")
                print(f"[cf-zero] Increase --warmup_sec or check /measured_cf publish rate.")
                cf_runtime_bias = np.zeros(cf_dim, dtype=np.float32)

        print("Warmup done. ACT will now take over from this pose.")

    # ── Control loop ────────────────────────────────────────────────────
    aggregator       = ChunkAggregator(decay_k=args.temporal_decay)
    last_chunk       = None
    last_chunk_start = 0
    step_idx         = 0
    dt               = 1.0 / args.fps

    skip_count_image = 0
    skip_count_state = 0
    skip_count_cf    = 0
    skip_count_other = 0
    cf_outlier_count = 0
    last_diag_print  = time.time()

    # ── Auto-restart bookkeeping ────────────────────────────────────────
    restart_count    = 0
    loop_start_time  = time.time()
    next_attempt     = 1

    if args.record_force:
        import re
        os.makedirs(args.force_output_dir, exist_ok=True)
        prefix = args.task_name or 'act_run'
        pattern = re.compile(rf"^{re.escape(prefix)}_attempt_(\d+)_force_")
        for d in os.listdir(args.force_output_dir):
            m = pattern.match(d)
            if m:
                num = int(m.group(1))
                if num >= next_attempt:
                    next_attempt = num + 1

    print(f"\nRunning ACT @ {args.fps:.1f} Hz. Ctrl+C to stop.\n", flush=True)
    if args.restart_after_sec > 0:
        cap = "∞" if args.max_restarts < 0 else str(args.max_restarts)
        print(f"[restart] Auto-restart every {args.restart_after_sec:.1f}s "
              f"(max restarts: {cap}).\n", flush=True)

    # ── Begin force recording (if enabled) ─────────────────────────────
    if args.record_force:
        sensor_node.start_force_recording()
        print(f"[force-record] Recording from {args.force_record_topic} "
              f"-> will save to {args.force_output_dir} on exit.")

    try:
        while True:
            if args.max_steps > 0 and step_idx >= args.max_steps:
                print(f"\nReached --max_steps={args.max_steps}, stopping.")
                break

            # ── Auto-restart based on wall-clock elapsed in the loop ───
            if args.restart_after_sec > 0:
                elapsed_loop = time.time() - loop_start_time
                if elapsed_loop >= args.restart_after_sec:
                    if args.max_restarts >= 0 and restart_count >= args.max_restarts:
                        print(f"\n[restart] Reached max restarts "
                              f"({args.max_restarts}). Stopping.")
                        break
                    restart_count += 1
                    print(f"\n[restart] {elapsed_loop:.1f}s elapsed -> "
                          f"restart #{restart_count} "
                          f"(steps run this attempt: {step_idx})")

                    if args.record_force:
                        try:
                            records = sensor_node.stop_force_recording()
                            save_force_recording(
                                records,
                                output_dir=args.force_output_dir,
                                run_name=f"{(args.task_name or 'act_run')}_attempt_{next_attempt}",
                                source_topic=args.force_record_topic,
                            )
                            next_attempt += 1
                            sensor_node.start_force_recording()
                        except Exception as e:
                            print(f"[force-record] ERROR while saving during restart: {e}")

                    if args.pause_between_restarts_sec > 0:
                        print(f"[restart] Pausing for {args.pause_between_restarts_sec}s before restarting...")
                        t0_pause = time.time()
                        pause_dt = 1.0 / WATCHDOG_SAFE_HZ
                        while time.time() - t0_pause < args.pause_between_restarts_sec:
                            if not args.dry_run:
                                arm.servo_jp(np.array(current_jp, dtype=np.float32))
                                arm.jaw.servo_jp(np.array([current_jaw], dtype=np.float32))
                            time.sleep(pause_dt)

                    # 1) Reset policy-side temporal state
                    aggregator.reset()
                    last_chunk       = None
                    last_chunk_start = 0
                    step_idx         = 0

                    # 2) Move robot back to start pose, if available
                    if not args.dry_run and target_start_arm is not None:
                        print(f"[restart] Moving back to start pose ...")
                        try:
                            arm.move_jp(target_start_arm).wait()
                            arm.jaw.move_jp(target_start_jaw).wait()
                            current_jp  = target_start_arm.tolist()
                            current_jaw = float(target_start_jaw[0])

                            # Mini-warmup: stream servo_jp briefly so the
                            # PSM watchdog stays happy and the arm settles.
                            t0 = time.time()
                            settle_dt = 1.0 / WATCHDOG_SAFE_HZ
                            while time.time() - t0 < 0.5:
                                arm.servo_jp(np.array(current_jp,
                                                      dtype=np.float32))
                                arm.jaw.servo_jp(np.array([current_jaw],
                                                          dtype=np.float32))
                                time.sleep(settle_dt)
                            print(f"[restart] Resuming inference.")
                        except Exception as e:
                            print(f"[restart] move_jp failed: {e}. "
                                  f"Resuming from current pose.")
                    elif args.dry_run:
                        print(f"[restart] DRY RUN: skipping move_jp.")
                    else:
                        print(f"[restart] No --start_from_episode given; "
                              f"resuming from current pose (state reset only).")

                    # 3) Restart the timer for the next attempt
                    loop_start_time = time.time()
                    last_diag_print = time.time()
                    continue

            t_step_start = time.time()

            if time.time() - last_diag_print > 2.0:
                if step_idx == 0:
                    print(f"[DIAG] Loop stuck. skip_image={skip_count_image} "
                          f"skip_state={skip_count_state} skip_cf={skip_count_cf} "
                          f"skip_other={skip_count_other}")
                    now = time.time()
                    for cam in camera_names:
                        t_img = sensor_node.img_times.get(cam)
                        if t_img is None:
                            print(f"  [{cam}] NO IMAGE EVER RECEIVED")
                        else:
                            age = now - t_img
                            print(f"  [{cam}] last image age = {age:.3f}s "
                                  f"(thr {args.max_image_age:.3f}) "
                                  f"{'TOO OLD' if age > args.max_image_age else 'ok'}")
                    if use_cf:
                        if sensor_node.cf_time is None:
                            print(f"  [cf] NO CF EVER RECEIVED")
                        else:
                            age = now - sensor_node.cf_time
                            print(f"  [cf] last cf age = {age:.3f}s "
                                  f"(thr {args.max_cf_age:.3f}) "
                                  f"{'TOO OLD' if age > args.max_cf_age else 'ok'}")
                last_diag_print = time.time()

            # Image freshness
            images_ok = True
            stale_cam = None
            for cam in camera_names:
                t_img = sensor_node.img_times.get(cam)
                if t_img is None or (time.time() - t_img) > args.max_image_age:
                    images_ok = False
                    stale_cam = cam
                    break
            if not images_ok:
                skip_count_image += 1
                if args.debug and skip_count_image % 10 == 1:
                    age = ("never" if sensor_node.img_times.get(stale_cam) is None
                           else f"{time.time() - sensor_node.img_times[stale_cam]:.3f}s")
                    print(f"[step {step_idx}] stale: {stale_cam} (age={age})")
                if not args.dry_run:
                    arm.servo_jp(np.array(current_jp, dtype=np.float32))
                    arm.jaw.servo_jp(np.array([current_jaw], dtype=np.float32))
                time.sleep(0.01)
                continue

            # Image tensor
            img_list = []
            for cam in camera_names:
                img = sensor_node.images[cam].copy()
                if resize is not None:
                    img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img_list.append(img)
            image_np = np.stack(img_list, axis=0)
            image_t  = torch.from_numpy(image_np).float().unsqueeze(0).to(device)

            # qpos
            if args.dry_run:
                arm_pos = np.array(current_jp, dtype=np.float32)
                jaw_pos = np.array([current_jaw], dtype=np.float32)
            else:
                try:
                    jp_now = safe_measured_jp(arm)
                    jaw_now = safe_measured_jp(arm.jaw)
                    arm_pos = np.asarray(jp_now[:args.arm_joints], dtype=np.float32)
                    jaw_pos = np.asarray(jaw_now[:args.jaw_joints], dtype=np.float32)
                except Exception as e:
                    skip_count_state += 1
                    if skip_count_state % 10 == 1:
                        print(f"[step {step_idx}] measured_jp failed: {e}")
                    if not args.dry_run:
                        arm.servo_jp(np.array(current_jp, dtype=np.float32))
                        arm.jaw.servo_jp(np.array([current_jaw], dtype=np.float32))
                    time.sleep(0.01)
                    continue

            qpos_full = np.concatenate([arm_pos, jaw_pos], axis=0).astype(np.float32)
            qpos_np   = qpos_full[:state_dim]
            qpos_np   = (qpos_np - qpos_mean) / qpos_std

            if use_cf:
                cf_raw = sensor_node.latest_cf
                cf_age = (time.time() - sensor_node.cf_time
                          if sensor_node.cf_time is not None else float("inf"))
                if cf_raw is None or cf_age > args.max_cf_age:
                    skip_count_cf += 1
                    if skip_count_cf % 10 == 1:
                        print(f"[step {step_idx}] cf stale (age={cf_age:.3f}s)")
                    if not args.dry_run:
                        arm.servo_jp(np.array(current_jp, dtype=np.float32))
                        arm.jaw.servo_jp(np.array([current_jaw], dtype=np.float32))
                    time.sleep(0.01)
                    continue

                # ── Path-1 calibration: subtract B_run, then z-score, then clip ─
                cf_corrected = cf_raw[:cf_dim].astype(np.float32) - cf_runtime_bias[:cf_dim]
                cf_np = (cf_corrected - cf_mean_arr) / cf_std_arr

                if cf_clip_val > 0:
                    if np.any(np.abs(cf_np) > cf_clip_val):
                        cf_outlier_count += 1
                        if args.debug and cf_outlier_count % 20 == 1:
                            print(f"[step {step_idx}] cf outlier "
                                  f"(z={[round(float(v),2) for v in cf_np]}) "
                                  f"-> clipped to ±{cf_clip_val}")
                    cf_np = np.clip(cf_np, -cf_clip_val, cf_clip_val)

                state_np = np.concatenate([qpos_np, cf_np], axis=-1)
            else:
                state_np = qpos_np

            qpos_t = torch.from_numpy(state_np).float().unsqueeze(0).to(device)

            # Inference
            with torch.inference_mode():
                if policy_class == "ACT":
                    if temporal_agg:
                        all_actions = policy(qpos_t, image_t)
                        all_actions = all_actions.squeeze(0).cpu().numpy()
                        aggregator.add_chunk(step_idx, all_actions)
                        raw_action = aggregator.get_action(step_idx)
                        if raw_action is None:
                            time.sleep(0.01); continue
                    else:
                        if step_idx % query_frequency == 0:
                            all_actions = policy(qpos_t, image_t)
                            last_chunk = all_actions.squeeze(0).cpu().numpy()
                            last_chunk_start = step_idx
                        if last_chunk is None:
                            time.sleep(0.01); continue
                        offset = step_idx - last_chunk_start
                        if offset >= len(last_chunk):
                            time.sleep(0.01); continue
                        raw_action = last_chunk[offset]
                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos_t, image_t).squeeze(0).cpu().numpy()
                else:
                    raise NotImplementedError

            action_np = raw_action * action_std + action_mean
            if args.clip_actions:
                action_np = np.clip(action_np, -args.action_clip_value, args.action_clip_value)

            jp_cmd  = action_np[:args.arm_joints].tolist()
            jaw_cmd = float(action_np[args.arm_joints]) if len(action_np) > args.arm_joints else 0.0

            if args.max_delta_jp > 0.0:
                cur = np.array(current_jp, dtype=np.float32)
                cmd = np.array(jp_cmd, dtype=np.float32)
                delta = cmd - cur
                norm = float(np.linalg.norm(delta))
                if norm > args.max_delta_jp:
                    cmd = cur + delta * (args.max_delta_jp / norm)
                    jp_cmd = cmd.tolist()
                    if args.debug:
                        print(f"[step {step_idx}] Δjp clipped {norm:.4f} -> {args.max_delta_jp:.4f}")

            if args.dry_run:
                if args.debug or step_idx % 10 == 0:
                    jp_str = "  ".join(f"{v:+.4f}" for v in jp_cmd)
                    cf_str = ""
                    if use_cf and sensor_node.latest_cf is not None:
                        cf_show = sensor_node.latest_cf[:cf_dim] - cf_runtime_bias[:cf_dim]
                        cf_str = "  cf_corr=[" + ",".join(f"{v:+.3f}" for v in cf_show) + "]"
                    print(f"[DRY] step={step_idx:5d} | jp=[{jp_str}]  jaw={jaw_cmd:+.4f}{cf_str}")
            else:
                arm.servo_jp(np.array(jp_cmd, dtype=np.float32))
                arm.jaw.servo_jp(np.array([jaw_cmd], dtype=np.float32))
                if args.debug:
                    jp_str = "  ".join(f"{v:+.4f}" for v in jp_cmd)
                    print(f"step={step_idx:5d} | jp=[{jp_str}]  jaw={jaw_cmd:+.4f}")

            current_jp  = jp_cmd
            current_jaw = jaw_cmd
            step_idx   += 1

            elapsed = time.time() - t_step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print(f"\nStopped after {step_idx} steps in current attempt "
              f"(restarts so far: {restart_count}).")
    finally:
        if use_cf and step_idx > 0:
            outlier_pct = 100.0 * cf_outlier_count / max(1, step_idx)
            print(f"[STATS] CF outliers (clipped): {cf_outlier_count}/{step_idx} "
                  f"= {outlier_pct:.1f}% "
                  f"({'normal' if outlier_pct < 5 else 'HIGH'})")
            if do_cf_zero:
                print(f"[STATS] B_run used: {cf_runtime_bias[:cf_dim].tolist()}")
        if args.restart_after_sec > 0:
            print(f"[STATS] Total restarts performed: {restart_count}")

        if not args.dry_run and arm is not None:
            try:
                t0 = time.time()
                while time.time() - t0 < 0.25:
                    arm.servo_jp(np.array(current_jp, dtype=np.float32))
                    arm.jaw.servo_jp(np.array([current_jaw], dtype=np.float32))
                    time.sleep(1.0 / WATCHDOG_SAFE_HZ)
            except Exception:
                pass

        # ── Save force recording (before tearing down sensor node) ────
        if args.record_force:
            try:
                records = sensor_node.stop_force_recording()
                save_force_recording(
                    records,
                    output_dir=args.force_output_dir,
                    run_name=f"{(args.task_name or 'act_run')}_attempt_{next_attempt}",
                    source_topic=args.force_record_topic,
                )
            except Exception as e:
                print(f"[force-record] ERROR while saving: {e}")

        sensor_exec.shutdown()
        sensor_node.destroy_node()
        print("Shutdown complete.")


# ============================================================
# CLI
# ============================================================

def main():
    argv = crtk.ral.parse_argv(sys.argv)

    p = argparse.ArgumentParser(
        description="Run ACT policy on dVRK PSM. Auto-detects CF mode and "
                    "auto-collects CF baseline (B_run) during warmup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--ckpt_dir",  required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--policy_class",    default="ACT", choices=["ACT", "CNNMLP"])
    p.add_argument("--state_dim",       type=int,   default=7)
    p.add_argument("--task_name",       default=None)
    p.add_argument("--camera_names",    default="left,wrist_1")
    p.add_argument("--chunk_size",      type=int,   default=50)
    p.add_argument("--hidden_dim",      type=int,   default=512)
    p.add_argument("--dim_feedforward", type=int,   default=3200)
    p.add_argument("--kl_weight",       type=float, default=10.0)
    p.add_argument("--lr",              type=float, default=1e-5)

    p.add_argument("--psm",        default="PSM2", choices=["PSM1", "PSM2", "PSM3"])
    p.add_argument("--arm_joints", type=int,   default=6)
    p.add_argument("--jaw_joints", type=int,   default=1)
    p.add_argument("--fps",        type=float, default=10.0)
    p.add_argument("--joint_dim",  type=int,   default=None)
    p.add_argument("--cpu",        action="store_true")
    p.add_argument("--warmup_sec",    type=float, default=2.0,
                   help="Warmup duration. CF baseline is collected during this window. "
                        "Increase to >=3 sec if CF publish rate is low.")
    p.add_argument("--state_timeout", type=float, default=10.0)

    p.add_argument("--left_image_topic",   default="/console1/left/image_raw")
    p.add_argument("--right_image_topic",  default="/console1/right/image_raw")
    p.add_argument("--wrist1_image_topic", default="/wrist/cam1/image_raw")
    p.add_argument("--wrist2_image_topic", default="/wrist/cam2/image_raw")
    p.add_argument("--image_encoding",     default="bgr8", choices=["bgr8", "rgb8"])
    p.add_argument("--resize_w", type=int, default=320)
    p.add_argument("--resize_h", type=int, default=240)

    p.add_argument("--measured_cf_topic", default="/measured_cf")
    p.add_argument("--cf_dim", type=int, default=None)
    p.add_argument("--max_cf_age", type=float, default=0.5)
    p.add_argument("--no_cf", action="store_true",
                   help="Force-disable cf usage (legacy ckpts only).")

    p.add_argument("--no_cf_zero", action="store_true",
                   help="Disable B_run baseline collection during warmup. NOT RECOMMENDED.")
    p.add_argument("--cf_zero_settle_sec", type=float, default=0.5,
                   help="Skip first N seconds of warmup before sampling cf for B_run.")

    p.add_argument("--temporal_agg",   action="store_true")
    p.add_argument("--temporal_decay", type=float, default=0.01)
    p.add_argument("--run_chunk_size", type=int,   default=None)
    p.add_argument("--max_image_age", type=float, default=1.0)
    p.add_argument("--max_state_age", type=float, default=0.2)
    p.add_argument("--clip_actions",      action="store_true")
    p.add_argument("--action_clip_value", type=float, default=10.0)
    p.add_argument("--max_delta_jp",      type=float, default=0.0)

    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--skip_jaw_init", action="store_true")
    p.add_argument("--skip_confirmations", action="store_true")
    p.add_argument("--start_from_episode", type=str, default=None)
    p.add_argument("--debug", action="store_true")

    # ── Auto-restart after timeout ────────────────────────────────────
    p.add_argument("--restart_after_sec", type=float, default=0.0,
                   help="If >0, automatically restart the policy after this many "
                        "seconds of wall-clock time. Resets the temporal aggregator "
                        "and chunk state, and (if --start_from_episode is given) "
                        "moves the robot back to the demonstrated start pose. "
                        "Useful for letting ACT 'try again' if an attempt fails.")
    p.add_argument("--max_restarts", type=int, default=-1,
                   help="Cap on the number of automatic restarts. -1 = unlimited.")
    p.add_argument("--pause_between_restarts_sec", type=float, default=0.0,
                   help="If >0, pause for this many seconds before executing a restart. "
                        "The robot will hold its current position during this pause.")

    # ── Force recording during ACT execution ───────────────────────────
    p.add_argument("--record_force", action="store_true",
                   help="Record external force-sensor data during ACT inference. "
                        "On exit (Ctrl+C / max_steps / error) saves raw data and "
                        "plots into a timestamped folder under --force_output_dir.")
    p.add_argument("--force_record_topic", default="/measured_cf",
                   help="ROS topic (WrenchStamped) to record force from. "
                        "If equal to --measured_cf_topic the existing subscription "
                        "is reused (no duplicate subscribe).")
    p.add_argument("--force_output_dir", default="./act_force_logs",
                   help="Parent folder for per-run force recordings.")

    args = p.parse_args(argv[1:])

    ral = crtk.ral("dvrk_act_runner")
    ral.spin_and_execute(lambda: run_act(ral, args))


if __name__ == "__main__":
    main()