#!/usr/bin/env python3
# record_dvrk_episodes.py
#
# Records dVRK teleoperation episodes into HDF5 files that are structurally
# identical to those produced by record_ambf_episodes.py, enabling the same
# ACT training pipeline to be used for both sim and real data.
#
# Key differences from record_ambf_episodes.py:
#   - State  : /PSM2/measured_js (position) + /PSM2/jaw/measured_js
#              => qpos = [arm_joints | jaw_joints]  (7 DOF by default)
#   - Action : /PSM2/setpoint_js (position) + /PSM2/jaw/setpoint_js
#              => the human operator's joint commands (what ACT learns to imitate)
#   - Images : /console1/left/image_raw, /console1/right/image_raw (stereo)
#              /wrist/cam1/image_raw, /wrist/cam2/image_raw (physical wrist cams)
#   - All msg types are sensor_msgs/JointState and sensor_msgs/Image (no ambf_msgs)
#
# HDF5 layout (identical to record_ambf_episodes.py):
#
#   attrs: source, sim, fps, dt, captured_steps, dropped_steps, capture_ratio,
#          joint_dim, wall_start/end_time, recorded_at_iso, camera_names,
#          color_order, resize_w/h, state_semantics, action_semantics,
#          state_tolerance_sec, image_tolerance_sec, command_tolerance_sec
#
#   observations/
#       qpos    (T, joint_dim)      PSM arm[6] + jaw[1] (measured positions)
#       qvel    (T, joint_dim)      PSM arm[6] + jaw[1] (measured velocities)
#       images/
#           left     (T, H, W, 3)  console left,  uint8, gzip
#           right    (T, H, W, 3)  console right, uint8, gzip
#           wrist_1  (T, H, W, 3)  wrist cam 1,   uint8, gzip
#           wrist_2  (T, H, W, 3)  wrist cam 2,   uint8, gzip
#
#   action      (T, joint_dim)  CMD arm[6] + jaw[1]      (operator command)
#   timestamp   (T,)            wall-clock seconds
#
#   source_timestamps/
#       psm_arm  (T,)   psm_jaw  (T,)
#       cmd_arm  (T,)   cmd_jaw (T,)
#       left (T,)  right (T,)  wrist_1 (T,)  wrist_2 (T,)
#
# Usage:
#   python record_dvrk_episodes.py \
#       --output_dir ./dvrk_dataset \
#       --psm PSM2 \
#       --seconds_per_episode 15 --fps 10
#
# Prerequisites:
#   1. dVRK system running:
#      ros2 run dvrk_robot dvrk_system -j <system.json> -m <manager.json>
#   2. Console cameras:
#      ros2 launch .../decklink_stereo_goovis.launch.py stereo_rig_name:=console1
#   3. Wrist cameras:
#      python wrist_cameras_video_new.py --cam1 0 --cam2 2

import os
import sys
import cv2
import time
import h5py
import signal
import select
import argparse
import termios
import tty
import subprocess
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped
from cv_bridge import CvBridge


# ============================================================
# Utility
# ============================================================

def wait_for_key(key='q'):
    print(f'\nPress "{key}" to start recording (Ctrl+C to quit)')
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue
            ch = sys.stdin.read(1)
            if ch == key:
                return True
            if ch == '\x03':
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def get_next_episode_idx(output_dir):
    existing = []
    if not os.path.exists(output_dir):
        return 0
    for name in os.listdir(output_dir):
        if name.startswith("episode_") and name.endswith(".hdf5"):
            try:
                existing.append(int(name[len("episode_"):-len(".hdf5")]))
            except ValueError:
                pass
    return 0 if not existing else max(existing) + 1


def msg_time_to_sec(msg):
    """Use local wall-clock receive time for all queues (same domain as target_t)."""
    return time.time()


def start_rosbag_recording(bag_path, topics):
    cmd = ["ros2", "bag", "record", "-o", bag_path] + topics
    print(f"[rosbag] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    return proc


def stop_rosbag_recording(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=10)
        print("[rosbag] Stopped")
    except Exception as e:
        print(f"[rosbag] Stop failed: {e}")


def nearest_from_queue(q, target_t, tolerance):
    if not q:
        return None
    best, best_dt = None, None
    for ts, payload in q:
        dt = abs(ts - target_t)
        if best is None or dt < best_dt:
            best, best_dt = (ts, payload), dt
    if best is None or best_dt > tolerance:
        return None
    return best


def latest_before_or_nearest(q, target_t, tolerance):
    if not q:
        return None
    best_before = None
    for ts, payload in q:
        if ts <= target_t:
            if best_before is None or ts > best_before[0]:
                best_before = (ts, payload)
    if best_before is not None and abs(best_before[0] - target_t) <= tolerance:
        return best_before
    return nearest_from_queue(q, target_t, tolerance)


# ============================================================
# Recorder Node
# ============================================================

class DVRKRecorder(Node):
    """
    Subscribes to dVRK state, CMD command, and camera topics.
    Buffers samples in deques for synchronized extraction.

    State  (qpos)  : PSM arm joints + jaw
    Action         : CMD arm joints + jaw      (operator commands)
    Cameras        : console left/right + wrist 1/2
    """

    def __init__(self,
                 psm_arm_topic,
                 psm_jaw_topic,
                 cmd_arm_topic,
                 cmd_jaw_topic,
                 left_topic,
                 right_topic,
                 wrist1_topic,
                 wrist2_topic,
                 spatial_cf_topic=None,
                #  body_cf_topic=None,
                 measured_cf_topic=None,
                 gravity_topic=None,
                 arm_joints=6,
                 jaw_joints=1,
                 queue_size=600,
                 image_encoding='bgr8'):

        super().__init__('dvrk_recorder')

        self.bridge = CvBridge()
        self.queue_size = queue_size
        self.image_encoding = image_encoding
        self.arm_joints = arm_joints
        self.jaw_joints = jaw_joints
        self.joint_dim  = arm_joints + jaw_joints

        # queues: (timestamp, payload)
        self.psm_arm_q  = deque(maxlen=queue_size)
        self.psm_jaw_q  = deque(maxlen=queue_size)
        self.cmd_arm_q  = deque(maxlen=queue_size)
        self.cmd_jaw_q = deque(maxlen=queue_size)
        self.left_q     = deque(maxlen=queue_size)
        self.right_q    = deque(maxlen=queue_size)
        self.wrist1_q   = deque(maxlen=queue_size)
        self.wrist2_q   = deque(maxlen=queue_size)
        self.spatial_cf_q = deque(maxlen=queue_size)  # spatial measured_cf
        # self.body_cf_q   = deque(maxlen=queue_size)  # body measured_cf
        self.measured_cf_q = deque(maxlen=queue_size)  # external force sensor
        self.gravity_q  = deque(maxlen=queue_size)  # gravity compensation

        self.topic_names = {
            "psm_arm_topic":  psm_arm_topic,
            "psm_jaw_topic":  psm_jaw_topic,
            "cmd_arm_topic":  cmd_arm_topic,
            "cmd_jaw_topic": cmd_jaw_topic,
            "left_topic":     left_topic,
            "right_topic":    right_topic,
            "wrist1_topic":   wrist1_topic,
            "wrist2_topic":   wrist2_topic,
            "spatial_cf_topic": spatial_cf_topic,
            # "body_cf_topic": body_cf_topic,
            "measured_cf_topic": measured_cf_topic,
            "gravity_topic": gravity_topic,
        }

        # joint-state subscribers
        self.create_subscription(JointState, psm_arm_topic,  self.psm_arm_cb,  50)
        self.create_subscription(JointState, psm_jaw_topic,  self.psm_jaw_cb,  50)
        self.create_subscription(JointState, cmd_arm_topic,  self.cmd_arm_cb,  50)
        self.create_subscription(JointState, cmd_jaw_topic, self.cmd_jaw_cb, 50)

        # camera subscribers
        self.create_subscription(Image, left_topic,   self.left_cb,   20)
        self.create_subscription(Image, right_topic,  self.right_cb,  20)
        if wrist1_topic.lower() != 'none':
            self.create_subscription(Image, wrist1_topic, self.wrist1_cb, 20)
        if wrist2_topic.lower() != 'none':
            self.create_subscription(Image, wrist2_topic, self.wrist2_cb, 20)

        # force/torque subscribers (required)
        self.create_subscription(WrenchStamped, spatial_cf_topic, self.spatial_cf_cb, 50)
        # self.create_subscription(WrenchStamped, body_cf_topic, self.body_cf_cb, 50)

        # measured_cf subscriber
        if measured_cf_topic is not None and measured_cf_topic.lower() != 'none':
            self.create_subscription(WrenchStamped, measured_cf_topic, self.measured_cf_cb, 50)

        # gravity compensation subscriber
        if gravity_topic is not None and gravity_topic.lower() != 'none':
            self.create_subscription(JointState, gravity_topic, self.gravity_cb, 50)

        self.get_logger().info("DVRKRecorder initialized")
        self.get_logger().info(f"  State  : {psm_arm_topic} + {psm_jaw_topic}")
        self.get_logger().info(f"  Action : {cmd_arm_topic} + {cmd_jaw_topic}")
        self.get_logger().info(f"  joint_dim = {self.joint_dim} ({arm_joints} arm + {jaw_joints} jaw)")

    # ── callbacks ──────────────────────────────────────────────────────────

    def _js_positions(self, msg, n_joints):
        """Extract the first n_joints positions from a JointState message, or efforts if positions are empty."""
        pos = np.array(msg.position, dtype=np.float32)
        if len(pos) >= n_joints:
            return pos[:n_joints]
        else:
            effort = np.array(msg.effort, dtype=np.float32)
            if len(effort) >= n_joints:
                return effort[:n_joints]
            else:
                raise RuntimeError(
                    f"Expected >= {n_joints} joints in position or effort from {msg}, got {len(pos)} position, {len(effort)} effort"
                )

    def _js_state(self, msg, n_joints):
        """Extract the first n_joints positions, velocities, and efforts from a JointState message."""
        pos = self._js_positions(msg, n_joints)
        vel = np.array(msg.velocity, dtype=np.float32) if msg.velocity else np.zeros(n_joints, dtype=np.float32)
        vel = vel[:n_joints] if len(vel) >= n_joints else np.pad(vel, (0, n_joints - len(vel)), 'constant')
        effort = np.array(msg.effort, dtype=np.float32) if msg.effort else np.zeros(n_joints, dtype=np.float32)
        effort = effort[:n_joints] if len(effort) >= n_joints else np.pad(effort, (0, n_joints - len(effort)), 'constant')
        return pos, vel, effort

    def psm_arm_cb(self, msg):
        ts = msg_time_to_sec(msg)
        pos, vel, effort = self._js_state(msg, self.arm_joints)
        self.psm_arm_q.append((ts, (pos, vel, effort)))

    def psm_jaw_cb(self, msg):
        ts = msg_time_to_sec(msg)
        pos, vel, effort = self._js_state(msg, self.jaw_joints)
        self.psm_jaw_q.append((ts, (pos, vel, effort)))

    def cmd_arm_cb(self, msg):
        ts = msg_time_to_sec(msg)
        self.cmd_arm_q.append((ts, self._js_positions(msg, self.arm_joints)))

    def cmd_jaw_cb(self, msg):
        ts = msg_time_to_sec(msg)
        self.cmd_jaw_q.append((ts, self._js_positions(msg, self.jaw_joints)))

    def _img_cb(self, msg, q):
        ts = msg_time_to_sec(msg)
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
        q.append((ts, img))

    def left_cb(self,   msg): self._img_cb(msg, self.left_q)
    def right_cb(self,  msg): self._img_cb(msg, self.right_q)
    def wrist1_cb(self, msg): self._img_cb(msg, self.wrist1_q)
    def wrist2_cb(self, msg): self._img_cb(msg, self.wrist2_q)

    def spatial_cf_cb(self, msg):
        ts = msg_time_to_sec(msg)
        force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                          msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z], dtype=np.float32)
        self.spatial_cf_q.append((ts, force))

    def body_cf_cb(self, msg):
        ts = msg_time_to_sec(msg)
        force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                          msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z], dtype=np.float32)
        self.body_cf_q.append((ts, force))

    def measured_cf_cb(self, msg):
        ts = msg_time_to_sec(msg)
        # Extract force/torque from WrenchStamped
        cf = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                       msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z], dtype=np.float32)
        self.measured_cf_q.append((ts, cf))

    def gravity_cb(self, msg):
        ts = msg_time_to_sec(msg)
        gravity = self._js_positions(msg, self.arm_joints)
        self.gravity_q.append((ts, gravity))

    # ── synchronized sample extraction ────────────────────────────────────

    def get_synced_sample(self,
                          anchor_t,
                          state_tolerance,
                          image_tolerance,
                          command_tolerance,
                          resize=None,
                          command_mode='latest_before_or_nearest'):
        """
        Anchor on PSM arm state time; match all other streams within tolerance.
        Returns a dict mirroring record_ambf_episodes.get_synced_sample().
        """

        # ── PSM arm state (anchor) ────────────────────────────────────────
        arm_item = nearest_from_queue(self.psm_arm_q, anchor_t, state_tolerance)
        if arm_item is None:
            return None
        arm_ts, (arm_pos, arm_vel, arm_effort) = arm_item

        # ── PSM jaw state ─────────────────────────────────────────────────
        jaw_item = nearest_from_queue(self.psm_jaw_q, arm_ts, state_tolerance)
        if jaw_item is None:
            return None
        jaw_ts, (jaw_pos, jaw_vel, jaw_effort) = jaw_item

        # qpos = [arm | jaw]
        qpos = np.concatenate([arm_pos, jaw_pos], axis=0)

        # ── CMD arm command ───────────────────────────────────────────────
        if command_mode == 'latest_before_or_nearest':
            cmd_arm_item = latest_before_or_nearest(self.cmd_arm_q, arm_ts, command_tolerance)
        else:
            cmd_arm_item = nearest_from_queue(self.cmd_arm_q, arm_ts, command_tolerance)
        if cmd_arm_item is None:
            return None
        cmd_arm_ts, cmd_arm_pos = cmd_arm_item

        # ── CMD jaw command ───────────────────────────────────────────────
        if command_mode == 'latest_before_or_nearest':
            cmd_jaw_item = latest_before_or_nearest(self.cmd_jaw_q, arm_ts, command_tolerance)
        else:
            cmd_jaw_item = nearest_from_queue(self.cmd_jaw_q, arm_ts, command_tolerance)
        if cmd_jaw_item is None:
            return None
        cmd_jaw_ts, cmd_jaw_pos = cmd_jaw_item

        # action = [cmd_arm | cmd_jaw]
        action = np.concatenate([cmd_arm_pos, cmd_jaw_pos], axis=0)

        # qvel field: actual velocity from the measured joint states
        qvel = np.concatenate([arm_vel, jaw_vel], axis=0)

        # ── cameras ───────────────────────────────────────────────────────
        left_item   = nearest_from_queue(self.left_q,   arm_ts, image_tolerance)
        right_item  = nearest_from_queue(self.right_q,  arm_ts, image_tolerance)

        if left_item is None or right_item is None:
            return None

        imgs = {
            "left":  left_item[1].copy(),
            "right": right_item[1].copy(),
        }
        source_cams = {
            "left":  float(left_item[0]),
            "right": float(right_item[0]),
        }

        if self.topic_names["wrist1_topic"].lower() != 'none':
            w1_item = nearest_from_queue(self.wrist1_q, arm_ts, image_tolerance)
            if w1_item is None: return None
            imgs["wrist_1"] = w1_item[1].copy()
            source_cams["wrist_1"] = float(w1_item[0])

        if self.topic_names["wrist2_topic"].lower() != 'none':
            w2_item = nearest_from_queue(self.wrist2_q, arm_ts, image_tolerance)
            if w2_item is None: return None
            imgs["wrist_2"] = w2_item[1].copy()
            source_cams["wrist_2"] = float(w2_item[0])

        if resize is not None:
            w, h = resize
            imgs = {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA)
                    for k, v in imgs.items()}

        all_ts = [arm_ts, jaw_ts, cmd_arm_ts, cmd_jaw_ts] + list(source_cams.values())
        spread = float(max(all_ts) - min(all_ts))

        # ── spatial measured_cf ──────────────────────────────────────────
        spatial_item = nearest_from_queue(self.spatial_cf_q, arm_ts, state_tolerance)
        if spatial_item is None:
            return None
        spatial_cf = spatial_item[1]

        # ── body measured_cf ─────────────────────────────────────────────
        # body_item = nearest_from_queue(self.body_cf_q, arm_ts, state_tolerance)
        # if body_item is None:
        #     return None
        # body_cf = body_item[1]

        # ── measured_cf (external force sensor) ────────────────────────
        measured_cf = None
        if len(self.measured_cf_q) > 0:
            measured_cf_item = nearest_from_queue(self.measured_cf_q, arm_ts, state_tolerance)
            if measured_cf_item is not None:
                measured_cf = measured_cf_item[1]
        # ── gravity compensation (for joint force calculation) ────────
        joint_force = None
        if len(self.gravity_q) > 0:
            gravity_item = nearest_from_queue(self.gravity_q, arm_ts, state_tolerance)
            if gravity_item is not None:
                gravity_effort = gravity_item[1]
                # joint_force = measured_effort - gravity_effort
                joint_force = arm_effort - gravity_effort
        return {
            "sample_time": float(arm_ts),
            "source_times": {
                "psm_arm":  float(arm_ts),
                "psm_jaw":  float(jaw_ts),
                "cmd_arm":  float(cmd_arm_ts),
                "cmd_jaw": float(cmd_jaw_ts),
                **source_cams
            },
            "time_spread": spread,
            "qpos":   qpos,
            "qvel":   qvel,   # PSM measured velocity
            "action": action,
            "images": imgs,
            "arm_effort": arm_effort,
            "spatial_cf": spatial_cf,
            # "body_cf": body_cf,
            "measured_cf": measured_cf,
            "joint_force": joint_force,
        }


# ============================================================
# Episode recording  (mirrors record_ambf_episodes.record_episode)
# ============================================================

def record_episode(node,
                   seconds,
                   fps,
                   state_tolerance,
                   image_tolerance,
                   command_tolerance,
                   resize=None,
                   warmup_sec=1.0,
                   command_mode='latest_before_or_nearest'):

    dt = 1.0 / fps

    qpos_list   = []
    qvel_list   = []
    action_list = []
    timestamps  = []
    arm_effort_list = []
    spatial_cf_list = []
    body_cf_list = []
    measured_cf_list = []
    joint_force_list = []

    cam_names = ["left", "right"]
    if node.topic_names["wrist1_topic"].lower() != 'none': cam_names.append("wrist_1")
    if node.topic_names["wrist2_topic"].lower() != 'none': cam_names.append("wrist_2")

    source_times = {k: [] for k in ["psm_arm", "psm_jaw", "cmd_arm", "cmd_jaw"] + cam_names}
    images = {k: [] for k in cam_names}

    # warmup — let queues fill before sampling
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_sec:
        rclpy.spin_once(node, timeout_sec=0.01)

    wall_start   = time.time()
    sample_start = wall_start
    requested_steps = int(round(seconds * fps))
    dropped_steps   = 0
    accepted_steps  = 0

    for k in range(requested_steps):
        target_t = sample_start + k * dt

        # wait until target time
        while True:
            now = time.time()
            if now >= target_t:
                break
            rclpy.spin_once(node, timeout_sec=min(0.003, target_t - now))

        # brief settle
        settle_until = target_t + min(0.005, dt * 0.25)
        while time.time() < settle_until:
            rclpy.spin_once(node, timeout_sec=0.001)

        sample = node.get_synced_sample(
            anchor_t=target_t,
            state_tolerance=state_tolerance,
            image_tolerance=image_tolerance,
            command_tolerance=command_tolerance,
            resize=resize,
            command_mode=command_mode,
        )

        if sample is None:
            dropped_steps += 1
            continue

        qpos_list.append(sample["qpos"])
        qvel_list.append(sample["qvel"])
        action_list.append(sample["action"])
        timestamps.append(sample["sample_time"])
        arm_effort_list.append(sample["arm_effort"])
        spatial_cf_list.append(sample["spatial_cf"])
        # body_cf_list.append(sample["body_cf"])
        if sample["measured_cf"] is not None:
            measured_cf_list.append(sample["measured_cf"])
        if sample["joint_force"] is not None:
            joint_force_list.append(sample["joint_force"])

        for name in source_times:
            source_times[name].append(sample["source_times"][name])
        for cam in images:
            images[cam].append(sample["images"][cam])

        accepted_steps += 1

    capture_ratio = accepted_steps / requested_steps if requested_steps > 0 else 0.0

    return {
        "qpos":         qpos_list,
        "qvel":         qvel_list,
        "action":       action_list,
        "images":       images,
        "timestamps":   timestamps,
        "source_times": source_times,
        "arm_effort":   arm_effort_list,
        "spatial_cf":   spatial_cf_list,
        # "body_cf":      body_cf_list,
        "measured_cf":  measured_cf_list,
        "joint_force":  joint_force_list,
        "meta": {
            "requested_seconds":    float(seconds),
            "requested_fps":        float(fps),
            "requested_steps":      int(requested_steps),
            "captured_steps":       int(accepted_steps),
            "dropped_steps":        int(dropped_steps),
            "capture_ratio":        float(capture_ratio),
            "state_tolerance_sec":  float(state_tolerance),
            "image_tolerance_sec":  float(image_tolerance),
            "command_tolerance_sec": float(command_tolerance),

            "wall_start_time":      float(wall_start),
            "wall_end_time":        float(time.time()),
            "warmup_sec":           float(warmup_sec),
            "joint_dim":            int(node.joint_dim),
            "command_mode":         str(command_mode),
        }
    }


# ============================================================
# Save HDF5  (identical layout to record_ambf_episodes.save_episode)
# ============================================================

def save_episode(save_path,
                 data,
                 episode_idx,
                 topic_names,
                 resize,
                 image_encoding):

    qpos   = np.stack(data["qpos"])
    qvel   = np.stack(data["qvel"])
    action = np.stack(data["action"])
    cam_names = list(data["images"].keys())   # ["left", "right", "wrist_1", "wrist_2"]

    with h5py.File(save_path, 'w') as f:

        # ── attrs ─────────────────────────────────────────────────────────
        f.attrs['sim']             = False
        f.attrs['source']          = 'dvrk'
        f.attrs['episode_idx']     = int(episode_idx)
        f.attrs['fps']             = float(data["meta"]["requested_fps"])
        f.attrs['dt']              = 1.0 / float(data["meta"]["requested_fps"])
        f.attrs['requested_steps'] = int(data["meta"]["requested_steps"])
        f.attrs['captured_steps']  = int(data["meta"]["captured_steps"])
        f.attrs['dropped_steps']   = int(data["meta"]["dropped_steps"])
        f.attrs['capture_ratio']   = float(data["meta"]["capture_ratio"])

        f.attrs['state_tolerance_sec']   = float(data["meta"]["state_tolerance_sec"])
        f.attrs['image_tolerance_sec']   = float(data["meta"]["image_tolerance_sec"])
        f.attrs['command_tolerance_sec'] = float(data["meta"]["command_tolerance_sec"])

        f.attrs['joint_dim']       = int(data["meta"]["joint_dim"])
        f.attrs['wall_start_time'] = float(data["meta"]["wall_start_time"])
        f.attrs['wall_end_time']   = float(data["meta"]["wall_end_time"])
        f.attrs['recorded_at_iso'] = datetime.now().isoformat(timespec="seconds")
        f.attrs['command_mode']    = str(data["meta"]["command_mode"])

        f.attrs['camera_names']      = np.array(cam_names, dtype='S')
        f.attrs['color_order']       = str(image_encoding)
        f.attrs['state_semantics']   = 'PSM_arm[:arm_joints] | PSM_jaw[:jaw_joints]'
        f.attrs['action_semantics']  = 'cmd_arm[:arm_joints] | cmd_jaw[:jaw_joints]'
        f.attrs['qvel_semantics']    = 'PSM_arm_vel[:arm_joints] | PSM_jaw_vel[:jaw_joints] (measured velocity)'

        for k, v in topic_names.items():
            f.attrs[k] = v

        if resize is None:
            f.attrs['resize_w'] = 0
            f.attrs['resize_h'] = 0
        else:
            f.attrs['resize_w'] = int(resize[0])
            f.attrs['resize_h'] = int(resize[1])

        # ── observations ──────────────────────────────────────────────────
        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=qpos,   dtype='f4')
        obs.create_dataset('qvel', data=qvel,   dtype='f4')

        img_group = obs.create_group('images')
        for cam in cam_names:
            img_group.create_dataset(
                cam,
                data=np.stack(data["images"][cam]),
                dtype='u1',
                compression="gzip",
                compression_opts=2
            )

        # ── action + timestamp ─────────────────────────────────────────────
        f.create_dataset('action',    data=action, dtype='f4')
        f.create_dataset('timestamp', data=np.asarray(data["timestamps"], dtype=np.float64))

        # ── source timestamps ─────────────────────────────────────────────
        st = f.create_group("source_timestamps")
        for name, arr in data["source_times"].items():
            st.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

        # ── arm effort ────────────────────────────────────────────────────
        arm_effort = np.stack(data["arm_effort"])
        f.create_dataset('arm_effort', data=arm_effort, dtype='f4')

        # ── spatial measured_cf ────────────────────────────────────────────
        spatial_cf = np.stack(data["spatial_cf"])
        f.create_dataset('spatial_cf', data=spatial_cf, dtype='f4')

        # ── body measured_cf ───────────────────────────────────────────────
        # body_cf = np.stack(data["body_cf"])
        # f.create_dataset('body_cf', data=body_cf, dtype='f4')

        # ── measured_cf (external force sensor, optional) ─────────────────
        if data["measured_cf"]:
            measured_cf = np.stack(data["measured_cf"])
            f.create_dataset('measured_cf', data=measured_cf, dtype='f4')

        # ── joint_force (measured_js - gravity_compensation, optional) ────
        if data["joint_force"]:
            joint_force = np.stack(data["joint_force"])
            f.create_dataset('joint_force', data=joint_force, dtype='f4')

    print(f"Saved: {save_path}")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Record dVRK teleoperation episodes in ACT HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--output_dir', default='./dvrk_normal_dataset')

    parser.add_argument('--seconds_per_episode', type=float, default=15.0)
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Recording frequency (Hz). 10 Hz is a good ACT starting point.')

    parser.add_argument('--state_tolerance',   type=float, default=0.03,
                        help='PSM state anchor tolerance in seconds')
    parser.add_argument('--image_tolerance',   type=float, default=0.08,
                        help='Camera nearest-frame tolerance in seconds')
    parser.add_argument('--command_tolerance', type=float, default=0.05,
                        help='CMD command matching tolerance in seconds')

    parser.add_argument('--warmup_sec', type=float, default=0.0,
                        help='Queue warmup period before recording starts')

    # ── dVRK arm config ──────────────────────────────────────────────────
    parser.add_argument('--psm', type=str, default='PSM2',
                        choices=['PSM1', 'PSM2'])
    parser.add_argument('--arm_joints', type=int, default=6,
                        help='Number of PSM/CMD arm joints (excluding jaw)')
    parser.add_argument('--jaw_joints', type=int, default=1,
                        help='Number of PSM jaw / CMD jaw joints')

    # ── camera topics ─────────────────────────────────────────────────────
    parser.add_argument('--left_topic',   default='/console1/left/image_raw')
    parser.add_argument('--right_topic',  default='/console1/right/image_raw')
    parser.add_argument('--wrist1_topic', default='/wrist/cam1/image_raw')
    parser.add_argument('--wrist2_topic', default='/wrist/cam2/image_raw')
    parser.add_argument('--wrist_cameras', nargs='*', default=['cam1', 'cam2'],
                        choices=['cam1', 'cam2'],
                        help='Which wrist cameras to expect/record (e.g., --wrist_cameras cam1)')

    # ── image options ─────────────────────────────────────────────────────
    parser.add_argument('--resize_w', type=int, default=320)
    parser.add_argument('--resize_h', type=int, default=240)
    parser.add_argument('--image_encoding', choices=['bgr8', 'rgb8'], default='bgr8',
                        help='Must stay consistent with training and deployment')

    parser.add_argument('--command_mode',
                        choices=['latest_before_or_nearest', 'nearest'],
                        default='latest_before_or_nearest')

    # ── force/torque topics ──────────────────────────────────────────────
    parser.add_argument('--spatial_cf_topic', default='/PSM2/spatial/measured_cf',
                        help='ROS topic for spatial measured force/torque')
    parser.add_argument('--body_cf_topic', default='/PSM2/body/measured_cf',
                        help='ROS topic for body measured force/torque')
    parser.add_argument('--measured_cf_topic', default='/measured_cf',
                        help='ROS topic for external force sensor (set to "none" to disable)')
    parser.add_argument('--gravity_topic', default='/PSM2/gravity_compensation/setpoint_js',
                        help='ROS topic for gravity compensation setpoint (set to "none" to disable)')

    # ── optional rosbag backup ────────────────────────────────────────────
    parser.add_argument('--record_bag', action='store_true',
                        help='Also save a raw rosbag backup per episode')
    parser.add_argument('--bag_dir', default='./dvrk_bags')
    parser.add_argument('--force_sensor', action='store_true',
                        help='Also save external force sensor data to a separate rosbag')
    parser.add_argument('--force_topic', default='/measured_cf',
                        help='ROS topic for external force sensor messages (for force_bag)')
    parser.add_argument('--force_bag_dir', default='./dvrk_force_bags',
                        help='Directory for separate force-sensor rosbag backups')

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    resize = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize = (args.resize_w, args.resize_h)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.record_bag:
        os.makedirs(args.bag_dir, exist_ok=True)
    if args.force_sensor:
        os.makedirs(args.force_bag_dir, exist_ok=True)

    # derive dVRK topic names from --psm args
    psm_arm_topic  = f"/{args.psm}/measured_js"
    psm_jaw_topic  = f"/{args.psm}/jaw/measured_js"
    cmd_arm_topic  = f"/{args.psm}/setpoint_js"
    cmd_jaw_topic = f"/{args.psm}/jaw/setpoint_js"

    rclpy.init()

    node = DVRKRecorder(
        psm_arm_topic  = psm_arm_topic,
        psm_jaw_topic  = psm_jaw_topic,
        cmd_arm_topic  = cmd_arm_topic,
        cmd_jaw_topic = cmd_jaw_topic,
        left_topic     = args.left_topic,
        right_topic    = args.right_topic,
        wrist1_topic   = args.wrist1_topic if 'cam1' in args.wrist_cameras else 'none',
        wrist2_topic   = args.wrist2_topic if 'cam2' in args.wrist_cameras else 'none',
        spatial_cf_topic = args.spatial_cf_topic,
        # body_cf_topic    = args.body_cf_topic,
        measured_cf_topic = args.measured_cf_topic,
        gravity_topic    = args.gravity_topic,
        arm_joints     = args.arm_joints,
        jaw_joints     = args.jaw_joints,
        queue_size     = 800,
        image_encoding = args.image_encoding,
    )

    episode_idx = get_next_episode_idx(args.output_dir)
    print(f"Starting from episode index {episode_idx}")

    bag_topics = [
        psm_arm_topic, psm_jaw_topic,
        cmd_arm_topic, cmd_jaw_topic,
        args.left_topic, args.right_topic,
    ]
    if 'cam1' in args.wrist_cameras: bag_topics.append(args.wrist1_topic)
    if 'cam2' in args.wrist_cameras: bag_topics.append(args.wrist2_topic)

    try:
        while True:
            wait_for_key('q')

            print(f"\nRecording episode {episode_idx} ...")
            print("\a")   # terminal bell

            bag_proc = None
            force_bag_proc = None
            bag_path = None
            force_bag_path = None

            if args.record_bag:
                bag_path = os.path.join(args.bag_dir, f'episode_{episode_idx:03d}_bag')
                bag_proc = start_rosbag_recording(bag_path, bag_topics)

            if args.force_sensor:
                force_bag_path = os.path.join(args.force_bag_dir, f'episode_{episode_idx:03d}_force')
                force_bag_topics = [args.force_topic, args.gravity_topic, psm_arm_topic, args.spatial_cf_topic] #args.body_cf_topic,
                force_bag_proc = start_rosbag_recording(force_bag_path, force_bag_topics)

            if bag_proc is not None or force_bag_proc is not None:
                time.sleep(1.0)   # give rosbag a moment to start

            data = record_episode(
                node              = node,
                seconds           = args.seconds_per_episode,
                fps               = args.fps,
                state_tolerance   = args.state_tolerance,
                image_tolerance   = args.image_tolerance,
                command_tolerance = args.command_tolerance,
                resize            = resize,
                warmup_sec        = args.warmup_sec,
                command_mode      = args.command_mode,
            )

            if args.record_bag:
                stop_rosbag_recording(bag_proc)
            if args.force_sensor:
                stop_rosbag_recording(force_bag_proc)

            if len(data["qpos"]) < 2:
                print("Episode too short or too many dropped frames — discarding.")
                # Clean up rosbag files if episode is being discarded
                if force_bag_path is not None and os.path.exists(force_bag_path):
                    import shutil
                    shutil.rmtree(force_bag_path)
                    print(f"[force_bag] Discarded: {force_bag_path}")
                if bag_path is not None and os.path.exists(bag_path):
                    import shutil
                    shutil.rmtree(bag_path)
                    print(f"[rosbag] Discarded: {bag_path}")
                continue

            save_path = os.path.join(args.output_dir, f'episode_{episode_idx:03d}.hdf5')

            if os.path.exists(save_path):
                print(f"File already exists, skipping: {save_path}")
                episode_idx += 1
                continue

            save_episode(
                save_path      = save_path,
                data           = data,
                episode_idx    = episode_idx,
                topic_names    = node.topic_names,
                resize         = resize,
                image_encoding = args.image_encoding,
            )

            captured = data["meta"]["captured_steps"]
            requested = data["meta"]["requested_steps"]
            dropped   = data["meta"]["dropped_steps"]
            ratio     = data["meta"]["capture_ratio"]

            print(
                f"[Episode {episode_idx}] "
                f"captured={captured}/{requested}, "
                f"dropped={dropped}, "
                f"capture_ratio={ratio:.3f}"
            )

            if ratio < 0.90:
                print("[WARN] capture_ratio below 0.90. "
                      "Try relaxing --image_tolerance / --command_tolerance, "
                      "or lowering --fps.")

            episode_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
