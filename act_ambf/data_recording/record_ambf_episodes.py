#!/usr/bin/env python3
# record_ambf_episodes.py  (修正版 v2)
#
# 核心改动：
#   1. 完全移除 Command topic 订阅
#   2. 新增 jaw 订阅，来自 /MTML/gripper/measured_js
#   3. qpos 存 7 维：[j0,j1,j2,j3,j4,j5,jaw]
#   4. action = qpos[t+1]，不依赖 Command topic
#   5. 不做任何单位换算（SimToSI.linear_factor=1.0，RigidBodyState 和
#      servo_jp 用的是同一套单位，不需要转换）
#
# 用法：
#   python record_ambf_episodes.py --output_dir ./ambf_dataset --fps 20

import os
import sys
import cv2
import time
import h5py
import select
import argparse
import termios
import tty
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from ambf_msgs.msg import RigidBodyState


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
                idx = int(name[len("episode_"):-len(".hdf5")])
                existing.append(idx)
            except ValueError:
                pass
    return 0 if not existing else max(existing) + 1


def msg_time_to_sec(msg):
    return time.time()


def nearest_from_queue(q, target_t, tolerance):
    if not q:
        return None
    best = None
    best_dt = None
    for ts, payload in q:
        dt = abs(ts - target_t)
        if best is None or dt < best_dt:
            best = (ts, payload)
            best_dt = dt
    if best is None or best_dt > tolerance:
        return None
    return best


def gripper_msg_to_jaw(msg):
    """
    与官方 mtm_device_crtk.py gripper_cb() 完全一致。
    """
    min_val = -0.1
    max_val = 0.51
    jaw_angle = msg.position[0] + min_val / (max_val - min_val)
    return float(jaw_angle)


def safe_array_from_field(msg, field_name):
    if not hasattr(msg, field_name):
        return None
    value = getattr(msg, field_name)
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return arr
    except Exception:
        try:
            if hasattr(value, "data"):
                arr = np.asarray(value.data, dtype=np.float32).reshape(-1)
                return arr
        except Exception:
            return None
    return None


# ============================================================
# Recorder Node
# ============================================================

class AMBFRecorder(Node):
    def __init__(self,
                 state_topic,
                 jaw_topic,
                 left_topic,
                 right_topic,
                 wrist_topic,
                 queue_size=600,
                 image_encoding='bgr8'):
        super().__init__('ambf_recorder')

        self.bridge        = CvBridge()
        self.queue_size    = queue_size
        self.image_encoding = image_encoding

        self.state_q = deque(maxlen=queue_size)
        self.jaw_q   = deque(maxlen=queue_size)
        self.left_q  = deque(maxlen=queue_size)
        self.right_q = deque(maxlen=queue_size)
        self.wrist_q = deque(maxlen=queue_size)

        self.topic_names = {
            "state_topic": state_topic,
            "jaw_topic":   jaw_topic,
            "left_topic":  left_topic,
            "right_topic": right_topic,
            "wrist_topic": wrist_topic,
        }

        self.create_subscription(RigidBodyState, state_topic, self.state_cb, 50)
        self.create_subscription(JointState,     jaw_topic,   self.jaw_cb,   50)
        self.create_subscription(Image, left_topic,  self.left_cb,  20)
        self.create_subscription(Image, right_topic, self.right_cb, 20)
        self.create_subscription(Image, wrist_topic, self.wrist_cb, 20)

        self.get_logger().info("AMBFRecorder initialized (v2 - no unit conversion)")
        self.get_logger().info(f"  state : {state_topic}")
        self.get_logger().info(f"  jaw   : {jaw_topic}")

    # ── callbacks ──────────────────────────────────────────────

    def state_cb(self, msg):
        self.state_q.append((msg_time_to_sec(msg), msg))

    def jaw_cb(self, msg):
        if len(msg.position) > 0:
            self.jaw_q.append((msg_time_to_sec(msg), gripper_msg_to_jaw(msg)))

    def left_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
        self.left_q.append((msg_time_to_sec(msg), img))

    def right_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
        self.right_q.append((msg_time_to_sec(msg), img))

    def wrist_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
        self.wrist_q.append((msg_time_to_sec(msg), img))

    # ── extraction ─────────────────────────────────────────────

    def extract_qpos_qvel(self, state_msg):
        """
        提取前6个运动关节的位置和速度。
        不做任何单位换算：SimToSI.linear_factor=1.0，
        RigidBodyState 和 servo_jp 使用相同单位。
        """
        jp = safe_array_from_field(state_msg, "joint_positions")
        jv = safe_array_from_field(state_msg, "joint_velocities")

        if jp is None:
            jp = safe_array_from_field(state_msg, "joint_position")
        if jv is None:
            jv = safe_array_from_field(state_msg, "joint_velocity")

        if jp is None or jv is None:
            raise RuntimeError("无法从 State 消息提取 joint_positions/joint_velocities")

        # 只取前6个运动关节，不做单位换算
        qpos = jp[:6].copy()
        qvel = jv[:6].copy()

        return qpos, qvel

    def resize_images(self, imgs, resize=None):
        if resize is None:
            return imgs
        w, h = resize
        return {k: cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                for k, img in imgs.items()}

    def get_synced_sample(self, anchor_t, state_tolerance,
                          image_tolerance, jaw_tolerance, resize=None):
        state_item = nearest_from_queue(self.state_q, anchor_t, state_tolerance)
        if state_item is None:
            return None

        state_ts, state_msg = state_item

        jaw_item   = nearest_from_queue(self.jaw_q,   state_ts, jaw_tolerance)
        left_item  = nearest_from_queue(self.left_q,  state_ts, image_tolerance)
        right_item = nearest_from_queue(self.right_q, state_ts, image_tolerance)
        wrist_item = nearest_from_queue(self.wrist_q, state_ts, image_tolerance)

        if any(x is None for x in [jaw_item, left_item, right_item, wrist_item]):
            return None

        jaw_ts,   jaw_val   = jaw_item
        left_ts,  left_img  = left_item
        right_ts, right_img = right_item
        wrist_ts, wrist_img = wrist_item

        qpos_6, qvel_6 = self.extract_qpos_qvel(state_msg)

        # 合并为7维：[j0,j1,j2,j3,j4,j5, jaw]
        qpos = np.append(qpos_6, float(jaw_val)).astype(np.float32)
        qvel = np.append(qvel_6, 0.0).astype(np.float32)

        imgs = {
            "left":  left_img.copy(),
            "right": right_img.copy(),
            "wrist": wrist_img.copy(),
        }
        imgs = self.resize_images(imgs, resize=resize)

        return {
            "sample_time": float(state_ts),
            "source_times": {
                "state": float(state_ts),
                "jaw":   float(jaw_ts),
                "left":  float(left_ts),
                "right": float(right_ts),
                "wrist": float(wrist_ts),
            },
            "qpos":   qpos,
            "qvel":   qvel,
            "images": imgs,
        }


# ============================================================
# Episode recording
# ============================================================

def record_episode(node, seconds, fps, state_tolerance,
                   image_tolerance, jaw_tolerance, resize=None, warmup_sec=1.0):

    dt            = 1.0 / fps
    qpos_list     = []
    qvel_list     = []
    timestamps    = []
    source_times  = {"state": [], "jaw": [], "left": [], "right": [], "wrist": []}
    images        = {"left": [], "right": [], "wrist": []}

    t0 = time.time()
    while time.time() - t0 < warmup_sec:
        rclpy.spin_once(node, timeout_sec=0.01)

    wall_start   = time.time()
    sample_start = wall_start
    requested    = int(round(seconds * fps))
    dropped      = 0
    accepted     = 0

    for k in range(requested):
        target_t = sample_start + k * dt

        while True:
            if time.time() >= target_t:
                break
            rclpy.spin_once(node, timeout_sec=min(0.003, target_t - time.time()))

        settle = target_t + min(0.005, dt * 0.25)
        while time.time() < settle:
            rclpy.spin_once(node, timeout_sec=0.001)

        sample = node.get_synced_sample(
            anchor_t        = target_t,
            state_tolerance = state_tolerance,
            image_tolerance = image_tolerance,
            jaw_tolerance   = jaw_tolerance,
            resize          = resize,
        )

        if sample is None:
            dropped += 1
            continue

        qpos_list.append(sample["qpos"])
        qvel_list.append(sample["qvel"])
        timestamps.append(sample["sample_time"])
        for name in source_times:
            source_times[name].append(sample["source_times"][name])
        for cam in images:
            images[cam].append(sample["images"][cam])
        accepted += 1

    capture_ratio = accepted / requested if requested > 0 else 0.0

    return {
        "qpos":         qpos_list,
        "qvel":         qvel_list,
        "images":       images,
        "timestamps":   timestamps,
        "source_times": source_times,
        "meta": {
            "requested_seconds":   float(seconds),
            "requested_fps":       float(fps),
            "requested_steps":     int(requested),
            "captured_steps":      int(accepted),
            "dropped_steps":       int(dropped),
            "capture_ratio":       float(capture_ratio),
            "state_tolerance_sec": float(state_tolerance),
            "image_tolerance_sec": float(image_tolerance),
            "jaw_tolerance_sec":   float(jaw_tolerance),
            "wall_start_time":     float(wall_start),
            "wall_end_time":       float(time.time()),
            "warmup_sec":          float(warmup_sec),
            "joint_dim":           7,
        }
    }


# ============================================================
# Save HDF5
# ============================================================

def save_episode(save_path, data, episode_idx, topic_names, resize, image_encoding):
    qpos   = np.stack(data["qpos"])                          # (T, 7)
    qvel   = np.stack(data["qvel"])                          # (T, 7)
    action = np.vstack([qpos[1:], qpos[-1:]]).astype(np.float32)  # (T, 7)

    with h5py.File(save_path, 'w') as f:

        f.attrs['sim']             = True
        f.attrs['source']          = 'ambf'
        f.attrs['episode_idx']     = int(episode_idx)
        f.attrs['fps']             = float(data["meta"]["requested_fps"])
        f.attrs['dt']              = 1.0 / float(data["meta"]["requested_fps"])
        f.attrs['requested_steps'] = int(data["meta"]["requested_steps"])
        f.attrs['captured_steps']  = int(data["meta"]["captured_steps"])
        f.attrs['dropped_steps']   = int(data["meta"]["dropped_steps"])
        f.attrs['capture_ratio']   = float(data["meta"]["capture_ratio"])
        f.attrs['joint_dim']       = 7
        f.attrs['joint_names']     = np.array(
            [b'yaw', b'pitch', b'insertion', b'tool_roll',
             b'tool_pitch', b'tool_yaw', b'jaw'], dtype='S')
        f.attrs['wall_start_time'] = float(data["meta"]["wall_start_time"])
        f.attrs['wall_end_time']   = float(data["meta"]["wall_end_time"])
        f.attrs['recorded_at_iso'] = datetime.now().isoformat(timespec="seconds")
        f.attrs['camera_names']    = np.array(["left", "right", "wrist"], dtype='S')
        f.attrs['color_order']     = str(image_encoding)

        # 单位说明：无换算，与 servo_jp 直接兼容
        f.attrs['state_semantics']  = 'joint_positions[0:6] + jaw, no unit conversion'
        f.attrs['action_semantics'] = 'qpos[t+1]'
        f.attrs['unit_conversion']  = 'none (SimToSI.linear_factor=1.0)'

        for k, v in topic_names.items():
            f.attrs[k] = v

        if resize is None:
            f.attrs['resize_w'] = 0
            f.attrs['resize_h'] = 0
        else:
            f.attrs['resize_w'] = int(resize[0])
            f.attrs['resize_h'] = int(resize[1])

        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=qpos, dtype='f4')
        obs.create_dataset('qvel', data=qvel, dtype='f4')

        img_grp = obs.create_group('images')
        for cam in ["left", "right", "wrist"]:
            img_grp.create_dataset(
                cam,
                data=np.stack(data["images"][cam]),
                dtype='u1',
                compression="gzip",
                compression_opts=2,
            )

        f.create_dataset('action',    data=action, dtype='f4')
        f.create_dataset('timestamp', data=np.asarray(data["timestamps"], dtype=np.float64))

        st = f.create_group("source_timestamps")
        for name, arr in data["source_times"].items():
            st.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

    print(f"Saved: {save_path}  (steps={qpos.shape[0]}, joint_dim=7)")


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--output_dir',          default='./ambf_dataset')
    p.add_argument('--seconds_per_episode', type=float, default=10.0)
    p.add_argument('--fps',                 type=float, default=20.0)
    p.add_argument('--state_tolerance',     type=float, default=0.03)
    p.add_argument('--image_tolerance',     type=float, default=0.05)
    p.add_argument('--jaw_tolerance',       type=float, default=0.05)
    p.add_argument('--warmup_sec',          type=float, default=0)

    p.add_argument('--state_topic',
                   default='/ambf/env/psm1/baselink/State')
    p.add_argument('--jaw_topic',
                   default='/MTML/gripper/measured_js')
    p.add_argument('--left_topic',
                   default='/ambf/env/cameras/cameraL/ImageData')
    p.add_argument('--right_topic',
                   default='/ambf/env/cameras/cameraR/ImageData')
    p.add_argument('--wrist_topic',
                   default='/ambf/env/cameras/Wrist1Frame/ImageData')

    p.add_argument('--resize_w',       type=int,  default=320)
    p.add_argument('--resize_h',       type=int,  default=240)
    p.add_argument('--image_encoding', choices=['bgr8', 'rgb8'], default='bgr8')

    p.add_argument('--start_pose_file', default='./start_pose.json',
                   help='Pose file used when pressing "r" to reset PSM '
                        '(default: ./start_pose.json). '
                        'Create with: python set_start_pose.py --save')
    p.add_argument('--psm_name', default='psm1',
                   help='PSM name used for reset (default: psm1)')

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    resize = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize = (args.resize_w, args.resize_h)

    os.makedirs(args.output_dir, exist_ok=True)

    rclpy.init()

    node = AMBFRecorder(
        state_topic    = args.state_topic,
        jaw_topic      = args.jaw_topic,
        left_topic     = args.left_topic,
        right_topic    = args.right_topic,
        wrist_topic    = args.wrist_topic,
        queue_size     = 800,
        image_encoding = args.image_encoding,
    )

    episode_idx = get_next_episode_idx(args.output_dir)
    print(f"Starting from episode index {episode_idx}")
    print(f"Joint layout: [j0_yaw, j1_pitch, j2_insertion, j3_roll, j4_tpitch, j5_tyaw, jaw]")
    print(f"Unit conversion: none (SimToSI.linear_factor=1.0)")

    try:
        while True:
            wait_for_key('q')
            print(f"\nRecording episode {episode_idx} ...")

            data = record_episode(
                node            = node,
                seconds         = args.seconds_per_episode,
                fps             = args.fps,
                state_tolerance = args.state_tolerance,
                image_tolerance = args.image_tolerance,
                jaw_tolerance   = args.jaw_tolerance,
                resize          = resize,
                warmup_sec      = args.warmup_sec,
            )

            if len(data["qpos"]) < 2:
                print("Episode too short or too many frames dropped.")
                continue

            save_path = os.path.join(
                args.output_dir, f'episode_{episode_idx:03d}.hdf5')

            if os.path.exists(save_path):
                print(f"File already exists, skip: {save_path}")
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

            meta = data["meta"]
            print(
                f"[Episode {episode_idx}] "
                f"captured={meta['captured_steps']}/{meta['requested_steps']}, "
                f"dropped={meta['dropped_steps']}, "
                f"ratio={meta['capture_ratio']:.3f}"
            )
            if meta["capture_ratio"] < 0.90:
                print("[WARN] capture_ratio below 0.90 - consider lowering fps or relaxing tolerance")

            episode_idx += 1

            # Prompt after each episode
            print(f"\nEpisode {episode_idx-1} saved.")
            print(f"  Press 'r' to show start pose reference (use MTM to move there)")
            print(f"  Press 'q' to start next recording immediately")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()