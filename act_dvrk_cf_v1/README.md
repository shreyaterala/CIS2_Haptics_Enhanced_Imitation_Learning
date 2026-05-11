# ACT-dVRK-CF-v1: Action Chunking with Transformers + Cartesian Force (v1)

ACT adapted for the dVRK with **3-DOF Cartesian force feedback** from the ATI NetFT sensor.
The state vector is augmented with the tool-tip force reading (x, y, z), allowing the policy to condition on tissue-contact forces during surgical tasks.

> **Tuning tip:** Jerky or pausing policies typically just need more training epochs — success rate and smoothness improve well past loss plateau. See the [ACT tuning guide](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing).

---

## Table of Contents

1. [Model Architecture](#1-model-architecture)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Hardware Setup](#4-hardware-setup)
5. [Data Collection](#5-data-collection)
6. [Training](#6-training)
7. [Policy Deployment](#7-policy-deployment)

---

## 1. Model Architecture

ACT predicts **chunks** of future actions in a single forward pass to reduce compounding errors and enable smooth real-robot control.

### Haptics Extension — Cartesian Force (v1)

| Signal | Dim | Source |
|---|---|---|
| PSM joint positions (2 arms) | 14 | dVRK |
| Cartesian force (x, y, z) | 3 | ATI NetFT (`/measured_cf`) |
| **Total state** | **17** | — |

The 3-DOF force reading is timestep-aligned with joint and image data via `hdf5_filter_with_cf_align.py` before training.

### Key Components

| File | Role |
|---|---|
| `policy.py` | ACT wrapper; uses `CF_DIM = 3` from `constants.py` |
| `detr/` | Transformer encoder–decoder (modified DETR) |
| `imitate_episodes.py` | Training & evaluation entry point |
| `utils.py` | Data loading with force-channel support |
| `constants.py` | `STATE_DIM=14`, `CF_DIM=3`, task configs |
| `data_recording/hdf5_filter_with_cf_align.py` | Force timestamp alignment |

---

## 2. Repository Structure

```
act_dvrk_cf_v1/
├── README.md
├── conda_env.yaml
├── constants.py                      # STATE_DIM=14, CF_DIM=3
├── imitate_episodes.py
├── policy.py
├── scripted_policy.py
├── utils.py
├── detr/
├── data_recording/
│   ├── record_dvrk_episodes.py
│   ├── hdf5_filter_with_cf_align.py  # Align ATI timestamps to episode frames
│   └── print_hdf5.py
└── assets/
```

---

## 3. Environment Setup

```bash
conda env create -f conda_env.yaml
conda activate aloha
pip install torchvision torch pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.7 dm_control==1.0.14
pip install opencv-python matplotlib einops packaging h5py ipython
cd act_dvrk_cf_v1/detr && pip install -e .
```

Source ROS 2 in **every terminal**:

```bash
source /opt/ros/jazzy/setup.bash
```

**Multi-PC networking:**

```bash
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION
export ROS_DOMAIN_ID=46
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
ros2 daemon stop && ros2 daemon start
```

---

## 4. Hardware Setup

**Terminal 1** — Wrist cameras:
```bash
conda activate aloha && cd act_dvrk_cf_v1/
python wrist_cameras_video.py
```

**Terminal 2** — Endoscope cameras:
```bash
ros2 launch /home/hzhao78/ros2_ws/src/dvrk/dvrk_video/ros2/launch/decklink_stereo_goovis.launch.py stereo_rig_name:=console1
```

**Terminal 3** — dVRK teleoperation:
```bash
ros2 run dvrk_robot dvrk_system \
    -j ~/ros2_ws/src/dvrk/dvrk_config_jhu/jhu-dVRK/system-MTML-PSM1-PSM2-Teleop.json -C
```

**Terminal 4** — ATI force sensor:
```bash
ros2 run atinetft_ros atinetft_ros2_node -i 192.168.1.1
```

> Re-bias the sensor at startup. Verify with: `ros2 topic echo /measured_cf`

---

## 5. Data Collection

### Record Episodes

```bash
conda activate aloha && cd act_dvrk_cf_v1/
python data_recording/record_dvrk_episodes.py \
    --output_dir ./data/dvrk_dataset_cf \
    --seconds_per_episode 10 \
    --fps 20 \
    --wrist_cameras cam1 \
    --force_sensor
```

`--force_sensor` saves the `/measured_cf` topic alongside joint/image data.

### Align Force Timestamps

```bash
python data_recording/hdf5_filter_with_cf_align.py \
    --input_dir ./data/dvrk_dataset_cf \
    --output_dir ./data/dvrk_dataset_cf_aligned
```

### Inspect Episodes

```bash
python data_recording/print_hdf5.py --episode ./data/dvrk_dataset_cf_aligned/episode_001.hdf5
```

> Move the robot briefly before recording to ensure all topics are live. If >10 frames drop, restart everything.

---

## 6. Training

```bash
conda activate aloha && cd act_dvrk_cf_v1/
python imitate_episodes.py \
    --task_name dvrk_retraction_cf \
    --ckpt_dir ./checkpoints/dvrk_retraction_cf \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 3000 \
    --lr 1e-5 \
    --state_dim 17 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --kl_weight 10 \
    --temporal_agg
```

> `--state_dim 17` = 14 joint positions + 3 Cartesian force channels.  
> For real-world data use ≥ 5000 epochs.

---

## 7. Policy Deployment

```bash
conda activate aloha && cd act_dvrk_cf_v1/
python run_dvrk_act.py \
    --ckpt_dir ./checkpoints/dvrk_retraction_cf \
    --task_name dvrk_retraction_cf \
    --state_dim 17 \
    --chunk_size 100 \
    --fps 20 \
    --temporal_agg
```

> Ensure the ATI sensor is running and re-biased before inference.
