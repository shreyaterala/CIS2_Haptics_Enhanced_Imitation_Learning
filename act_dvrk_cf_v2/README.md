# ACT-dVRK-CF-v2: Action Chunking with Transformers + Cartesian Force Feedback (v2)

ACT adapted for the dVRK with **refined 3-DOF Cartesian force feedback**.
This is the second-generation Cartesian-force variant; it uses the same `hdf5_filter_with_cf_align.py` pipeline as v1 but incorporates updated data filtering and alignment logic for improved force-signal quality.

> **Tuning tip:** Jerky or pausing policies typically need more epochs. See the [ACT tuning guide](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing).

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

ACT predicts **chunks** of future actions in one forward pass to reduce compounding errors and produce smooth real-robot trajectories.

### Haptics Extension — Cartesian Force (v2)

This variant conditions the policy on 3-DOF Cartesian forces, identical in structure to v1 but with improved timestamp alignment and filtering in the data pipeline:

| Signal | Dim | Source |
|---|---|---|
| PSM joint positions (2 arms) | 14 | dVRK |
| Cartesian force (x, y, z) | 3 | ATI NetFT (`/measured_cf`) |
| **Total state** | **17** | — |

The v2 improvements over v1 are in `hdf5_filter_with_cf_align.py`: better interpolation and outlier rejection for the force signal.

### Key Components

| File | Role |
|---|---|
| `policy.py` | ACT wrapper; `CF_DIM = 3` |
| `detr/` | Transformer encoder–decoder |
| `imitate_episodes.py` | Training & evaluation |
| `utils.py` | Data loading with force channels |
| `constants.py` | `STATE_DIM=14`, `CF_DIM=3` |
| `data_recording/hdf5_filter_with_cf_align.py` | Improved force alignment (v2) |

---

## 2. Repository Structure

```
act_dvrk_cf_v2/
├── README.md
├── conda_env.yaml
├── constants.py
├── imitate_episodes.py
├── policy.py
├── scripted_policy.py
├── utils.py
├── detr/
├── data_recording/
│   ├── record_dvrk_episodes.py
│   ├── hdf5_filter_with_cf_align.py  # v2 alignment logic
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
cd act_dvrk_cf_v2/detr && pip install -e .
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
conda activate aloha && cd act_dvrk_cf_v2/
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
conda activate aloha && cd act_dvrk_cf_v2/
python data_recording/record_dvrk_episodes.py \
    --output_dir ./data/dvrk_dataset_cf_v2 \
    --seconds_per_episode 10 \
    --fps 20 \
    --wrist_cameras cam1 \
    --force_sensor
```

### Align & Filter Force Timestamps (v2 pipeline)

```bash
python data_recording/hdf5_filter_with_cf_align.py \
    --input_dir ./data/dvrk_dataset_cf_v2 \
    --output_dir ./data/dvrk_dataset_cf_v2_aligned
```

### Inspect Episodes

```bash
python data_recording/print_hdf5.py --episode ./data/dvrk_dataset_cf_v2_aligned/episode_001.hdf5
```

> Move the robot briefly before recording to ensure all topics are live. If >10 frames drop, restart all terminals.

---

## 6. Training

```bash
conda activate aloha && cd act_dvrk_cf_v2/
python imitate_episodes.py \
    --task_name dvrk_retraction_cf \
    --ckpt_dir ./checkpoints/dvrk_retraction_cf_v2 \
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
> Use ≥ 5000 epochs for real-world data.

---

## 7. Policy Deployment

```bash
conda activate aloha && cd act_dvrk_cf_v2/
python run_dvrk_act.py \
    --ckpt_dir ./checkpoints/dvrk_retraction_cf_v2 \
    --task_name dvrk_retraction_cf \
    --state_dim 17 \
    --chunk_size 100 \
    --fps 20 \
    --temporal_agg
```

> Ensure the ATI sensor is running and re-biased before inference.
