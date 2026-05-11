# ACT-dVRK-JF: Action Chunking with Transformers + Joint-Space Force Feedback

ACT adapted for the dVRK with **6-DOF joint-space force feedback**.
This variant appends a full 6-DOF Cartesian wrench (3 forces + 3 torques) to the state, enabling the policy to respond to richer haptic information during tissue interaction.

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

ACT predicts **chunks** of future actions in one forward pass, reducing compounding errors and enabling smooth real-robot trajectories.

### Haptics Extension — Joint-Space Force (6-DOF)

This variant extends the state with a **full 6-DOF wrench** (Fx, Fy, Fz, Tx, Ty, Tz):

| Signal | Dim | Source |
|---|---|---|
| PSM joint positions (2 arms) | 14 | dVRK |
| 6-DOF wrench (force + torque) | 6 | ATI NetFT (`/measured_cf`) |
| **Total state** | **20** | — |

The additional torque channels give the policy information about rotational contact forces, which is especially useful for tasks involving grasping or tool rotation against tissue.

### Key Components

| File | Role |
|---|---|
| `policy.py` | ACT wrapper; `CF_DIM = 6` from `constants.py` |
| `detr/` | Transformer encoder–decoder |
| `imitate_episodes.py` | Training & evaluation |
| `utils.py` | Data loading with 6-DOF force channels |
| `constants.py` | `STATE_DIM=14`, `CF_DIM=6`, extended task configs |
| `data_recording/hdf5_filter_with_cf_align.py` | Force timestamp alignment |

---

## 2. Repository Structure

```
act_dvrk_jf/
├── README.md
├── conda_env.yaml
├── constants.py                      # STATE_DIM=14, CF_DIM=6
├── imitate_episodes.py
├── policy.py
├── scripted_policy.py
├── utils.py
├── detr/
├── data_recording/
│   ├── record_dvrk_episodes.py
│   ├── hdf5_filter_with_cf_align.py  # Aligns 6-DOF wrench timestamps
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
cd act_dvrk_jf/detr && pip install -e .
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
conda activate aloha && cd act_dvrk_jf/
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

> Re-bias the sensor at startup to zero gravity/offset. Verify with: `ros2 topic echo /measured_cf`

---

## 5. Data Collection

### Record Episodes (with 6-DOF wrench)

```bash
conda activate aloha && cd act_dvrk_jf/
python data_recording/record_dvrk_episodes.py \
    --output_dir ./data/dvrk_dataset_jf \
    --seconds_per_episode 10 \
    --fps 20 \
    --wrist_cameras cam1 \
    --force_sensor
```

The recording script captures all 6 wrench channels (3 forces + 3 torques).

### Align Force Timestamps

```bash
python data_recording/hdf5_filter_with_cf_align.py \
    --input_dir ./data/dvrk_dataset_jf \
    --output_dir ./data/dvrk_dataset_jf_aligned
```

### Inspect Episodes

```bash
python data_recording/print_hdf5.py --episode ./data/dvrk_dataset_jf_aligned/episode_001.hdf5
```

> Move the robot briefly before recording to ensure all topics are live. If >10 frames drop, restart all terminals.

---

## 6. Training

```bash
conda activate aloha && cd act_dvrk_jf/
python imitate_episodes.py \
    --task_name dvrk_retraction_u_cf \
    --ckpt_dir ./checkpoints/dvrk_retraction_jf \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 3000 \
    --lr 1e-5 \
    --state_dim 20 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --kl_weight 10 \
    --temporal_agg
```

> `--state_dim 20` = 14 joint positions + 6 wrench channels (`CF_DIM=6`).  
> Use ≥ 5000 epochs for real-world data.

### Key Training Flags

| Flag | Value | Description |
|---|---|---|
| `--task_name` | `dvrk_retraction_u_cf` | Matches key in `constants.py` |
| `--state_dim` | 20 | 14 joints + 6 wrench channels |
| `--chunk_size` | 100 | Action chunk length |
| `--kl_weight` | 10 | CVAE regularization strength |
| `--temporal_agg` | flag | Enable temporal ensembling |

---

## 7. Policy Deployment

```bash
conda activate aloha && cd act_dvrk_jf/
python run_dvrk_act.py \
    --ckpt_dir ./checkpoints/dvrk_retraction_jf \
    --task_name dvrk_retraction_u_cf \
    --state_dim 20 \
    --chunk_size 100 \
    --fps 20 \
    --temporal_agg
```

> Ensure the ATI sensor is running and re-biased before inference. The full 6-DOF wrench must be publishing on `/measured_cf`.
