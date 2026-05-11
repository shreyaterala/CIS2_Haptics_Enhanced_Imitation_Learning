# ACT-dVRK: Action Chunking with Transformers on the dVRK

ACT (Action Chunking with Transformers) adapted for the da Vinci Research Kit (dVRK) with real robot hardware.
This is the **baseline** variant — robot joint positions only, no haptic/force feedback.

> **ACT tuning tip:** If your policy is jerky or pauses mid-episode, train for longer.
> Success rate and smoothness can improve well after the loss plateaus.
> See the full [ACT tuning guide](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing).

---

## Table of Contents

1. [Model Architecture](#1-model-architecture)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Hardware Setup](#4-hardware-setup)
5. [Data Collection](#5-data-collection)
6. [Training](#6-training)
7. [Policy Deployment (Inference)](#7-policy-deployment-inference)

---

## 1. Model Architecture

ACT (Action Chunking with Transformers) is an imitation learning policy that predicts **chunks** of future actions rather than single-step actions. This reduces the effective horizon for compounding errors and enables smoother, more reactive behavior on real hardware.

### Key Components

| Component | Description |
|---|---|
| `policy.py` | Thin adapter that wraps the ACT model for the dVRK observation/action space |
| `detr/` | Transformer encoder–decoder (modified from DETR) that forms the core policy network |
| `imitate_episodes.py` | Training and evaluation entry point |
| `utils.py` | Dataset loading, normalization, and temporal ensembling helpers |
| `constants.py` | Task configs, camera names, state dimensions |
| `scripted_policy.py` | Scripted baseline policies for sanity-checking |

### How ACT Works

1. **Observation**: At each timestep the policy receives the current robot joint positions (14-dim for two PSMs) plus images from the wrist and endoscope cameras.
2. **Encoder (CVAE)**: During training a CVAE encodes the full action chunk into a latent style variable `z`. At inference `z` is sampled from the prior (zero mean).
3. **Transformer Decoder**: `z` is concatenated with the visual and proprioceptive tokens and decoded into a sequence of `chunk_size` joint-position targets.
4. **Temporal Ensembling**: Overlapping chunks are averaged with exponential weighting, smoothing transitions between chunks.
5. **State Dimension**: `STATE_DIM = 14` (7 joints × 2 PSMs). No force channels in this baseline variant.

---

## 2. Repository Structure

```
act_dvrk/
├── README.md
├── conda_env.yaml              # Conda environment spec
├── constants.py                # Task configs and state dimensions
├── imitate_episodes.py         # Training + evaluation script
├── policy.py                   # ACT policy wrapper
├── scripted_policy.py          # Scripted baseline policies
├── utils.py                    # Data loading and helper functions
├── detr/                       # ACT transformer model (modified from DETR)
├── data_recording/
│   ├── record_dvrk_episodes.py # Live data collection from dVRK
│   ├── filter_hdf5.py          # Filter / trim recorded HDF5 episodes
│   └── print_hdf5.py           # Inspect HDF5 episode contents
└── assets/                     # MuJoCo XML and other static assets
```

---

## 3. Environment Setup

### Prerequisites

- Ubuntu with ROS 2 Jazzy installed
- Conda / Miniconda
- dVRK console running (hardware PC)
- Wrist cameras plugged in via USB

### Installation

```bash
conda env create -f conda_env.yaml
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
cd act_dvrk/detr && pip install -e .
```

> **Note:** The environment is named `aloha` and uses Python 3.9 with PyTorch 2.0 + CUDA 11.8.

### Source ROS 2

Run this in **every terminal** before any of the steps below:

```bash
source /opt/ros/jazzy/setup.bash
```

### Multi-PC ROS 2 Networking (if using two computers)

Run the following block in **every terminal on both PCs**:

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION
export ROS_DOMAIN_ID=46
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
ros2 daemon stop
ros2 daemon start
```

---

## 4. Hardware Setup

### Step 1 — Set Up Wrist Cameras

Attach the wrist cameras to the PSM and plug them into a USB port, then verify they stream:

```bash
conda activate aloha
cd act_dvrk/
python wrist_cameras_video.py
```

### Step 2 — Launch Endoscope Cameras

In a new terminal:

```bash
ros2 launch /home/hzhao78/ros2_ws/src/dvrk/dvrk_video/ros2/launch/decklink_stereo_goovis.launch.py \
    stereo_rig_name:=console1
```

### Step 3 — Start dVRK Teleoperation

In a new terminal:

```bash
ros2 run dvrk_robot dvrk_system \
    -j ~/ros2_ws/src/dvrk/dvrk_config_jhu/jhu-dVRK/system-MTML-PSM1-PSM2-Teleop.json -C
```

---

## 5. Data Collection

> **Before recording:** Move the robot around briefly to ensure all ROS 2 topics are publishing before you start an episode.

### Record Episodes

```bash
conda activate aloha
cd act_dvrk/
python data_recording/record_dvrk_episodes.py \
    --output_dir ./data/dvrk_dataset \
    --seconds_per_episode 10 \
    --fps 20 \
    --wrist_cameras cam1
```

Key flags:
| Flag | Description |
|---|---|
| `--output_dir` | Directory where HDF5 episode files are saved |
| `--seconds_per_episode` | Duration of each episode (keep consistent across a dataset) |
| `--fps` | Recording frame rate (20 Hz recommended) |
| `--wrist_cameras` | Camera ID(s) for the wrist-mounted camera(s) |

### Inspect a Recorded Episode

```bash
python data_recording/print_hdf5.py --episode ./data/dvrk_dataset/episode_001.hdf5
```

### Filter / Trim Episodes

```bash
python data_recording/filter_hdf5.py \
    --input_dir ./data/dvrk_dataset \
    --output_dir ./data/dvrk_dataset_filtered
```

> **Notes:**
> - Make sure teleoperation is **off** if running bilateral teleop code separately.
> - If more than ~10 frames are being dropped, kill all processes and restart.

---

## 6. Training

### Train the ACT Policy

```bash
conda activate aloha
cd act_dvrk/
python imitate_episodes.py \
    --task_name dvrk_retraction \
    --ckpt_dir ./checkpoints/dvrk_retraction \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 3000 \
    --lr 1e-5 \
    --state_dim 14 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --kl_weight 10 \
    --temporal_agg
```

### Key Training Flags

| Flag | Default | Description |
|---|---|---|
| `--task_name` | — | Must match a key in `constants.py` `SIM_TASK_CONFIGS` |
| `--ckpt_dir` | — | Directory to save checkpoints |
| `--state_dim` | 14 | Observation dimension (14 for two PSMs, joints only) |
| `--chunk_size` | 100 | Number of future actions predicted per forward pass |
| `--hidden_dim` | 512 | Transformer hidden dimension |
| `--dim_feedforward` | 3200 | Transformer feedforward dimension |
| `--kl_weight` | 10 | Weight of the CVAE KL-divergence loss term |
| `--num_epochs` | 2000 | Training epochs (use ≥5000 for real-world data) |
| `--lr` | 1e-5 | Learning rate |
| `--temporal_agg` | off | Enable temporal ensembling at inference |

> **Tip:** For real-world data, train for at least 5000 epochs or 3–4× the epoch at which the loss plateaued.

### Evaluate the Policy (Simulation)

Add `--eval` to load the best validation checkpoint and run rollouts:

```bash
python imitate_episodes.py \
    --task_name dvrk_retraction \
    --ckpt_dir ./checkpoints/dvrk_retraction \
    --policy_class ACT \
    --state_dim 14 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --temporal_agg \
    --eval
```

---

## 7. Policy Deployment (Inference)

Run inference on the real dVRK robot:

```bash
conda activate aloha
cd act_dvrk/
python run_dvrk_act.py \
    --ckpt_dir ./checkpoints/dvrk_retraction \
    --task_name dvrk_retraction \
    --state_dim 14 \
    --chunk_size 100 \
    --fps 20 \
    --temporal_agg
```

> **Before deploying:** Confirm all camera topics are publishing and the dVRK system is in the correct state.
> Move the robot to its start pose manually before running inference.
