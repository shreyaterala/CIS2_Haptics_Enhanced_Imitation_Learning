# ACT-AMBF: Haptics-Enhanced Imitation Learning

ACT (Action Chunking with Transformers) adapted for the AMBF simulation environment with dVRK hardware.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Multi-PC ROS 2 Networking](#2-multi-pc-ros-2-networking)
3. [AMBF Simulation](#3-ambf-simulation)
4. [Data Recording](#4-data-recording)
5. [Data Verification](#5-data-verification)
6. [Training](#6-training)
7. [Policy Deployment (Inference)](#7-policy-deployment-inference)

---

## 1. Environment Setup

### Prerequisites

- Ubuntu with ROS 2 Jazzy installed
- Conda / Miniconda
- AMBF simulator installed and sourced
- dVRK console (on the hardware PC)

### Create the Conda Environment

```bash
conda env create -f conda_env.yaml
conda activate aloha
```

> **Note:** The environment is named `aloha` and uses Python 3.9 with PyTorch 2.0 + CUDA 11.8.

### Install Additional pip Dependencies

After activating the environment, install any remaining packages:

```bash
pip install -e .
```

> If no `setup.py` / `pyproject.toml` is present, skip this step.

### Source ROS 2

Run this in **every terminal** before executing any of the steps below:

```bash
source /opt/ros/jazzy/setup.bash
```

---

## 2. Multi-PC ROS 2 Networking

Run the following block in **every terminal on both PCs** to configure cross-machine ROS2 discovery if using the dVRK Classic computer:

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION
export ROS_DOMAIN_ID=46
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
ros2 daemon stop
ros2 daemon start
```

---

## 3. AMBF Simulation

### Terminal 1 — Launch AMBF Environment

Navigate to the surgical robotics challenge directory and launch one of the environments:

```bash
cd surgical_robotics_challenge/

# Choose one:
./run_env_3D_MED_STRAIGHT_LND_420006.sh
./run_env_SIMPLE_LND_420006.sh
./run_env_3D_MED_COMPLEX_LND_420006.sh
```

### Terminal 2 — Start dVRK Hardware Console (Hardware PC only)

On the **hardware PC**:

1. Start the dVRK console
2. Start the MTM pair console

### Verify Camera Topics

To confirm camera streams are publishing correctly:

```bash
ros2 topic hz /ambf/env/cameras/Wrist1Frame/ImageData
ros2 topic hz /ambf/env/cameraR/ImageData
ros2 topic hz /ambf/env/cameraL/ImageData
ros2 topic list | grep cameras
```

---

## 4. Data Recording

> **Important:** Run the ROS 2 networking block from [Section 2](#2-multi-pc-ros-2-networking) in every terminal before proceeding.

### Step 1 — Set the Start Pose (optional but recommended)

Before recording, capture and save the robot start pose:

```bash
cd act_ambf/
python policy_deployment_scripts/set_start_pose.py --output ./policy_deployment_scripts/start_pose.json
```

### Step 2 — Record Episodes

```bash
cd act_ambf/
python data_collection_scripts/record_ambf_episodes.py \
    --output_dir ./data/ambf_dataset_new \
    --fps 20 \
    --seconds_per_episode 7 \
    --resize_w 320 \
    --resize_h 240
```

> **Tip:** Change `--seconds_per_episode` to match your task duration. Ensure **all episodes in a dataset use the same length**.

---

## 5. Data Verification

### View a Recorded Episode

```bash
# View all camera streams
python view_episode.py --episode ./data/ambf_dataset_new/episode_001.hdf5 --all_cams

# View a specific camera (e.g., wrist)
python view_episode.py --episode ./data/ambf_dataset_new/episode_001.hdf5 --cam wrist

# Print metadata / joint info only
python view_episode.py --episode ./data/ambf_dataset_new/episode_001.hdf5 --info
```

### Replay an Episode Inside AMBF

> Close everything on the dVRK hardware PC before replaying — only the simulation PC is needed.

```bash
# Replay joint positions in AMBF
python data_collection_scripts/replay_ambf_episode.py \
    --episode ./data/ambf_dataset_new/episode_001.hdf5 \
    --start_pose ./policy_deployment_scripts/start_pose.json

# Replay with side-by-side visual comparison
python replay_with_vis.py \
    --episode ./data/ambf_dataset_new/episode_001.hdf5
```

### Check Episode Lengths

```bash
python length_check.py
```

---

## 6. Training

All training scripts are located in `training_scripts/`. Run from the repo root (`act_ambf/`).

### AMBF Pickup Task (recommended settings)

```bash
python training_scripts/imitate_episodes.py \
  --task_name ambf_task_pickup \
  --ckpt_dir ./checkpoints/ambf_pickup \
  --policy_class ACT \
  --batch_size 8 \
  --seed 0 \
  --num_epochs 2000 \
  --lr 2e-6 \
  --state_dim 7 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --kl_weight 10 \
  --temporal_agg
```

### AMBF Pickup Task (longer run / alternative config)

```bash
python training_scripts/imitate_episodes.py \
  --task_name ambf_task_pickup \
  --ckpt_dir ./checkpoints/ambf_task_pickup \
  --policy_class ACT \
  --batch_size 7 \
  --seed 0 \
  --num_epochs 3000 \
  --lr 1e-5 \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --temporal_agg
```

### dVRK Retraction Task

```bash
python training_scripts/imitate_episodes.py \
  --task_name dvrk_retraction_u_nof \
  --ckpt_dir ./checkpoints/dvrk_retraction_u_nof \
  --policy_class ACT \
  --batch_size 8 \
  --seed 0 \
  --num_epochs 5000 \
  --lr 2e-6 \
  --state_dim 8 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --kl_weight 10 \
  --temporal_agg
```

> **Tip:** Monitor training loss with `visualize_episodes.py` in `training_scripts/`.

---

## 7. Policy Deployment (Inference)

All deployment scripts are located in `policy_deployment_scripts/`. Run from the repo root.

> The AMBF simulation must be running (see [Section 3](#3-ambf-simulation)) before executing inference.

### AMBF Move Task

```bash
python policy_deployment_scripts/run_ambf_act.py \
    --ckpt_dir ./checkpoints/ambf_move \
    --task_name ambf_task_move \
    --state_dim 7 \
    --chunk_size 50 \
    --fps 20 \
    --resize_w 320 \
    --resize_h 240
```

### AMBF Pickup Task

```bash
python policy_deployment_scripts/run_ambf_act.py \
    --ckpt_dir ./checkpoints/ambf_pickup \
    --task_name ambf_task_pickup \
    --state_dim 7 \
    --chunk_size 50 \
    --start_pose ./policy_deployment_scripts/start_pose.json \
    --fps 20 \
    --resize_w 320 \
    --resize_h 240
```
