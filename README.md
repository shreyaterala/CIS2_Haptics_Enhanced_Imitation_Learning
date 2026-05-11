# CIS2 Haptics-Enhanced Imitation Learning

ACT (Action Chunking with Transformers) training and deployment code for surgical robot tasks on the dVRK and AMBF simulator.
This repository contains the full pipeline — from bilateral teleoperation and data collection through model training to policy deployment — across several haptics configurations.

---

## Repository Overview

```
CIS2_Haptics_Enhanced_Imitation_Learning/
├── bilateral_teleoperation/    # Bilateral teleoperation scripts (force-feedback teleoperation)
├── act_ambf/                   # ACT in AMBF simulation (no force feedback)
├── act_dvrk/                   # ACT on real dVRK — baseline (joints only)
├── act_dvrk_cf_v1/             # ACT on real dVRK — Cartesian force v1 (3-DOF)
├── act_dvrk_cf_v2/             # ACT on real dVRK — Cartesian force v2 (3-DOF, improved)
└── act_dvrk_jf/                # ACT on real dVRK — Joint-space force (6-DOF wrench)
```

---

## Pipeline at a Glance

```
Teleoperation ──► Data Collection ──► (Force Alignment) ──► ACT Training ──► Policy Deployment
     │                  │                     │                   │                   │
bilateral_        act_dvrk/            act_dvrk_cf_v*/      imitate_             run_dvrk_act.py
teleoperation/    data_recording/      data_recording/      episodes.py          (or run_ambf_act.py)
```

---

## Module Descriptions

### `bilateral_teleoperation/`

Scripts for bilateral teleoperation of the dVRK with optional neural-network-based force estimation.
Used independently of ACT training — this is the **operator interface** for collecting demonstrations.

Key files:
- `dvrk_bila_teleop.py` — Main teleoperation driver. Supports three modes:
  - `"Normal"` — standard teleoperation without force feedback
  - `"Neural Network"` — force estimation via ONNX model
  - `"Data Collection"` — records data for training the force estimation network

See [`bilateral_teleoperation/README.md`](bilateral_teleoperation/README.md) for full setup and usage.

---

### `act_ambf/`

ACT running inside the **AMBF surgical robotics simulator**. Used for sim-to-real development and testing policy architectures before transferring to real hardware.

- **State dim:** 7 (single-arm simulation)
- **Force feedback:** None
- **Entry points:** `data_collection_scripts/record_ambf_episodes.py`, `training_scripts/imitate_episodes.py`, `policy_deployment_scripts/run_ambf_act.py`

See [`act_ambf/README.md`](act_ambf/README.md) for full setup, AMBF launch instructions, and training configs.

---

### `act_dvrk/`

Baseline ACT on the **real dVRK** — joint positions and camera images only, no haptic augmentation.

- **State dim:** 14 (7 joints × 2 PSMs)
- **Force feedback:** None
- **Task:** `dvrk_retraction`
- **Entry points:** `data_recording/record_dvrk_episodes.py`, `imitate_episodes.py`, `run_dvrk_act.py`

See [`act_dvrk/README.md`](act_dvrk/README.md) for hardware setup, data collection, training, and deployment.

---

### `act_dvrk_cf_v1/`

First-generation haptics-enhanced ACT on the dVRK. Appends **3-DOF Cartesian force** (Fx, Fy, Fz) from an ATI NetFT sensor to the state.

- **State dim:** 17 (14 joints + 3 Cartesian force channels)
- **Force feedback:** ATI NetFT → `/measured_cf` (3-DOF)
- **Task:** `dvrk_retraction_cf`
- **Post-processing:** `data_recording/hdf5_filter_with_cf_align.py` aligns force timestamps

See [`act_dvrk_cf_v1/README.md`](act_dvrk_cf_v1/README.md) for full pipeline details.

---

### `act_dvrk_cf_v2/`

Second-generation Cartesian-force variant. Structurally identical to v1 but with **improved force timestamp alignment and filtering** in `hdf5_filter_with_cf_align.py`.

- **State dim:** 17 (14 joints + 3 Cartesian force channels)
- **Force feedback:** ATI NetFT → `/measured_cf` (3-DOF, improved alignment)
- **Task:** `dvrk_retraction_cf`

See [`act_dvrk_cf_v2/README.md`](act_dvrk_cf_v2/README.md) for full pipeline details.

---

### `act_dvrk_jf/`

Full **6-DOF wrench** haptics-enhanced ACT. Appends both linear forces and torques (Fx, Fy, Fz, Tx, Ty, Tz) to give the policy richer contact information during grasping and rotation tasks.

- **State dim:** 20 (14 joints + 6-DOF wrench)
- **Force feedback:** ATI NetFT → `/measured_cf` (6-DOF)
- **Task:** `dvrk_retraction_u_cf`

See [`act_dvrk_jf/README.md`](act_dvrk_jf/README.md) for full pipeline details.

---

## Quick-Start: Real dVRK (Baseline)

```bash
# 1. Setup cameras and dVRK (see act_dvrk/README.md)

# 2. Record demonstrations
conda activate aloha && cd act_dvrk/
python data_recording/record_dvrk_episodes.py \
    --output_dir ./data/dvrk_dataset \
    --seconds_per_episode 10 --fps 20 --wrist_cameras cam1

# 3. Train
python imitate_episodes.py \
    --task_name dvrk_retraction \
    --ckpt_dir ./checkpoints/dvrk_retraction \
    --policy_class ACT --state_dim 14 --chunk_size 100 \
    --hidden_dim 512 --dim_feedforward 3200 \
    --kl_weight 10 --num_epochs 3000 --lr 1e-5 --temporal_agg

# 4. Deploy
python run_dvrk_act.py \
    --ckpt_dir ./checkpoints/dvrk_retraction \
    --task_name dvrk_retraction \
    --state_dim 14 --chunk_size 100 --fps 20 --temporal_agg
```

---

## Choosing a Variant

| Variant | State Dim | Force Signal | When to Use |
|---|---|---|---|
| `act_dvrk` | 14 | None | Baseline; no ATI sensor |
| `act_dvrk_cf_v1` | 17 | 3-DOF Cartesian force | First-pass haptics experiments |
| `act_dvrk_cf_v2` | 17 | 3-DOF Cartesian force (improved) | Cleaner force data needed |
| `act_dvrk_jf` | 20 | 6-DOF wrench (force + torque) | Grasping / rotational contact tasks |
| `act_ambf` | 7 | None | Simulation only |

---

## Notes

- Move the robot around **before** each recording session to ensure all ROS 2 topics are publishing.
- Re-bias the ATI sensor at the start of every session when using force feedback variants.
- If more than ~10 frames are dropped during recording, kill all terminals and restart.
- For real-world data, train for at least **5000 epochs** or until 3–4× past loss plateau.
- Use `--temporal_agg` at both training and deployment for smoother trajectories.
