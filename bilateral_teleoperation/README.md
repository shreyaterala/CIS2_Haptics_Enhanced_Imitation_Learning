# Bilateral Teleoperation

---

## 1. First-Time Setup

Navigate to the `dvrk_python` scripts folder and make the teleoperation script executable:

```bash
cd ~/ros2_ws/src/dvrk/dvrk_python/scripts
chmod +x dvrk_bila_teleop.py
```

Then build the workspace (only required the **first time** you add the script):

```bash
cd ~/ros2_ws
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

> **Note:** After the first build, if you modify the script you only need to rebuild the `dvrk_python` package:
> ```bash
> colcon build --packages-select dvrk_python
> source install/setup.bash
> ```

---

## 2. Launch the dVRK System

Start the dVRK robot system in a dedicated terminal:

```bash
ros2 run dvrk_robot dvrk_system \
    -j ~/ros2_ws/src/dvrk/dvrk_config_jhu/jhu-dVRK/system-MTML-MTMR-PSM1-PSM2.json
```

---

## 3. Running Bilateral Teleoperation

### Option A — Normal Teleoperation (no force feedback)

```bash
cd ~/ros2_ws/src/dvrk/dvrk_python/scripts
ros2 run dvrk_python dvrk_bila_teleop.py -m MTML -p PSM2 -t "Normal"
```

### Option B — Neural Network Force Estimation

Activate the dedicated virtual environment, then run:

```bash
cd ~/ros2_ws/src/dvrk/dvrk_python
source bil_env/bin/activate
pip install onnxruntime      # only needed once
cd scripts
ros2 run dvrk_python dvrk_bila_teleop.py -m MTML -p PSM2 -t "Neural Network"
```

### Option C — Data Collection (for training the force model)

> **Before running:** Open `dvrk_bila_teleop.py` and set the `DATA_PATH` variable at the top of the file.
> The script is located at: `~/ros2_ws/src/dvrk/dvrk_python/scripts/dvrk_bila_teleop.py`

```bash
cd ~/ros2_ws/src/dvrk/dvrk_python/scripts
ros2 run dvrk_python dvrk_bila_teleop.py -m MTML -p PSM2 -t "Data Collection"
```
