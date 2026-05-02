# Bilateral Teleoperation

----- Bilateral Teleop ----------------------

ros2 run dvrk_robot dvrk_system \
-j ~/ros2_ws/src/dvrk/dvrk_config_jhu/jhu-dVRK/system-MTML-MTMR-PSM1-PSM2.json

to run normal bilateral teleoperation:
cd ~/ros2_ws/src/dvrk/dvrk_python/scripts_s
ros2 run dvrk_python dvrk_bila_teleop.py -m MTML -p PSM2 -t "Normal"

to run the bilateral teleop with the neural network for force estimation:
cd ~/ros2_ws/src/dvrk/dvrk_python
source bil_env/bin/activate
pip install onnxruntime
cd scripts
ros2 run dvrk_python dvrk_bila_teleop.py -m MTML -p PSM2 -t "Neural Network"

to collect data for training the neural network:
** change DATA_PATH at the top of the dvrk_bila_teleop.py file which is located in the following folder
cd ~/ros2_ws/src/dvrk/dvrk_python/scripts
ros2 run dvrk_python dvrk_bila_teleop.py -m MTML -p PSM2 -t "Data Collection"
