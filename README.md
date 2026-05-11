# CIS2_Haptics_Enhanced_Imitation_Learning
ACT Training and Deployment Code for CIS2

dVRK Data Collection

----- Normal Teleop w/o Force Feedback ------

1. Setup wrist cameras
- Attach to PSM and plug into any usb plugin on the computer
- Run the following:
source act_env/bin/activate
cd ~/act_dvrk
python wrist_cameras_video.py

2. Set up endoscope cameras
- Run the following in a new terminal:
ros2 launch /home/hzhao78/ros2_ws/src/dvrk/dvrk_video/ros2/launch/decklink_stereo_goovis.launch.py stereo_rig_name:=console1

3. Run teleoperation 
- Run the following in a new terminal:
ros2 run dvrk_robot dvrk_system \
-j ~/ros2_ws/src/dvrk/dvrk_config_jhu/jhu-dVRK/system-MTML-PSM1-PSM2-Teleop.json -C

4. Data Collection
- Run the following in a new terminal:
cd ~/act_dvrk
python record_dvrk_episodes.py --output_dir ./dvrk_dataset --seconds_per_episode 10 --fps 20 --wrist_cameras cam1

----- Bilateral Teleop ----------------------

ros2 run dvrk_robot dvrk_system \
-j ~/ros2_ws/src/dvrk/dvrk_config_jhu/jhu-dVRK/system-MTML-MTMR-PSM1-PSM2.json

to run normal bilateral teleoperation:
cd ~/ros2_ws/src/dvrk/dvrk_python/scripts
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

----- Data Collection w/ Force Feedback -----

1. Publish the ros2 node
cd ~/ros2_ws
ros2 run atinetft_ros atinetft_ros2_node -i 192.168.1.1

*Rebias the sensor at startup to zero it

2. To listen to the ros topic:
ros2 topic echo /measured_cf

3. To save the data for ground truth later
--force_sensor

----- Notes from Data Collection ----
* Make sure to move the robot around prior to collecting data to make sure all the rostopics are published before recording an episode
* Make sure teleoperation is off if running the bilateral teleop code
* If there are more than a couple of frames being dropped (ie >10) kill all the processes and restart everything running in the terminals
