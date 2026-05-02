# ACT AMBF Implementation

Clean before connecting two pc

source /opt/ros/jazzy/setup.bash
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION
export ROS_DOMAIN_ID=46
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
ros2 daemon stop
ros2 daemon start


AMBF Operation guide

Terminal 1:
cd surgical_robotics_challenge/
./run_env_3D_MED_STRAIGHT_LND_420006.sh 
./run_env_SIMPLE_LND_420006.sh
./run_env_3D_MED_COMPLEX_LND_420006.sh

Step2:

Step3:

ros2 topic hz /ambf/env/cameras/Wrist1Frame/ImageData
ros2 topic hz /ambf/env/cameracameraR/ImageDatata
ros2 topic hz /ambf/env/cameracameraL/ImageData
ros2 topic list | grep cameras


Recording AMBF data and run ACT
--------------------------------------------------------------------------------------------------------------
Step1: Clean data for each Terminal!!!!!!!!!!!!

source /opt/ros/jazzy/setup.bash
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION
export ROS_DOMAIN_ID=46
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
ros2 daemon stop
ros2 daemon start
--------------------------------------------------------------------------------------------------------------
Step2: On another PC

Start dVRk console
Start MTM pair console
--------------------------------------------------------------------------------------------------------------
Step3: Data recording

cd act-AMBF
python record_ambf_episodes.py \
    --output_dir ./data/ambf_dataset_new \
    --fps 20 \
    --seconds_per_episode 7 \
    --resize_w 320 \
    --resize_h 240

#####Changing this one for different seconds, make sure all the data have same length##### --seconds_per_episode 10 \
--------------------------------------------------------------------------------------------------------------
Step4: Data check

1.view_episode.py
# View HDF5 recorded episode data with image playback and joint positions.
	Usage:
	python view_episode.py --episode ./data/ambf_dataset_new/episode_001.hdf5
	python view_episode.py --episode ./data/ambf_dataset_new/episode_001.hdf5 --cam wrist
	python view_episode.py --episode ./data/ambf_dataset_new/episode_001.hdf5 --all_cams
	python view_episode.py --episode ./data/ambf_dataset_new/episode_001.hdf5 --info
	
2.replay_ambf_episode.py
# Replay it inside AMBF, only need to use this pc, close everything on dvrk one!!!!
	Usage:
	python replay_ambf_episode.py     --episode ./data/ambf_dataset_new/episode_001.hdf5     --start_pose ./start_pose.json

3.replay_with_vis.py 
# Replay it inside AMBF, compare with recording, only need to use this pc, close everything on dvrk one!!!!
	Usage:
	python replay_with_vis.py     --episode ./data/ambf_dataset_new/episode_001.hdf5



--------------------------------------------------------------------------------------------------------------
Step5: Train
  python imitate_episodes.py \
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

dvrk_task_move

python imitate_episodes.py \
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


--------------------------------------------------------------------------------------------------------------
Step6: Run

python run_ambf_act.py \
    --ckpt_dir ./checkpoints/ambf_move \
    --task_name ambf_task_move \
    --state_dim 7 \
    --chunk_size 50 \
    --fps 20 \
    --resize_w 320 \
    --resize_h 240
    
    
python run_ambf_act.py \
    --ckpt_dir ./checkpoints/ambf_pickup \
    --task_name ambf_task_pickup \
    --state_dim 7 \
    --chunk_size 50 \
    --start_pose ./start_pose.json \
    --fps 20


python imitate_episodes.py \
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
