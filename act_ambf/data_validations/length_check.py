import h5py, os
data_dir = "./data/ambf_dataset_pickup"
n = 0
for f in sorted(os.listdir(data_dir)):
    with h5py.File(os.path.join(data_dir, f), 'r') as d:
        print(f, d['/observations/qpos'].shape[0])