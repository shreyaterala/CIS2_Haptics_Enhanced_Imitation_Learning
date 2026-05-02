import h5py, matplotlib.pyplot as plt

with h5py.File('./data/ambf_dataset_move/episode_000.hdf5', 'r') as f:
    img = f['/observations/images/left'][10]  # 取第10帧
    plt.imshow(img)
    plt.savefig('check_frame.png')
    print("非白像素比例:", (img < 250).mean())  # 应该 > 0.3 才正常