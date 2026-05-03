import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, state_dim, chunk_size):
        super(EpisodicDataset).__init__()
        self.episode_ids  = episode_ids
        self.dataset_dir  = dataset_dir
        self.camera_names = camera_names
        self.norm_stats   = norm_stats
        self.state_dim    = state_dim
        self.chunk_size   = chunk_size
        self.is_sim       = None
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]

        dataset_path_padded = os.path.join(self.dataset_dir, f'episode_{episode_id:03d}.hdf5')
        dataset_path_nopad  = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        if os.path.exists(dataset_path_padded):
            dataset_path = dataset_path_padded
        elif os.path.exists(dataset_path_nopad):
            dataset_path = dataset_path_nopad
        else:
            raise FileNotFoundError(
                f"Episode file not found for index {episode_id}.\n"
                f"  Tried: {dataset_path_padded}\n"
                f"  Tried: {dataset_path_nopad}"
            )

        with h5py.File(dataset_path, 'r') as root:
            is_sim      = root.attrs['sim']
            episode_len = root['/action'].shape[0]

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            qpos = root['/observations/qpos'][start_ts, :self.state_dim]
            qvel = root['/observations/qvel'][start_ts]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            if is_sim:
                action     = root['/action'][start_ts:, :self.state_dim]
                action_len = episode_len - start_ts
            else:
                action     = root['/action'][max(0, start_ts - 1):, :self.state_dim]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim

        valid_len     = min(action_len, self.chunk_size)
        padded_action = np.zeros((self.chunk_size, self.state_dim), dtype=np.float32)
        padded_action[:valid_len] = action[:valid_len]
        is_pad        = np.zeros(self.chunk_size)
        if valid_len < self.chunk_size:
            is_pad[valid_len:] = 1

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data  = torch.from_numpy(all_cam_images)
        qpos_data   = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad      = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum('k h w c -> k c h w', image_data)

        image_data  = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data   = (qpos_data   - self.norm_stats["qpos_mean"])   / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, state_dim):
    all_qpos_data   = []
    all_action_data = []

    for episode_idx in range(num_episodes):

        dataset_path_padded = os.path.join(dataset_dir, f'episode_{episode_idx:03d}.hdf5')
        dataset_path_nopad  = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')

        if os.path.exists(dataset_path_padded):
            dataset_path = dataset_path_padded
        elif os.path.exists(dataset_path_nopad):
            dataset_path = dataset_path_nopad
        else:
            raise FileNotFoundError(
                f"Episode file not found for index {episode_idx}.\n"
                f"  Tried: {dataset_path_padded}\n"
                f"  Tried: {dataset_path_nopad}"
            )

        with h5py.File(dataset_path, 'r') as root:
            qpos   = root['/observations/qpos'][()]
            action = root['/action'][()]

        # Use 2D tensors (T, state_dim) so cat works even with different T
        all_qpos_data.append(torch.from_numpy(qpos[:, :state_dim]).float())
        all_action_data.append(torch.from_numpy(action[:, :state_dim]).float())

    # cat instead of stack: handles episodes with different lengths
    all_qpos_data   = torch.cat(all_qpos_data,   dim=0)  # (sum_T, state_dim)
    all_action_data = torch.cat(all_action_data, dim=0)  # (sum_T, state_dim)

    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std  = all_action_data.std(dim=0,  keepdim=True)
    action_std  = torch.clip(action_std, 0.1, np.inf)

    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std  = all_qpos_data.std(dim=0,  keepdim=True)
    qpos_std  = torch.clip(qpos_std, 0.1, np.inf)

    stats = {
        "action_mean":  action_mean.numpy().squeeze(),
        "action_std":   action_std.numpy().squeeze(),
        "qpos_mean":    qpos_mean.numpy().squeeze(),
        "qpos_std":     qpos_std.numpy().squeeze(),
        "example_qpos": all_qpos_data[0].numpy(),
    }
    return stats


def load_data(dataset_dir, num_episodes, camera_names,
              batch_size_train, batch_size_val, state_dim, chunk_size):
    print(f'\nData from: {dataset_dir}')
    print(f'Using first {state_dim} dimensions of qpos/action')
    print(f'Chunk size: {chunk_size}\n')

    train_ratio      = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices    = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices      = shuffled_indices[int(train_ratio * num_episodes):]

    norm_stats = get_norm_stats(dataset_dir, num_episodes, state_dim)

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names,
                                    norm_stats, state_dim, chunk_size)
    val_dataset   = EpisodicDataset(val_indices,   dataset_dir, camera_names,
                                    norm_stats, state_dim, chunk_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train,
                                  shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size_val,
                                  shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)