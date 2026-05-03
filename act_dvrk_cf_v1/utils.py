import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,
                 state_dim, chunk_size, cf_dim, cf_clip=None):
        super(EpisodicDataset).__init__()
        self.episode_ids  = episode_ids
        self.dataset_dir  = dataset_dir
        self.camera_names = camera_names
        self.norm_stats   = norm_stats
        self.state_dim    = state_dim
        self.chunk_size   = chunk_size
        self.cf_dim       = cf_dim

        # ── cf_clip 解析 ─────────────────────────────────────────────
        # 优先级: 显式传入 > stats 里存的 > None (不 clip)
        if cf_clip is not None:
            self.cf_clip = float(cf_clip)
        elif "cf_clip" in norm_stats:
            self.cf_clip = float(norm_stats["cf_clip"])
        else:
            self.cf_clip = None

        # ── 可选: clip 命中率监控 (跑完一个 epoch 后可以看比例) ──────
        self.clip_hit_count = 0
        self.total_count    = 0

        self.is_sim = None
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
            T_action = root['/action'].shape[0]
            T_qpos   = root['/observations/qpos'].shape[0]
            T_cf     = root['/measured_cf'].shape[0]
            episode_len = min(T_action, T_qpos, T_cf)

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            qpos = root['/observations/qpos'][start_ts, :self.state_dim]
            qvel = root['/observations/qvel'][start_ts]
            cf   = root['/measured_cf'][start_ts, :self.cf_dim]

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
        cf_data     = torch.from_numpy(cf).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad      = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # ── 归一化 ───────────────────────────────────────────────────
        image_data  = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data   = (qpos_data   - self.norm_stats["qpos_mean"])   / self.norm_stats["qpos_std"]
        cf_data     = (cf_data     - self.norm_stats["cf_mean"])     / self.norm_stats["cf_std"]

        # ── cf clip (用 self.cf_clip,不要 hardcode!) ───────────────
        if self.cf_clip is not None:
            pre_clip = cf_data
            cf_data  = torch.clamp(cf_data, -self.cf_clip, self.cf_clip)
            # 监控: 这一帧有没有任何一维被 clip 到了
            if (pre_clip != cf_data).any().item():
                self.clip_hit_count += 1
            self.total_count += 1

        # ── 方案 A: 把 cf 拼到 qpos 后面 ────────────────────────────
        qpos_data = torch.cat([qpos_data, cf_data], dim=-1)

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, state_dim, cf_dim, cf_clip=3.0):
    all_qpos_data   = []
    all_action_data = []
    all_cf_data     = []

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
            cf     = root['/measured_cf'][()]
        T = min(len(qpos), len(action), len(cf))
        qpos   = qpos[:T]
        action = action[:T]
        cf     = cf[:T]

        # Use 2D tensors (T, dim) so cat works even with different T
        all_qpos_data.append(torch.from_numpy(qpos[:, :state_dim]).float())
        all_action_data.append(torch.from_numpy(action[:, :state_dim]).float())
        all_cf_data.append(torch.from_numpy(cf[:, :cf_dim]).float())

    # cat instead of stack: handles episodes with different lengths
    all_qpos_data   = torch.cat(all_qpos_data,   dim=0)  # (sum_T, state_dim)
    all_action_data = torch.cat(all_action_data, dim=0)  # (sum_T, state_dim)
    all_cf_data     = torch.cat(all_cf_data,     dim=0)  # (sum_T, cf_dim)

    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std  = all_action_data.std(dim=0,  keepdim=True)
    action_std  = torch.clip(action_std, 0.1, np.inf)

    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std  = all_qpos_data.std(dim=0,  keepdim=True)
    qpos_std  = torch.clip(qpos_std, 0.1, np.inf)

    cf_mean = all_cf_data.mean(dim=0, keepdim=True)
    cf_std  = all_cf_data.std(dim=0,  keepdim=True)
    cf_std  = torch.clip(cf_std, 1e-2, np.inf)

    # ── 诊断信息: 训练数据里 cf 的 z-score 分布,验证 clip 阈值是否合理 ──
    # |max| 远大于 99-percentile 说明有 outlier,clip 有意义
    # |max| 接近 99-percentile 说明分布很均匀,clip 的影响小
    cf_zscored = (all_cf_data - cf_mean) / cf_std
    cf_z_max = cf_zscored.abs().max(dim=0).values.numpy().tolist()
    cf_z_99  = torch.quantile(cf_zscored.abs(), 0.99, dim=0).numpy().tolist()
    cf_z_95  = torch.quantile(cf_zscored.abs(), 0.95, dim=0).numpy().tolist()
    print(f'[get_norm_stats] cf z-score |max| per dim:    {[round(x,2) for x in cf_z_max]}')
    print(f'[get_norm_stats] cf z-score 99-percentile:    {[round(x,2) for x in cf_z_99]}')
    print(f'[get_norm_stats] cf z-score 95-percentile:    {[round(x,2) for x in cf_z_95]}')
    print(f'[get_norm_stats] applying clip = ±{cf_clip}')

    # 估算会被 clip 的帧比例
    if cf_clip is not None and cf_clip > 0:
        clip_mask = (cf_zscored.abs() > cf_clip).any(dim=-1)
        clip_pct  = float(clip_mask.float().mean().item()) * 100
        print(f'[get_norm_stats] estimated clip-hit rate:     {clip_pct:.2f}% of frames')

    stats = {
        "action_mean":  action_mean.numpy().squeeze(),
        "action_std":   action_std.numpy().squeeze(),
        "qpos_mean":    qpos_mean.numpy().squeeze(),
        "qpos_std":     qpos_std.numpy().squeeze(),
        "cf_mean":      cf_mean.numpy().squeeze(),
        "cf_std":       cf_std.numpy().squeeze(),
        "cf_clip":      float(cf_clip) if cf_clip is not None else 0.0,
        "example_qpos": all_qpos_data[0].numpy(),
    }
    return stats


def load_data(dataset_dir, num_episodes, camera_names,
              batch_size_train, batch_size_val,
              state_dim, chunk_size, cf_dim, cf_clip=3.0):
    print(f'\nData from: {dataset_dir}')
    print(f'Using first {state_dim} dimensions of qpos/action')
    print(f'Using first {cf_dim} dimensions of measured_cf')
    print(f'CF z-score clip: ±{cf_clip}')
    print(f'Chunk size: {chunk_size}\n')

    train_ratio      = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices    = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices      = shuffled_indices[int(train_ratio * num_episodes):]

    # 关键: 把 cf_clip 传给 get_norm_stats,这样 stats 里存的是同一个值
    norm_stats = get_norm_stats(dataset_dir, num_episodes, state_dim, cf_dim,
                                cf_clip=cf_clip)

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names,
                                    norm_stats, state_dim, chunk_size, cf_dim,
                                    cf_clip=cf_clip)
    val_dataset   = EpisodicDataset(val_indices,   dataset_dir, camera_names,
                                    norm_stats, state_dim, chunk_size, cf_dim,
                                    cf_clip=cf_clip)

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