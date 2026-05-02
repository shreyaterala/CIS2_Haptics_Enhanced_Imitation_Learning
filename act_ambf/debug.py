# debug_dataloader.py
import sys
sys.path.insert(0, '/home/hzhao78/act-AMBF')
from utils import load_data

train_loader, val_loader, norm_stats, _ = load_data(
    './data/ambf_dataset_move',
    num_episodes=20,
    camera_names=['left'],
    batch_size_train=2,
    batch_size_val=2,
    state_dim=8,
    chunk_size=50
)

# 取一个 batch 检查
image_data, qpos_data, action_data, is_pad = next(iter(train_loader))

print("=== Shape 检查 ===")
print(f"image_data  : {image_data.shape}")   # 期望 (2, 1, 3, 120, 160)
print(f"qpos_data   : {qpos_data.shape}")    # 期望 (2, 8)
print(f"action_data : {action_data.shape}")  # 期望 (2, chunk_size, 8)
print(f"is_pad      : {is_pad.shape}")

print("\n=== 数值检查 ===")
print(f"qpos  mean={qpos_data.mean():.4f}  std={qpos_data.std():.4f}")
print(f"action mean={action_data.mean():.4f}  std={action_data.std():.4f}")
print(f"image  mean={image_data.mean():.4f}  std={image_data.std():.4f}")

print("\n=== norm_stats 检查 ===")
for k, v in norm_stats.items():
    print(f"{k}: shape={v.shape}, values={v}")