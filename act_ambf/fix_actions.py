#!/usr/bin/env python3
# fix_actions.py
#
# 修复录制数据中错误的 action 字段。
#
# 问题原因：
#   MTM 遥操作使用 servo_cp（笛卡尔控制），AMBF 的 joint_cmds 字段从未被填充，
#   导致录制的 action 是错误的值（接近0或无意义）。
#
# 修复方法：
#   正确的 action[t] = qpos[t+1]（下一时刻机器人实际到达的关节位置）
#   最后一步用 qpos[-1] 填充（保持维度不变）
#
# 用法：
#   # 修复单个文件
#   python fix_actions.py --input ./ambf_dataset/episode_001.hdf5
#
#   # 修复整个文件夹里所有 episode
#   python fix_actions.py --input ./ambf_dataset
#
#   # 预览修复结果，不实际写入
#   python fix_actions.py --input ./ambf_dataset --dry_run
#
#   # 不保留备份（默认会保留 .bak 文件）
#   python fix_actions.py --input ./ambf_dataset --no_backup

import os
import sys
import glob
import shutil
import argparse
import numpy as np
import h5py
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# 核心修复逻辑
# ─────────────────────────────────────────────────────────────

def compute_fixed_action(qpos: np.ndarray) -> np.ndarray:
    """
    action[t] = qpos[t+1]
    action[-1] = qpos[-1]  （最后一步用自身填充）
    """
    action_fixed = np.vstack([qpos[1:], qpos[-1:]])  # (T, J)
    return action_fixed.astype(np.float32)


def fix_episode(path: str, dry_run: bool, backup: bool, output_dir: str = None) -> dict:
    """
    修复单个 HDF5 文件的 action 字段。
    返回一个结果 dict，供汇总打印使用。
    """
    result = {
        "path":    path,
        "out_path": path,
        "status":  "ok",
        "steps":   0,
        "joints":  0,
        "before_mean_abs": 0.0,
        "after_mean_abs":  0.0,
        "error":   None,
    }

    try:
        # ── 读取原始数据 ──────────────────────────────────────
        with h5py.File(path, "r") as f:
            qpos          = f["observations/qpos"][:].astype(np.float32)
            action_old    = f["action"][:].astype(np.float32)
            all_attrs     = dict(f.attrs)

        T, J = qpos.shape
        result["steps"]  = T
        result["joints"] = J

        # ── 计算修复前后的误差对比 ────────────────────────────
        expected = compute_fixed_action(qpos)

        # 修复前：action_old 与 qpos[t+1] 的差距
        result["before_mean_abs"] = float(np.abs(action_old - expected).mean())

        # 修复后误差必然是 0（因为就是这么构造的）
        result["after_mean_abs"] = 0.0

        if dry_run:
            result["status"] = "dry_run"
            return result

        # ── 决定输出路径 ──────────────────────────────────────
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, os.path.basename(path))
            # 把原文件完整复制到新路径，再在新路径上修改
            shutil.copy2(path, out_path)
        else:
            out_path = path
            # 原地修改时才需要备份
            if backup:
                bak_path = path + ".bak"
                if not os.path.exists(bak_path):
                    shutil.copy2(path, bak_path)

        result["out_path"] = out_path

        # ── 写入修复后的 action ───────────────────────────────
        with h5py.File(out_path, "a") as f:
            # 删除旧的错误 action
            del f["action"]

            # 写入正确的 action
            f.create_dataset("action", data=expected, dtype="f4")

            # 更新 attrs 里的语义说明
            f.attrs["action_semantics"] = "qpos[t+1] — next joint positions"
            f.attrs["action_fixed_at"]  = datetime.now().isoformat(timespec="seconds")
            f.attrs["action_fix_note"]  = (
                "Fixed: original joint_cmds were empty (MTM uses servo_cp). "
                "Action now = qpos[t+1]."
            )

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)

    return result


# ─────────────────────────────────────────────────────────────
# 文件收集
# ─────────────────────────────────────────────────────────────

def collect_files(input_path: str) -> list:
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "episode_*.hdf5")))
        # 排除备份文件
        files = [f for f in files if not f.endswith(".bak")]
        return files
    else:
        print(f"ERROR: 路径不存在: {input_path}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# 验证修复结果
# ─────────────────────────────────────────────────────────────

def verify_episode(path: str):
    with h5py.File(path, "r") as f:
        qpos   = f["observations/qpos"][:].astype(np.float32)
        action = f["action"][:].astype(np.float32)

    expected = compute_fixed_action(qpos)
    diff     = np.abs(action - expected)

    print(f"\n  验证: {os.path.basename(path)}")
    print(f"    action 与 qpos[t+1] 误差: max={diff.max():.8f}  mean={diff.mean():.8f}")
    print(f"    qpos 变化量（机器人是否在动）: {np.abs(np.diff(qpos, axis=0)).mean(axis=0).round(4)}")

    if diff.max() < 1e-5:
        print("    ✅ 修复正确")
    else:
        print("    ⚠️  误差偏大，请检查")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="修复 AMBF 录制数据中错误的 action 字段（MTM servo_cp 场景）"
    )
    p.add_argument("--input", required=True,
                   help="单个 .hdf5 文件路径，或包含 episode_*.hdf5 的文件夹")
    p.add_argument("--output", default=None, metavar="DIR",
                   help="把修复后的文件存到这个新文件夹（原文件不动）。"
                        "不填则原地修改（自动备份为 .bak）")
    p.add_argument("--dry_run", action="store_true",
                   help="只预览，不实际写入文件")
    p.add_argument("--no_backup", action="store_true",
                   help="不保留 .bak 备份文件（默认保留）")
    p.add_argument("--verify", action="store_true",
                   help="修复完成后验证每个文件的结果")
    return p.parse_args()


def main():
    args   = parse_args()
    backup = not args.no_backup
    files  = collect_files(args.input)

    if not files:
        print("没有找到任何 episode_*.hdf5 文件")
        sys.exit(1)

    print(f"\n找到 {len(files)} 个文件")
    if args.dry_run:
        print("【预览模式】不会实际修改任何文件\n")
    elif args.output:
        print(f"修复后文件将保存到: {args.output}/\n")
    elif backup:
        print("备份文件将保存为 <原文件名>.bak\n")

    # ── 逐文件处理 ───────────────────────────────────────────
    results = []
    for path in files:
        r = fix_episode(path, dry_run=args.dry_run, backup=backup, output_dir=args.output)
        results.append(r)

        status_str = {
            "ok":      "✅ 修复完成",
            "dry_run": "👀 预览",
            "error":   "❌ 出错",
        }.get(r["status"], r["status"])

        out_str = ""
        if not args.dry_run and r.get("out_path") and r["out_path"] != path:
            out_str = f"  → {r['out_path']}"

        print(
            f"  {status_str}  {os.path.basename(path)}"
            f"  steps={r['steps']}  joints={r['joints']}"
            f"  修复前误差={r['before_mean_abs']:.4f}"
            + out_str
            + (f"  错误: {r['error']}" if r["error"] else "")
        )

    # ── 汇总 ─────────────────────────────────────────────────
    n_ok    = sum(1 for r in results if r["status"] in ("ok", "dry_run"))
    n_err   = sum(1 for r in results if r["status"] == "error")
    errs = [r["before_mean_abs"] for r in results if r["before_mean_abs"] > 0]
    avg_err = np.mean(errs) if errs else 0.0

    print(f"\n{'='*50}")
    print(f"处理完成: {n_ok} 成功  {n_err} 失败")
    print(f"修复前 action 平均误差: {avg_err:.4f}  （越大说明原来越错）")

    if not args.dry_run and args.verify:
        print(f"\n{'='*50}")
        print("开始验证...")
        for r in results:
            if r["status"] == "ok":
                verify_episode(r.get("out_path", r["path"]))

    if args.dry_run:
        print("\n预览完成，如确认无误请去掉 --dry_run 参数重新运行。")


if __name__ == "__main__":
    main()