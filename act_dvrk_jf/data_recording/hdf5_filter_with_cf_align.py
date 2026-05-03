#!/usr/bin/env python3
"""
hdf5_filter_with_cf_align.py

Filter HDF5 datasets (keep only certain topics) with optional CF
zero-alignment.

Two CF alignment modes:
  --align_cf_mode per_batch   (default)
      One baseline B = mean of (each episode's first-N-frame mean) across
      the whole batch. All episodes subtract the same B.
      → Removes static sensor zero offset of the whole recording session.

  --align_cf_mode per_episode
      Each episode subtracts its OWN first-N-frame mean.
      → Removes any per-episode startup variability.
      → CAUTION: also wipes any real "initial contact force" differences
        between episodes, if such differences exist in your task.

Usage examples:

  # Diagnose only — show baseline stats per episode
  python hdf5_filter_with_cf_align.py -i ./data/raw --check_cf

  # Filter without alignment (original behavior)
  python hdf5_filter_with_cf_align.py -i ./data/raw -o ./data/filtered

  # Filter + per-batch alignment (recommended)
  python hdf5_filter_with_cf_align.py -i ./data/raw -o ./data/filtered_aligned \\
      --align_cf

  # Filter + per-episode alignment (each episode self-aligned, 5 baseline frames)
  python hdf5_filter_with_cf_align.py -i ./data/raw -o ./data/filtered_perep \\
      --align_cf --align_cf_mode per_episode --baseline_frames 5

  # Multi-batch merge with per-batch alignment per batch
  python hdf5_filter_with_cf_align.py \\
      --batch_input ./data/day1 \\
      --batch_input ./data/day2 \\
      -o ./data/merged_aligned --align_cf
"""

# python hdf5_filter_with_cf_align.py     
#     -i /home/hzhao78/act_dvrk/dvrk_retraction_A_unilateral/dvrk_dataset     
#     -o /home/hzhao78/act_dvrk/dvrk_retraction_A_unilateral/normal_pin_f_cf     
#     -t observations action timestamp source_timestamps measured_cf     
#     --align_cf   
#     --align_cf_mode per_episode

# python hdf5_filter_with_cf_align.py 
#     --batch_input /home/hzhao78/act_dvrk/dvrk_retraction_A_unilateral/dvrk_dataset 
#     --batch_input /home/hzhao78/act_dvrk/dvrk_retraction_B_unilateral/dvrk_dataset 
#     -o /home/hzhao78/act_dvrk/dvrk_retraction_AB_unilateral/normal_pin_cf 
#     -t observations action timestamp source_timestamps measured_cf --align_cf --align_cf_mode per_episode


import argparse
import h5py
import os
import sys
import glob
import shutil
import numpy as np


# ============================================================
# Topic listing
# ============================================================

def list_topics(group, prefix=''):
    topics = []
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            topics.append(path)
        elif isinstance(item, h5py.Group):
            topics.append(path + "/")
            topics.extend(list_topics(item, path))
    return topics


DEFAULT_TOPICS = [
    "observations",
    "action",
    "timestamp",
    "source_timestamps",
    "arm_effort",
    "spatial_cf",
    "measured_cf",
    "joint_force",
]


# ============================================================
# CF baseline computations
# ============================================================

def episode_baseline(path, baseline_frames):
    """
    Compute the mean of the first N frames of /measured_cf for one file.
    Returns full-width baseline (matching the file's cf width, typically 6-D),
    or None if the file has no /measured_cf or is too short.
    """
    try:
        with h5py.File(path, "r") as f:
            if "measured_cf" not in f:
                return None
            cf_full = f["measured_cf"][:baseline_frames].astype(np.float32)
            if len(cf_full) < baseline_frames:
                return None
            return cf_full.mean(axis=0)
    except Exception:
        return None


def compute_batch_cf_baseline(batch_dir, baseline_frames=5, cf_dim=6, verbose=True):
    """
    Walk every episode in batch_dir, compute its first-N-frames mean, then
    average across episodes. Returns:
        B_batch         : per-dim batch baseline (full width)
        per_episode     : dict {path: per-dim baseline (full width)}
        paths           : list of all paths found (sorted)
        skipped         : list of paths that had no usable cf
    """
    paths = sorted(glob.glob(os.path.join(batch_dir, "*.hdf5"))) + \
            sorted(glob.glob(os.path.join(batch_dir, "*.h5")))
    if not paths:
        raise FileNotFoundError(f"No HDF5 files found in {batch_dir}")

    per_episode = {}
    skipped = []
    for path in paths:
        b = episode_baseline(path, baseline_frames)
        if b is None:
            skipped.append(path)
            continue
        per_episode[path] = b

    if not per_episode:
        raise RuntimeError(f"No usable episodes in {batch_dir} for baseline")

    stacked = np.stack(list(per_episode.values()))   # (N_ep, full_width)
    B_batch = stacked.mean(axis=0).astype(np.float32)

    if verbose:
        full_width = stacked.shape[1]
        n_show = min(cf_dim, full_width)
        labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"][:n_show]
        col_trim = stacked[:, :n_show]
        print(f"  Per-episode baseline summary "
              f"(first {baseline_frames} frames mean, across {len(per_episode)} episodes):")
        print(f"    {'ch':<4} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
        for i, lab in enumerate(labels):
            col = col_trim[:, i]
            print(f"    {lab:<4} {col.mean():>+10.4f} {col.std():>10.4f} "
                  f"{col.min():>+10.4f} {col.max():>+10.4f}")
        if skipped:
            print(f"  [{len(skipped)} episode(s) skipped — no /measured_cf or too short]")

    return B_batch, per_episode, paths, skipped


# ============================================================
# Per-file processing
# ============================================================

def process_file(input_path, output_path, topics, force, list_only=False,
                 cf_subtract=None):
    """
    Copy selected topics from input_path → output_path.
    cf_subtract: 1-D ndarray. If provided and /measured_cf is being copied,
    subtract it from every frame.
    """
    try:
        with h5py.File(input_path, 'r') as f_in:
            if list_only:
                print(f"Topics available in '{input_path}':")
                for t in sorted(list_topics(f_in)):
                    print(f"  {t}")
                return True

            if not output_path:
                print("Error: --output is required when not using --list.")
                return False

            if os.path.exists(output_path) and not force:
                print(f"Error: Output file '{output_path}' already exists. "
                      f"Use -f/--force to overwrite.")
                return False

            with h5py.File(output_path, 'w') as f_out:
                for attr_name, attr_value in f_in.attrs.items():
                    f_out.attrs[attr_name] = attr_value

                for topic in topics:
                    clean_topic = topic.rstrip('/')

                    if (clean_topic == "measured_cf" and cf_subtract is not None
                            and clean_topic in f_in):
                        cf_data = f_in[clean_topic][()].astype(np.float32)
                        sub = np.zeros(cf_data.shape[-1], dtype=np.float32)
                        n = min(len(cf_subtract), cf_data.shape[-1])
                        sub[:n] = cf_subtract[:n]
                        cf_aligned = cf_data - sub
                        ds = f_out.create_dataset("measured_cf", data=cf_aligned, dtype="f4")
                        try:
                            for k, v in f_in["measured_cf"].attrs.items():
                                ds.attrs[k] = v
                        except Exception:
                            pass
                        f_out.attrs["cf_zero_aligned"] = True
                        f_out.attrs["cf_zero_alignment_B"] = sub.tolist()
                        continue

                    if clean_topic in f_in:
                        f_in.copy(clean_topic, f_out, clean_topic)
        return True
    except OSError as e:
        print(f"Failed to process HDF5 file '{input_path}': {e}")
        return False


# ============================================================
# Mode dispatchers
# ============================================================

def cmd_check_cf(batch_dirs, baseline_frames, cf_dim):
    print("=" * 72)
    print(f"CHECK MODE — analyzing CF baselines (first {baseline_frames} frames per episode)")
    print("=" * 72)
    summaries = []
    for bd in batch_dirs:
        print(f"\n[Batch] {bd}")
        B, per_ep, _, _ = compute_batch_cf_baseline(
            bd, baseline_frames, cf_dim, verbose=True
        )
        summaries.append((bd, B, per_ep))

    if len(summaries) >= 2:
        print()
        print("=" * 72)
        print("Cross-batch differences (vs first batch)")
        print("=" * 72)
        labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"][:cf_dim]
        ref = summaries[0][1][:cf_dim]
        for i, lab in enumerate(labels):
            print(f"  {lab:<4}", end="")
            for j, (bd, B, _) in enumerate(summaries):
                if j == 0:
                    print(f"  ref={B[i]:>+8.4f}", end="")
                else:
                    print(f"  Δ{j}={(B[i] - ref[i]):>+8.4f}", end="")
            print()

    print()
    print("Next steps:")
    print("  --align_cf                           → per-batch alignment")
    print("  --align_cf --align_cf_mode per_episode  → each episode self-aligned")


def cmd_single(args):
    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' does not exist.")
        sys.exit(1)

    per_episode_baselines = None
    batch_baseline = None
    if args.align_cf and os.path.isdir(args.input):
        print(f"\n[CF ALIGN] Computing baselines for: {args.input}")
        B_batch, per_ep, _, _ = compute_batch_cf_baseline(
            args.input, args.baseline_frames, args.cf_dim
        )
        if args.align_cf_mode == "per_episode":
            per_episode_baselines = per_ep
            print(f"[CF ALIGN] Mode: per_episode "
                  f"(each file subtracts its own first-{args.baseline_frames}-frame mean)")
            n_dim = min(args.cf_dim, len(B_batch))
            print(f"[CF ALIGN] (For reference, batch-mean B = "
                  f"{B_batch[:n_dim].tolist()})\n")
        else:
            batch_baseline = B_batch
            n_dim = min(args.cf_dim, len(B_batch))
            print(f"[CF ALIGN] Mode: per_batch "
                  f"(every file subtracts the same B = {B_batch[:n_dim].tolist()})\n")
    elif args.align_cf and not os.path.isdir(args.input):
        print("[WARN] --align_cf with single file makes no sense. Skipping alignment.")

    if os.path.isdir(args.input):
        if not args.list and not args.output:
            print("Error: --output directory is required when processing a directory.")
            sys.exit(1)
        if args.output and not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)

        files = sorted(glob.glob(os.path.join(args.input, "*.hdf5")) +
                       glob.glob(os.path.join(args.input, "*.h5")))
        if not files:
            print(f"No .hdf5 or .h5 files found in '{args.input}'.")
            sys.exit(0)

        print(f"Found {len(files)} file(s) to process.")
        n_done = 0
        for f in files:
            out_file = os.path.join(args.output, os.path.basename(f)) if args.output else None

            if per_episode_baselines is not None:
                cf_sub = per_episode_baselines.get(f)  # per-file's own
            elif batch_baseline is not None:
                cf_sub = batch_baseline
            else:
                cf_sub = None

            success = process_file(f, out_file, args.topics, args.force, args.list,
                                   cf_subtract=cf_sub)
            if not success and not args.list:
                print(f"Aborting due to error on '{f}'.")
                sys.exit(1)
            if success and not args.list:
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  ... {n_done}/{len(files)} files done")

        if not args.list:
            print(f"\nFiltering complete. {n_done} file(s) written to '{args.output}'.")
            if args.align_cf:
                _save_aux(args.output, args.cf_dim, args.align_cf_mode,
                          batch_baseline)
    else:
        success = process_file(args.input, args.output, args.topics, args.force,
                               args.list, cf_subtract=None)
        if not success:
            sys.exit(1)


def cmd_multi(args):
    if not args.output:
        print("Error: --output is required for multi-batch mode.")
        sys.exit(1)
    if os.path.exists(args.output) and not args.force:
        print(f"Error: Output dir '{args.output}' exists. Use -f to overwrite.")
        sys.exit(1)
    if os.path.exists(args.output) and args.force:
        print(f"[overwrite] Removing existing {args.output}")
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    batch_info = []
    for bd in args.batch_input:
        if not os.path.isdir(bd):
            print(f"Error: --batch_input '{bd}' is not a directory.")
            sys.exit(1)
        print(f"\n[Batch] {bd}")
        if args.align_cf:
            B, per_ep, paths, _ = compute_batch_cf_baseline(
                bd, args.baseline_frames, args.cf_dim
            )
            if args.align_cf_mode == "per_episode":
                print(f"  Mode: per_episode")
                batch_info.append((bd, "per_episode", per_ep, paths))
            else:
                n_dim = min(args.cf_dim, len(B))
                print(f"  Mode: per_batch (B = {B[:n_dim].tolist()})")
                batch_info.append((bd, "per_batch", B, paths))
        else:
            paths = sorted(glob.glob(os.path.join(bd, "*.hdf5")) +
                           glob.glob(os.path.join(bd, "*.h5")))
            print(f"  Found {len(paths)} files (no CF alignment)")
            batch_info.append((bd, "none", None, paths))

    out_idx = 0
    for bd, mode, baseline_data, paths in batch_info:
        print(f"\n[Processing] {bd}")
        for src_path in paths:
            out_name = f"episode_{out_idx:03d}.hdf5"
            out_path = os.path.join(args.output, out_name)

            if mode == "per_episode":
                cf_sub = baseline_data.get(src_path)
            elif mode == "per_batch":
                cf_sub = baseline_data
            else:
                cf_sub = None

            success = process_file(src_path, out_path, args.topics,
                                   force=True, list_only=False,
                                   cf_subtract=cf_sub)
            if not success:
                print(f"  Skipping {src_path} due to error")
                continue
            out_idx += 1
            if out_idx % 20 == 0:
                print(f"  ... {out_idx} files written")

    print(f"\nMerge complete. {out_idx} file(s) written to '{args.output}'.")
    if args.align_cf:
        _save_aux(args.output, args.cf_dim, args.align_cf_mode, None)


def _save_aux(output_dir, cf_dim, mode, batch_B):
    bias_path = os.path.join(output_dir, "cf_bias_train.npy")
    np.save(bias_path, np.zeros(cf_dim, dtype=np.float32))

    info_path = os.path.join(output_dir, "cf_alignment_info.txt")
    with open(info_path, "w") as f:
        f.write("This dataset has been CF zero-aligned at filter time.\n\n")
        f.write(f"Mode: {mode}\n")
        if mode == "per_batch" and batch_B is not None:
            n = min(cf_dim, len(batch_B))
            f.write(f"Subtracted batch baseline B (first {n} dims): {batch_B[:n].tolist()}\n")
        elif mode == "per_episode":
            f.write("Each episode subtracted its own first-N-frame mean.\n")
            f.write("Inspect each file's attrs['cf_zero_alignment_B'] to see what was subtracted.\n")
        f.write("\nAt inference time:\n")
        f.write(f"  - Use --cf_bias_path {bias_path}\n")
        f.write(f"  - cf_bias_train.npy is all zeros\n")
        f.write(f"  - Effective correction = cf_raw - B_run (only B_run subtracted at runtime)\n")
    print(f"  Saved: {bias_path}")
    print(f"  Saved: {info_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filter HDF5 datasets, optionally with CF zero-alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", "--input",
                        help="Input HDF5 file or directory.")
    parser.add_argument("--batch_input", action="append", default=[],
                        help="One or more recording-session directories. "
                             "Triggers merge mode.")

    parser.add_argument("-o", "--output", help="Output file or directory.")
    parser.add_argument("-t", "--topics", nargs='+', default=DEFAULT_TOPICS,
                        help="List of topics to keep.")
    parser.add_argument("-l", "--list", action="store_true",
                        help="List all available topics and exit.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite existing output.")

    parser.add_argument("--align_cf", action="store_true",
                        help="Apply CF zero-alignment when copying files.")
    parser.add_argument("--align_cf_mode",
                        choices=["per_batch", "per_episode"], default="per_batch",
                        help="per_batch: every file subtracts the same batch-wide B. "
                             "per_episode: each file subtracts its own first-N-frame mean.")
    parser.add_argument("--check_cf", action="store_true",
                        help="Only diagnose CF baselines; no file ops.")
    parser.add_argument("--baseline_frames", type=int, default=5,
                        help="Use first N frames of each episode as the static baseline.")
    parser.add_argument("--cf_dim", type=int, default=6,
                        help="How many CF dims to consider in baseline summary.")

    args = parser.parse_args()

    is_multi = len(args.batch_input) > 0

    if not is_multi and not args.input and not args.check_cf:
        parser.error("Provide either --input or --batch_input.")
    if args.check_cf and not is_multi and args.input:
        args.batch_input = [args.input]
        is_multi = True

    if args.check_cf:
        cmd_check_cf(args.batch_input, args.baseline_frames, args.cf_dim)
    elif is_multi:
        cmd_multi(args)
    else:
        cmd_single(args)


if __name__ == "__main__":
    main()