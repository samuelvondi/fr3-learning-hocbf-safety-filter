#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def write_list(path: Path, items: list[str]):
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-dir",
        type=str,
        required=True,
        help="Directory containing rollout npz files.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        required=True,
        help="Directory where split manifests will be written.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to 1.0")

    rollout_dir = Path(args.rollout_dir).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(rollout_dir.glob("scenario_*_offline_rollout.npz"))
    files = [str(p.resolve()) for p in files]

    rng = random.Random(args.seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    write_list(manifest_dir / "train_files.txt", train_files)
    write_list(manifest_dir / "val_files.txt", val_files)
    write_list(manifest_dir / "test_files.txt", test_files)

    split_info = {
        "seed": args.seed,
        "rollout_dir": str(rollout_dir),
        "counts": {
            "total": n,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": args.test_frac,
        },
    }

    with open(manifest_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()