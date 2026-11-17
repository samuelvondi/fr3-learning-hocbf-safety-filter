"""Plot how the mean NN gains (gamma = p1, beta = p2) and the loss evolve over the training epochs.

Reads the epoch logs written by train_rollout.py (one JSON object per epoch with
"train" and optional "val" entries holding p1_mean / p2_mean) and draws one column
per run: gamma, beta and loss, training mean as a line, validation mean as dots.
Selected checkpoints can be marked with vertical lines.

Example (from src/):
    python -m fr3_hocbf.train.plot_gain_evolution \
        --run "Published=../artifacts/training_logs/fr3_v10_..._log.jsonl" \
        --mark "Published=22:rule 1" --out ../data/plots/gain_evolution.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MARK_STYLES = {"rule 1": "--", "rule 2": ":"}
FLOOR = 1e-3  # beta can reach ~0; clip for the log axis


def read_log(path: Path):
    train, val = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            ep = row["train"]["epoch"]
            train.append((ep, row["train"]["p1_mean"], row["train"]["p2_mean"], row["train"]["loss"]))
            v = row.get("val")
            if v and "p1_mean" in v:
                val.append((ep, v["p1_mean"], v["p2_mean"], v.get("loss", float("nan"))))
    return train, val


def parse_pairs(items):
    out = []
    for item in items:
        name, _, rest = item.partition("=")
        out.append((name, rest))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=path/to/log.jsonl (repeatable, sets column order)")
    ap.add_argument("--mark", action="append", default=[], help="NAME=EPOCH:LABEL, e.g. 'Seed 0=8:rule 1' (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = parse_pairs(args.run)
    marks: dict[str, list[tuple[int, str]]] = {}
    for name, spec in parse_pairs(args.mark):
        ep, _, label = spec.partition(":")
        marks.setdefault(name, []).append((int(ep), label or "selected"))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(3, len(runs), figsize=(2.6 * len(runs) + 0.6, 7.6),
                             sharex=True, sharey="row", squeeze=False)
    for col, (name, path) in enumerate(runs):
        train, val = read_log(Path(path))
        for row, (idx, sym) in enumerate([(1, "mean γ"), (2, "mean β"), (3, "loss")]):
            ax = axes[row][col]
            ax.plot([t[0] for t in train], [(max(t[idx], FLOOR) if idx == 2 else t[idx]) for t in train],
                    color=TRAIN_COLOR, lw=2, label="train mean")
            ax.plot([v[0] for v in val], [(max(v[idx], FLOOR) if idx == 2 else v[idx]) for v in val], "o",
                    ms=4, color=VAL_COLOR, mec="white", mew=0.8, label="val mean")
            for ep, label in marks.get(name, []):
                ax.axvline(ep, color=INK_2, lw=1, ls=MARK_STYLES.get(label, "-."), zorder=0)
                if row == 0:
                    ax.annotate(label, (ep, 1), xycoords=("data", "axes fraction"), xytext=(2, -10),
                                textcoords="offset points", fontsize=7, color=INK_2)
            ax.set_yscale("log")
            ax.grid(True, which="major", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if col == 0:
                ax.set_ylabel(sym + (" (clipped at 1e-3)" if row == 1 else ""))
            if row == 0:
                ax.set_title(name, fontsize=10)
            if row == 2:
                ax.set_xlabel("epoch")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Mean NN gains and loss over training epochs", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
