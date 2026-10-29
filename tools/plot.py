"""Success curves for a run: evaluation return, failed clips, and training return vs wall-clock time."""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REF = "#8a8984"


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel("wall-clock training time (h), 1 GPU", color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=float, default=0.82)
    a = ap.parse_args()
    ev = read(os.path.join(a.run_dir, "eval.csv"))
    tr = read(os.path.join(a.run_dir, "train_log.csv"))
    ev.sort(key=lambda r: int(r["iteration"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURF)
    # (a) evaluation return
    ax = axes[0]
    style(ax, "(a) Evaluation normalized return, 40 LAFAN1 clips", "normalized return")
    if ev:
        x = [float(r["train_wall_h"]) for r in ev]
        y = [float(r["norm_return"]) for r in ev]
        ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.2, zorder=3)
        hit = next((i for i, v in enumerate(y) if v >= a.target), None)
        if hit is not None:
            ax.annotate(f"{a.target} reached at {x[hit]:.2f} h\n(it {ev[hit]['iteration']}, saved {ev[hit]['saved_at'][11:16]} UTC)",
                        (x[hit], y[hit]), xytext=(-40, -70), textcoords="offset points", fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))  # fmt: skip
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(-4, 10), textcoords="offset points", fontsize=8, color=INK, ha="center")
    ax.axhline(a.target, color=REF, lw=1, ls="--", zorder=1)
    ax.text(0.01, a.target + 0.01, f"target {a.target} (paper Fig. 4a)", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2)
    ax.plot([1.2], [0.82], marker="D", ms=7, color=REF, mec=SURF, zorder=4)
    ax.annotate("paper: 0.82 at 1.2 h (H100)", (1.2, 0.82), xytext=(-60, -22), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_ylim(0, 1)
    # (b) failed clips
    ax = axes[1]
    style(ax, "(b) Failed clips in evaluation (of 40)", "clips terminated early")
    if ev:
        x = [float(r["train_wall_h"]) for r in ev]
        y = [int(r["n_failed"]) for r in ev]
        ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.2, zorder=3, drawstyle="default")
        ax.annotate(f"{y[-1]}", (x[-1], y[-1]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
        zero = next((i for i, v in enumerate(y) if v == 0), None)
        if zero is not None:
            ax.annotate(f"40/40 succeed at {x[zero]:.2f} h\n(it {ev[zero]['iteration']}, saved {ev[zero]['saved_at'][11:16]} UTC)",
                        (x[zero], 0), xytext=(-150, 70), textcoords="offset points", fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))  # fmt: skip
    ax.set_ylim(-0.5, 40.5)
    # (c) training return
    ax = axes[2]
    style(ax, "(c) Training-episode normalized return", "mean normalized return per episode")
    if tr:
        x = [float(r["wall_h"]) for r in tr]
        y = [float(r["ep_ret_norm"]) for r in tr]
        k = 10
        ys = [sum(y[max(0, i - k) : i + 1]) / len(y[max(0, i - k) : i + 1]) for i in range(len(y))]
        ax.plot(x, ys, color=BLUE, lw=2, zorder=3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = a.out or os.path.join(a.run_dir, "success_curve.png")
    fig.savefig(out, dpi=140, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
