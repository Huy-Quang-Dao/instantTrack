"""Build the README/report tables and figures from the per-run summaries in ``results/runs/<run>/``
(``eval.csv`` from tools/eval_batch.py, ``train_log.csv`` from training).

    uv run python tools/make_report.py              # writes results/report.md and results/fig_*.png
"""

import argparse
import csv
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = 0.82
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"InstantTrack": "#2a78d6", "SuperMimic + EMA": "#eb6834", "SuperMimic": "#8a8984"}

# 3 seeds per method; plain SuperMimic has one 2 h run (it needs ~1.5 h to reach the target)
METHODS = {
    "SuperMimic": ["e0_supermimic_s0"],
    "SuperMimic + EMA": ["e3_supermimic_ema_s0", "e58_sm_ema_s1", "e59_sm_ema_s2"],
    "InstantTrack": ["e64_cr025_s0", "e65_cr025_s1", "e66_cr025_s2"],
}

# cumulative ladder from SuperMimic to InstantTrack: (label, runs) — each row adds one change
LADDER = [
    ("SuperMimic (paper hyperparameters)", ["e0_supermimic_s0"]),
    ("+ EMA policy 0.995", ["e3_supermimic_ema_s0", "e58_sm_ema_s1", "e59_sm_ema_s2"]),
    ("+ lr 4e-4, 5 WM batches, no host syncs, fused Adam", ["e18_ema_lr4e-4_wm5_s0", "e22_best_s1", "e23_best_s2"]),
    ("+ fused 3×3 rotations (`--fused_rot`)", ["e39_best_fusedrot_s0"]),
    ("+ compiled reward / obs / goal (`--compile_more`)", ["e40_best_fusedrot_cm_s0", "e41_gpu_s1", "e42_gpu_s2"]),
    ("+ world model 1024³ (was 2048³)", ["e35_best_wm1024_s0", "e44_wm1024_s1", "e45_wm1024_s2"]),
    ("+ failure-focused resets (`--fail_reset 0.3`)", ["e51_fr03_wm1024_s0", "e52_fr03_wm1024_s1", "e53_fr03_wm1024_s2"]),
    ("+ contact reward 0.25 = **InstantTrack**", ["e64_cr025_s0", "e65_cr025_s1", "e66_cr025_s2"]),
]

# single-axis changes that did not help (all on top of the configuration named in the note)
NEGATIVE = [
    ("Smaller policy and WM (1024-512-256)", ["e1_small_s0"], "SuperMimic"),
    ("Demonstration replay (DiffMimic)", ["e10_ema_replay_s0"], "SuperMimic + EMA"),
    ("Policy batch 2048", ["e13_ema_pib2048_s0"], "SuperMimic + EMA"),
    ("EMA 0.999", ["e14_ema999_s0"], "SuperMimic + EMA"),
    ("lr 6e-4", ["e20_ema_lr6e-4_wm5_s0"], "+ lr 4e-4, wm5"),
    ("3 WM batches", ["e21_ema_lr4e-4_wm3_s0"], "+ lr 4e-4"),
    ("Horizon 24", ["e24_best_h24_s0"], "+ lr 4e-4, wm5"),
    ("Horizon 16", ["e36_best_h16_s0"], "+ compile_more"),
    ("Foot-contact WM loss 0.05 (PhysMamba-style)", ["e27_best_foot05_s0"], "+ lr 4e-4, wm5"),
    ("Anchored imagination (on-policy corrections)", ["e34_best_anchored_s0"], "+ lr 4e-4, wm5"),
    ("2 critic epochs", ["e29_best_critic2_s0"], "+ lr 4e-4, wm5"),
    ("UTD 0.5", ["e30_best_utd05_s0"], "+ lr 4e-4, wm5"),
    ("CUDA graphs (`--cudagraphs`)", ["e43_gpu_cg_s0"], "+ compile_more"),
    ("WM window 4", ["e31_best_k4_s0"], "+ compile_more"),
    ("Contact-mode world model (`--wm_contact 0.1`)", ["e54_wc01_s0", "e55_wc01_s1", "e56_wc01_s2"], "+ fail_reset"),
    ("Contact reward 0.5", ["e60_cr05_s0", "e62_cr05_s1", "e63_cr05_s2"], "+ fail_reset"),
]


def read_eval(run):
    rows = list(csv.DictReader(open(os.path.join(ROOT, run, "eval.csv"))))
    return sorted((float(r["train_wall_h"]), float(r["norm_return"]), int(r["n_failed"]), int(r["iteration"])) for r in rows)


def its(run):
    rows = list(csv.DictReader(open(os.path.join(ROOT, run, "train_log.csv"))))
    v = [float(r["it_per_s"]) for r in rows if int(r["iteration"]) > 2000]
    return st.mean(v[:-2]) if len(v) > 4 else float("nan")  # the last lines include the final checkpoint


def held(rows, hold=3):
    """First checkpoint from which return >= TARGET with 0 failed clips holds for ``hold`` checkpoints."""
    ok = [ret >= TARGET and nf == 0 for _, ret, nf, _ in rows]
    return next((rows[i] for i in range(len(rows) - hold + 1) if all(ok[i : i + hold])), None)


def at(rows, h):
    prev = [r for r in rows if r[0] <= h + 1e-6]
    return prev[-1] if prev else None


def fmt(xs, f="{:.2f}"):
    return " / ".join(f.format(x) for x in xs)


def summary(runs):
    ev = {r: read_eval(r) for r in runs}
    hs = [held(ev[r]) for r in runs]
    return {
        "held_h": [h[0] if h else float("nan") for h in hs],
        "held_it": [h[3] if h else -1 for h in hs],
        "its": [its(r) for r in runs],
        "best": [max(x[1] for x in ev[r]) for r in runs],
        "ev": ev,
    }


def main():
    out = []
    S = {m: summary(r) for m, r in METHODS.items()}
    base = st.mean(S["SuperMimic + EMA"]["held_h"])

    out.append("## Time to target (return ≥ 0.82 with 0/40 failed clips, held for 3 checkpoints)\n")
    out.append("| method | seeds | held at (h) | mean (h) | iteration at held | it/s | speed-up vs SuperMimic + EMA |")
    out.append("|---|---|---|---|---|---|---|")
    for m, s in S.items():
        mean = st.mean(s["held_h"])
        out.append(f"| {m} | {len(s['held_h'])} | {fmt(s['held_h'])} | {mean:.2f} | {fmt(s['held_it'], '{}')} | "
                   f"{st.mean(s['its']):.2f} | {base / mean:.1f}× |")  # fmt: skip

    out.append("\n## Equal training time (mean over seeds: normalized return / failed clips)\n")
    hours = (0.15, 0.25, 0.5, 0.75)
    out.append("| method | " + " | ".join(f"{h:.2f} h" for h in hours) + " |")
    out.append("|---|" + "---|" * len(hours))
    for m, s in S.items():
        cells = []
        for h in hours:
            rs = [at(e, h) for e in s["ev"].values()]
            rs = [r for r in rs if r]
            cells.append(f"{st.mean(r[1] for r in rs):.3f} / {st.mean(r[2] for r in rs):.1f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")

    out.append("\n## Ablation ladder (each row adds one change to the row above)\n")
    out.append("| configuration | seeds | held at (h) | mean (h) | it/s | best return (0.75 h) |")
    out.append("|---|---|---|---|---|---|")
    for label, runs in LADDER:
        s = summary(runs)
        out.append(f"| {label} | {len(runs)} | {fmt(s['held_h'])} | {st.mean(s['held_h']):.2f} | {st.mean(s['its']):.2f} | "
                   f"{fmt(s['best'], '{:.3f}')} |")  # fmt: skip

    out.append("\n## Variants without a time-to-target gain\n")
    out.append("| change | on top of | seeds | held at (h) | it/s |")
    out.append("|---|---|---|---|---|")
    for label, runs, ref in NEGATIVE:
        s = summary(runs)
        out.append(f"| {label} | {ref} | {len(runs)} | {fmt(s['held_h'])} | {st.mean(s['its']):.2f} |")
    text = "\n".join(out) + "\n"
    open(os.path.join(OUT, "report.md"), "w").write(text)
    print(text)
    figures(S)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def figures(S):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})
    # time to target: one dot per seed, a bar at the mean
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    for i, (m, s) in enumerate(reversed(S.items())):
        mean = st.mean(s["held_h"])
        ax.barh(i, mean, height=0.5, color=COLORS[m], alpha=0.25, edgecolor="none")
        ax.scatter(s["held_h"], [i] * len(s["held_h"]), s=40, color=COLORS[m], edgecolor="white", lw=1.5, zorder=3)
        ax.text(max(s["held_h"]) + 0.04, i, f"{mean:.2f} h" + (" (mean)" if len(s["held_h"]) > 1 else ""), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(S)), list(reversed(S)))
    ax.set_xlabel("hours to return ≥ 0.82 with 0/40 failed clips (held ×3), 1× RTX 5090")
    ax.set_xlim(0, 1.75)
    style(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_time_to_target.png"), dpi=160)

    # learning curves vs wall-clock: mean line, min-max band over seeds (0.75 h: the length of the 3-seed runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    grid = [i * 0.025 for i in range(0, 31)]
    for m, s in S.items():
        for k, ax in ((1, axes[0]), (2, axes[1])):
            ys = []
            for e in s["ev"].values():
                ys.append([(at(e, h) or (0, 0, 40, 0))[k] for h in grid])
            ys = list(zip(*ys))
            mean = [st.mean(y) for y in ys]
            ax.plot(grid, mean, color=COLORS[m], lw=2, label=f"{m} ({len(s['ev'])} seed{'s' if len(s['ev']) > 1 else ''})")
            if len(s["ev"]) > 1:
                ax.fill_between(grid, [min(y) for y in ys], [max(y) for y in ys], color=COLORS[m], alpha=0.15, lw=0)
    axes[0].axhline(TARGET, color=INK2, ls="--", lw=1)
    axes[0].text(0.75, TARGET + 0.01, "0.82", ha="right", fontsize=8, color=INK2)
    axes[0].set_ylim(0.3, 0.92)
    axes[0].set_title("Evaluation normalized return (40 LAFAN1 clips)", loc="left", fontsize=10)
    axes[1].set_title("Failed clips (of 40)", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("training wall-clock (h)")
        ax.set_xlim(0, 0.75)
        style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_curves.png"), dpi=160)
    print("wrote", OUT + "/report.md and figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    ROOT, OUT = a.runs, a.out
    main()
