"""Compare runs on the SuperMimic metrics (see README, "Time to target"): time to return 0.82, time to 0.82 with 0 failed
clips, and return / failed clips at fixed wall-clock budgets. Reads ``logs/<run>/eval.csv``.

    uv run python tools/compare.py logs/e0_supermimic_s0 logs/it_small_s0 --out results/compare
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#8a8984", "#2a78d6", "#d6562a", "#2aa876", "#8f4fd6", "#d6a12a"]


def read_eval(run):
    path = os.path.join(run, "eval.csv")
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path)))
    out = [(float(r["train_wall_h"]), float(r["norm_return"]), int(r["n_failed"]), int(r["iteration"])) for r in rows]
    return sorted(out)


def first(rows, cond, hold=1, field=0):
    """First checkpoint (time, or iteration with ``field=3``) from which ``cond`` holds for ``hold`` consecutive checkpoints."""
    ok = [cond(ret, nf) for _, ret, nf, _ in rows]
    return next((rows[i][field] for i in range(len(rows) - hold + 1) if all(ok[i : i + hold])), None)


def at(rows, hours):
    """Last evaluated checkpoint at or before ``hours``."""
    prev = [r for r in rows if r[0] <= hours + 1e-6]
    return prev[-1] if prev else None


def fmt_h(x):
    return "–" if x is None else f"{x:.2f} h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--target", type=float, default=0.82)
    ap.add_argument("--out", default="results/compare")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    data = {os.path.basename(r.rstrip("/")): read_eval(r) for r in a.runs}

    lines = [f"| run | T→{a.target} (first / held ×3) | T→{a.target} & 0 fail (first / held ×3) | iteration at held | best return (h) | return / fail @0.5 h | @1 h | @2 h |",
             "|---|---|---|---|---|---|---|---|"]
    for name, rows in data.items():
        if not rows:
            lines.append(f"| {name} | no eval.csv | | | | | | |")
            continue
        c1 = lambda ret, nf: ret >= a.target  # noqa: E731
        c2 = lambda ret, nf: ret >= a.target and nf == 0  # noqa: E731
        t1 = f"{fmt_h(first(rows, c1))} / {fmt_h(first(rows, c1, 3))}"
        t2 = f"{fmt_h(first(rows, c2))} / {fmt_h(first(rows, c2, 3))}"
        it_held = first(rows, c2, 3, field=3)
        best = max(rows, key=lambda r: r[1])
        cells = []
        for h in (0.5, 1.0, 2.0):
            r = at(rows, h)
            cells.append("–" if r is None else f"{r[1]:.3f} / {r[2]}")
        lines.append(f"| {name} | {t1} | {t2} | {it_held if it_held is not None else '–'} | {best[1]:.3f} ({best[0]:.2f}) | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    open(a.out + ".md", "w").write(table + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (name, rows) in enumerate(data.items()):
        if not rows:
            continue
        c = PALETTE[i % len(PALETTE)]
        x = [r[0] for r in rows]
        axes[0].plot(x, [r[1] for r in rows], color=c, lw=2, marker="o", ms=3, label=name)
        axes[1].plot(x, [r[2] for r in rows], color=c, lw=2, marker="o", ms=3, label=name)
    axes[0].axhline(a.target, color="#8a8984", ls="--", lw=1)
    axes[0].set_title("Evaluation normalized return (40 LAFAN1 clips)", loc="left", fontsize=10)
    axes[1].set_title("Failed clips (of 40)", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("training wall-clock (h), 1× RTX 5090")
        ax.grid(True, color="#e4e3df")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=140)
    print("wrote", a.out + ".md", a.out + ".png")


if __name__ == "__main__":
    main()
