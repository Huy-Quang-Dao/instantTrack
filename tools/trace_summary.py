"""Summarize a torch.profiler Chrome trace written by ``train.py --torch_profile N`` (``logs/<run>/trace.json``):
GPU busy time as the union of kernel intervals, idle share, kernel counts, and time per kernel category.

    uv run python tools/trace_summary.py logs/prof_best/trace.json logs/prof_fusedrot/trace.json --iters 5
"""

import argparse
import collections
import json


def category(name: str) -> str:
    n = name.lower()
    if any(t in n for t in ("gemm", "cutlass", "gemv", "nvjet", "cublas", "s1688", "s16816", "xmma", "splitk")):
        return "GEMM bf16" if "bf16" in n else "GEMM fp32/tf32"
    if n.startswith("triton") or "triton_" in n:
        return "triton (torch.compile)"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "aten elementwise (eager)"
    if "adam" in n or "multi_tensor" in n:
        return "optimizer"
    if "reduce" in n:
        return "aten reduce"
    return "other"


def summarize(path: str, iters: int) -> dict:
    ev = [e for e in json.load(open(path))["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev)
    busy, (cs, ce) = 0.0, iv[0]
    for s, e in iv[1:]:
        if s > ce:
            busy, cs, ce = busy + ce - cs, s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = iv[-1][1] - iv[0][0]
    cats = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        c = cats[category(e["name"])]
        c[0] += e["dur"]
        c[1] += 1
    return {
        "wall_ms": span / iters / 1e3,
        "busy_ms": busy / iters / 1e3,
        "kernels": len(ev) / iters,
        "short_kernels": sum(e["dur"] < 10 for e in ev) / iters,
        "cats": {k: (d / iters / 1e3, n / iters) for k, (d, n) in cats.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    res = {p: summarize(p, a.iters) for p in a.traces}
    names = sorted({k for r in res.values() for k in r["cats"]}, key=lambda k: -max(r["cats"].get(k, (0, 0))[0] for r in res.values()))
    head = "| per iteration | " + " | ".join(p.split("/")[-2] for p in a.traces) + " |"
    rows = [head, "|---" * (len(a.traces) + 1) + "|"]
    rows.append("| wall (ms) | " + " | ".join(f"{r['wall_ms']:.1f}" for r in res.values()) + " |")
    rows.append("| GPU busy (ms, %) | " + " | ".join(f"{r['busy_ms']:.1f} ({100 * r['busy_ms'] / r['wall_ms']:.0f} %)" for r in res.values()) + " |")
    rows.append("| kernels (< 10 µs) | " + " | ".join(f"{r['kernels']:.0f} ({r['short_kernels']:.0f})" for r in res.values()) + " |")
    for k in names:
        rows.append(f"| {k} (ms) | " + " | ".join(f"{r['cats'].get(k, (0, 0))[0]:.1f}" for r in res.values()) + " |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
