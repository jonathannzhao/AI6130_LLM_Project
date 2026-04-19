"""Bootstrap 95% CI for per-sample metrics across strategy output jsonls.

Computes mean + 95% percentile CI for BiasPhraseRetentionRate, OverEditRate,
and NoBias indicator. Also reports pairwise difference CI vs a reference strategy.

Usage:
  python bootstrap_ci.py --inputs constrained=path1.jsonl soft=path2.jsonl ce=path3.jsonl \
      --reference constrained --n_boot 10000 --seed 42
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def load_per_sample(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def bpr(row: dict) -> float:
    return float(row.get("bias_phrase_retention_sample", 0.0))


def oer(row: dict) -> float:
    return float(row.get("over_edit_rate_sample", 0.0))


def no_bias(row: dict) -> float:
    # 1 if model did NOT reduce bias count (NoBias event)
    src = row.get("source_bias_count", 0)
    pred = row.get("prediction_bias_count", 0)
    return 1.0 if pred >= src else 0.0


METRICS = {
    "BPR": bpr,
    "OER": oer,
    "NoBias%": no_bias,
}


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_mean(xs: List[float], n_boot: int, rng: random.Random) -> List[float]:
    n = len(xs)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        means.append(s / n)
    return means


def bootstrap_pair(xs: List[float], ys: List[float], n_boot: int, rng: random.Random) -> List[float]:
    assert len(xs) == len(ys)
    n = len(xs)
    diffs = []
    for _ in range(n_boot):
        sx = 0.0
        sy = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            sx += xs[idx]
            sy += ys[idx]
        diffs.append(sx / n - sy / n)
    return diffs


def percentile(values: List[float], p: float) -> float:
    vs = sorted(values)
    k = (len(vs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vs) - 1)
    frac = k - lo
    return vs[lo] * (1 - frac) + vs[hi] * frac


def align(rows_by_strategy: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    id_sets = [set(r["id"] for r in rows) for rows in rows_by_strategy.values()]
    common = set.intersection(*id_sets)
    out: Dict[str, List[dict]] = {}
    for name, rows in rows_by_strategy.items():
        by_id = {r["id"]: r for r in rows}
        aligned = [by_id[i] for i in sorted(common)]
        out[name] = aligned
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="name=path pairs, e.g. constrained=a.jsonl soft=b.jsonl")
    ap.add_argument("--reference", required=True, help="reference name for pairwise diff")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows_by_strategy: Dict[str, List[dict]] = {}
    for item in args.inputs:
        name, path = item.split("=", 1)
        rows_by_strategy[name] = load_per_sample(Path(path))
        print(f"[load] {name}: {len(rows_by_strategy[name])} rows")

    aligned = align(rows_by_strategy)
    n = len(next(iter(aligned.values())))
    print(f"[align] common n = {n}")
    print()

    ref_name = args.reference
    assert ref_name in aligned, f"reference {ref_name} not in inputs"

    for metric_name, fn in METRICS.items():
        print(f"=== {metric_name} ===")
        vals_by = {name: [fn(r) for r in rows] for name, rows in aligned.items()}
        for name in aligned:
            xs = vals_by[name]
            m = mean(xs)
            boots = bootstrap_mean(xs, args.n_boot, rng)
            lo = percentile(boots, 0.025)
            hi = percentile(boots, 0.975)
            print(f"  {name:25s} mean={m:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
        print(f"  -- pairwise diff vs {ref_name} --")
        for name in aligned:
            if name == ref_name:
                continue
            diffs = bootstrap_pair(vals_by[name], vals_by[ref_name], args.n_boot, rng)
            md = mean(diffs)
            lo = percentile(diffs, 0.025)
            hi = percentile(diffs, 0.975)
            signif = "SIG" if (lo > 0 or hi < 0) else "ns"
            print(f"  {name:25s} - {ref_name}: mean={md:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  [{signif}]")
        print()


if __name__ == "__main__":
    main()
