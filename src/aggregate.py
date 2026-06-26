"""Aggregate result JSONs into a mean +/- 95% CI table per (model, p), and flag
non-monotonicity in the JEPA linear-probe-vs-predictability curve.

The claim predicts JEPA's switch_color retention should be non-decreasing in p_repeat
(more predictable -> more retained). A dip as p increases is a red flag worth surfacing.

Run: python -m src.aggregate            (reads results/all_results.json)
     python -m src.aggregate path.json
"""
import json
import math
import os
import statistics
import sys

from scipy import stats  # only for the Student-t critical value

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
METRICS = ['linear_probe_acc', 'control_regret_rollout', 'effective_rank']


def load_results(path=None):
    path = path or os.path.join(RESULTS_DIR, 'all_results.json')
    with open(path) as f:
        return json.load(f)


def _ci95(vals):
    """Mean and 95% CI half-width (Student-t, falls back to 0 for n<2)."""
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    half = stats.t.ppf(0.975, n - 1) * statistics.stdev(vals) / math.sqrt(n)
    return mean, half


def _metric(r, key):
    # control_regret_rollout lives under the rollout dict on fresh results; tolerate older JSONs.
    if key == 'control_regret_rollout':
        return r.get('control_regret_rollout', r.get('control_regret'))
    return r.get(key)


def summarize(results):
    """-> {(model_type, p_repeat): {metric: (mean, ci95, n)}}, sorted keys."""
    groups = {}
    for r in results:
        key = (r['model_type'], r['p_repeat'])
        groups.setdefault(key, []).append(r)
    summary = {}
    for key, rs in sorted(groups.items()):
        summary[key] = {}
        for m in METRICS:
            vals = [_metric(r, m) for r in rs if _metric(r, m) is not None]
            if vals:
                mean, half = _ci95(vals)
                summary[key][m] = (mean, half, len(vals))
    return summary


def check_monotonicity(results, model_type='jepa', metric='linear_probe_acc', tol=0.02):
    """Flag (p_lo, p_hi) pairs where the curve DECREASES by more than tol as p increases."""
    curve = {}
    for r in results:
        if r['model_type'] != model_type:
            continue
        v = _metric(r, metric)
        if v is not None:
            curve.setdefault(r['p_repeat'], []).append(v)
    ps = sorted(curve)
    means = {p: sum(curve[p]) / len(curve[p]) for p in ps}
    violations = []
    for lo, hi in zip(ps, ps[1:]):
        if means[hi] < means[lo] - tol:
            violations.append((lo, hi, means[lo], means[hi]))
    return violations, means


def print_summary(results):
    summary = summarize(results)
    print("\n" + "=" * 78)
    print("AGGREGATE  (mean +/- 95% CI, n)")
    print("=" * 78)
    print(f"{'model':<14}{'p':>6}  {'lin_probe':>20}{'regret(roll)':>20}{'eff_rank':>16}")
    for (mt, p), mets in summary.items():
        cells = []
        for m in METRICS:
            if m in mets:
                mean, half, n = mets[m]
                cells.append(f"{mean:.3f}+/-{half:.3f}(n{n})")
            else:
                cells.append("--")
        print(f"{mt:<14}{p:>6.2f}  {cells[0]:>20}{cells[1]:>20}{cells[2]:>16}")

    violations, means = check_monotonicity(results)
    print("-" * 78)
    if means:
        curve = "  ".join(f"p{p:.2f}={means[p]:.3f}" for p in sorted(means))
        print(f"JEPA linear-probe vs p: {curve}")
    if violations:
        print("!! NON-MONOTONIC JEPA curve (acc should rise with p):")
        for lo, hi, mlo, mhi in violations:
            print(f"   p {lo:.2f}->{hi:.2f}: {mlo:.3f} -> {mhi:.3f}  (drop {mlo - mhi:.3f})")
    else:
        print("JEPA linear-probe vs p: monotonic non-decreasing (OK)")
    print("=" * 78)
    return summary


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    print_summary(load_results(path))


if __name__ == '__main__':
    main()
