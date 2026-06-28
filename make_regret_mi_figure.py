#!/usr/bin/env python
"""Build the de-circularized figure: independent rollout regret (y) vs InfoNCE MI (x).

Reads results/regret_mi_sweep.json and writes results/figures/figure3b_regret_vs_mi.{pdf,png}.
Each point is one (model, p_repeat, seed) run. MI (InfoNCE critic) and rollout regret
(behaviour-cloned policy rolled out in the env) are both independent of the linear probe,
so the regret-vs-MI relationship is non-circular.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, 'results', 'regret_mi_sweep.json')
FIGDIR = os.path.join(ROOT, 'results', 'figures')

COLORS = {'jepa': '#d62728', 'ae': '#2ca02c', 'jepa_invdyn': '#1f77b4'}
LABELS = {'jepa': 'JEPA', 'ae': 'AE (recon)', 'jepa_invdyn': 'JEPA + inv-dyn'}


def _rank_avg(x):
    order = np.argsort(x); sx = x[order]; ranks = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(x, y):  # tie-corrected
    return float(np.corrcoef(_rank_avg(np.asarray(x)), _rank_avg(np.asarray(y)))[0, 1])


def pearson(x, y):
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def main():
    rows = json.load(open(SRC))['results']
    mi = np.array([r['mi_infonce'] for r in rows])
    reg = np.array([r['control_regret_rollout'] for r in rows])
    rho = spearman(mi, reg)
    r = pearson(mi, reg)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        xs = [r['mi_infonce'] for r in rows if r['model_type'] == mt]
        ys = [r['control_regret_rollout'] for r in rows if r['model_type'] == mt]
        if xs:
            ax.scatter(xs, ys, c=COLORS[mt], label=LABELS[mt], alpha=0.8, s=45, edgecolors='white', linewidths=0.5)

    # least-squares trend line for the eye (not a model claim)
    if len(mi) >= 2:
        b, a = np.polyfit(mi, reg, 1)
        gx = np.linspace(mi.min(), mi.max(), 50)
        ax.plot(gx, a + b*gx, color='gray', ls='--', lw=1.2, label='linear trend')

    ax.axvline(np.log(2), color='black', ls=':', lw=1, alpha=0.6)
    ax.text(np.log(2), ax.get_ylim()[1]*0.96, r'  $\log 2$', fontsize=8, va='top')
    ax.set_xlabel('InfoNCE mutual information $I(Z;c)$ (nats)')
    ax.set_ylabel('Rollout control regret')
    ax.set_title('Independent control regret vs retained information', fontsize=10)
    ax.legend(fontsize=8, loc='center right')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, 'figure3b_regret_vs_mi.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'figure3b_regret_vs_mi.png'), dpi=150)
    print(f"Pearson r = {r:.3f} | Spearman rho = {rho:.3f} over {len(rows)} runs")
    print("wrote figure3b_regret_vs_mi.{pdf,png}")


if __name__ == '__main__':
    main()
