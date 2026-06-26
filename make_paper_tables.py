#!/usr/bin/env python
"""Generate LaTeX tables for the paper (booktabs format, siunitx-friendly).

Writes to results/tables/:
  table1_quadrant.tex        — full objective×cell matrix (probe acc mean±std, 3 seeds)
  table2_cell4.tex           — cell-4 failure across both envs (probe acc + MI)
  table3_capacity_minreward.tex — capacity sweep + min-reward-signal result

Usage:
    python make_paper_tables.py
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')


def load(name):
    return json.load(open(os.path.join(RESULTS_DIR, name), encoding='utf-8'))


def fmt(mean, std):
    """Format as mean±std (2 decimal places, siunitx-friendly)."""
    return f"${mean:.2f} \\pm {std:.2f}$"


def write(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  wrote {os.path.relpath(path, HERE)}")


# ── Table 1: full quadrant matrix ────────────────────────────────────────────

def table1_quadrant():
    d = load('quadrant_matrix.json')
    objs = d['objectives']
    cells = d['cells']

    # collect per-run values
    acc = defaultdict(list)
    for r in d['results']:
        acc[(r['objective'], r['cell'])].append(r['linear_probe_acc'])

    obj_labels = {
        'recon': r'\texttt{recon}',
        'jepa': r'\texttt{jepa}',
        'jepa_ac': r'\texttt{jepa\_ac}',
        'jepa_ctrl': r'\texttt{jepa\_ctrl}',
        'jepa_invdyn': r'\texttt{jepa\_invdyn}',
        'jepa_reward': r'\texttt{jepa\_reward}',
        'oracle': r'\texttt{oracle}',
    }
    cell_headers = {1: r'Cell 1\\ctrl+rel', 2: r'Cell 2\\ctrl+irr',
                    3: r'Cell 3\\exo+irr', 4: r'Cell 4\\exo+rel'}

    col_spec = 'l' + 'c' * len(cells)
    header_row = ' & '.join([r'\textbf{Objective}'] + [
        r'\makecell{\textbf{' + cell_headers[c].replace('\\\\', r'}\\\\' + r'\textbf{') + '}}'
        for c in cells
    ])

    rows = []
    for obj in objs:
        vals = [acc[(obj, c)] for c in cells]
        cells_fmt = [fmt(np.mean(v), np.std(v)) if v else '---' for v in vals]
        rows.append(f"  {obj_labels.get(obj, obj)} & " + ' & '.join(cells_fmt) + r' \\')

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Full objective $\times$ cell retention matrix. Values are linear probe accuracy'
        r' (mean\,$\pm$\,std over 3 seeds, chance\,=\,0.50). Highlighted: cell 4 (exo\,+\,rel),'
        r' the hard case where only reward-grounded signals retain the feature.}',
        r'\label{tab:quadrant_matrix}',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        header_row + r' \\',
        r'\midrule',
    ] + rows + [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ── Table 2: cell-4 failure both envs ────────────────────────────────────────

def table2_cell4():
    dq = load('quadrant_matrix_cell4_failure_quadrant.json')
    dg = load('quadrant_matrix_cell4_failure_gridworld.json')

    objs = dq['objectives']

    def _extract(d):
        acc, mi = defaultdict(list), defaultdict(list)
        for r in d['results']:
            acc[r['objective']].append(r['linear_probe_acc'])
            mi[r['objective']].append(r.get('mi_infonce', 0.0))
        return acc, mi

    acc_q, mi_q = _extract(dq)
    acc_g, mi_g = _extract(dg)

    obj_labels = {
        'recon': r'\texttt{recon}',
        'jepa': r'\texttt{jepa}',
        'jepa_ac': r'\texttt{jepa\_ac}',
        'jepa_ctrl': r'\texttt{jepa\_ctrl}',
        'jepa_invdyn': r'\texttt{jepa\_invdyn}',
        'jepa_reward': r'\texttt{jepa\_reward}',
        'oracle': r'\texttt{oracle}',
    }

    # multicolumn headers: Quadrant (acc, MI), GridWorld (acc, MI)
    rows = []
    for obj in objs:
        aq = acc_q[obj]; mq = mi_q[obj]
        ag = acc_g[obj]; mg = mi_g[obj]
        cells_fmt = [
            fmt(np.mean(aq), np.std(aq)),
            fmt(np.mean(mq), np.std(mq)),
            fmt(np.mean(ag), np.std(ag)),
            fmt(np.mean(mg), np.std(mg)),
        ]
        rows.append(f"  {obj_labels.get(obj, obj)} & " + ' & '.join(cells_fmt) + r' \\')

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Cell-4 failure across both environments (mean\,$\pm$\,std, 3 seeds, random-action'
        r' policy). Probe acc\,$<$\,0.75 = feature dropped. MI in nats; log\,2\,$\approx$\,0.693'
        r' = fully retained. Note large std for \texttt{jepa\_invdyn}:'
        r' its rescue tracks action informativeness rather than the feature itself.}',
        r'\label{tab:cell4_failure}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r' & \multicolumn{2}{c}{\textbf{Quadrant}} & \multicolumn{2}{c}{\textbf{GridWorld}} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r'\textbf{Objective} & Probe acc & MI (nats) & Probe acc & MI (nats) \\',
        r'\midrule',
    ] + rows + [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ── Table 3: capacity sweep + min reward signal ───────────────────────────────

def table3_capacity_minreward():
    cap = load('capacity_sweep.json')
    mr = load('min_reward_signal_quadrant.json')

    dims = cap['latent_dims']
    series = cap['series']

    # capacity sub-table: rows = latent_dim, cols = jepa/jepa_reward for quadrant + switch
    # For std we need per-run data... but capacity_sweep.json only has means in 'series'.
    # Fall back to showing just means (no per-run in capacity_sweep).
    # Actually let me check if there's per_run...
    per_run = cap.get('per_run', {})

    def _cap_fmt(label, d):
        if per_run and label in per_run and str(d) in per_run[label]:
            vals = per_run[label][str(d)]
            return fmt(np.mean(vals), np.std(vals))
        # just mean, no std
        idx = dims.index(d)
        return f"${series[label][idx]:.2f}$"

    cap_rows = []
    for d in dims:
        cells = [str(d)] + [_cap_fmt(k, d) for k in
                  ['quadrant/jepa', 'quadrant/jepa_reward', 'switch_color/jepa', 'switch_color/jepa_reward']]
        cap_rows.append('  ' + ' & '.join(cells) + r' \\')

    # min-reward sub-table: fraction -> mean acc
    mr_rows = []
    for frac, mean_acc in zip(mr['reward_fractions'], mr['mean_acc']):
        marker = r' $\leftarrow$ min passing' if frac == mr.get('min_fraction_passing') else ''
        mr_rows.append(f"  {frac:g} & ${mean_acc:.2f}$ & {marker}" + r' \\')

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{(Top) Capacity sweep: cell-4 linear probe accuracy vs latent dimension for'
        r' \texttt{jepa} and \texttt{jepa\_reward} on both environments (3 seeds; values are means'
        r' where per-seed data is available, else mean only). \texttt{jepa} stays near chance and'
        r' \texttt{jepa\_reward} stays near perfect across all sizes, ruling out capacity as the'
        r' bottleneck. (Bottom) Minimum reward-signal fraction: the smallest'
        r' \texttt{reward\_label\_fraction} for which \texttt{jepa\_reward} retains cell 4'
        r' (probe acc\,$\geq$\,0.75) on Quadrant (3 seeds, mean shown).}',
        r'\label{tab:capacity_minreward}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{latent\_dim} & \multicolumn{2}{c}{\textbf{Quadrant}} & \multicolumn{2}{c}{\textbf{SwitchColor}} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r' & \texttt{jepa} & \texttt{jepa\_reward} & \texttt{jepa} & \texttt{jepa\_reward} \\',
        r'\midrule',
    ] + cap_rows + [
        r'\midrule',
        r'\multicolumn{5}{l}{\textit{Minimum reward-label fraction (Quadrant, cell 4)}} \\',
        r'\midrule',
        r'\textbf{reward\_frac} & \multicolumn{4}{c}{\textbf{Cell-4 probe acc (mean)}} \\',
        r'\midrule',
    ] + mr_rows + [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    print("Generating paper tables -> results/tables/")

    t1 = table1_quadrant()
    write(os.path.join(TABLES_DIR, 'table1_quadrant.tex'), t1)

    t2 = table2_cell4()
    write(os.path.join(TABLES_DIR, 'table2_cell4.tex'), t2)

    t3 = table3_capacity_minreward()
    write(os.path.join(TABLES_DIR, 'table3_capacity_minreward.tex'), t3)

    print("Done.")


if __name__ == '__main__':
    main()
