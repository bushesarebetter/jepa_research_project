#!/usr/bin/env python
"""Regenerate every figure from result JSONs. No training required.

Usage:
    python make_all_figures.py          # all figures -> results/figures/
    python make_all_figures.py --smoke  # use smoke JSONs where available

Each figure writes both .pdf and .png.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')


def load(name):
    return json.load(open(os.path.join(RESULTS_DIR, name), encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='use smoke JSONs where available')
    args = ap.parse_args()
    tag = '_smoke' if args.smoke else ''

    from src.figures import (
        figure1_main_result, figure2_bisimulation, figure3_theory_vs_empirical,
        figure4_ablations, figure5_environment, quadrant_heatmap,
        figure7_min_reward_signal, figure8_capacity, figure9_cell4_failure_both_envs,
    )

    # ── figures 1–5: SwitchColor p_repeat sweep ──────────────────────────────
    results = load('all_results.json')
    p_values = sorted(set(r['p_repeat'] for r in results))
    print("Generating figure1 (main result)...")
    figure1_main_result(results, p_values, FIG_DIR)
    print("Generating figure2 (bisimulation)...")
    figure2_bisimulation(results, p_values, FIG_DIR)
    print("Generating figure3 (theory vs empirical)...")
    figure3_theory_vs_empirical(results, FIG_DIR)
    print("Generating figure4 (ablations)...")
    figure4_ablations(results, p_values, FIG_DIR)
    print("Generating figure5 (environment)...")
    figure5_environment(FIG_DIR)

    # ── figure6: quadrant objective×cell heatmap ──────────────────────────────
    qm_file = f'quadrant_matrix{tag}.json'
    qm = load(qm_file)
    matrix = np.array(qm['matrix_linear_probe_acc'])
    print(f"Generating figure6 (quadrant matrix from {qm_file})...")
    quadrant_heatmap(matrix, qm['objectives'], qm['cells'], FIG_DIR,
                     name=f'figure6_quadrant_matrix{tag}')

    # ── figure7: min reward signal ────────────────────────────────────────────
    mr_file = f'min_reward_signal_quadrant{tag}.json'
    mr = load(mr_file)
    print(f"Generating figure7 (min reward signal from {mr_file})...")
    figure7_min_reward_signal(
        mr['reward_fractions'], mr['mean_acc'], FIG_DIR,
        min_frac=mr.get('min_fraction_passing'),
        name=f'figure7_min_reward_signal_quadrant{tag}',
    )

    # ── figure8: capacity sweep ───────────────────────────────────────────────
    cap_file = f'capacity_sweep{tag}.json'
    cap = load(cap_file)
    print(f"Generating figure8 (capacity from {cap_file})...")
    figure8_capacity(cap['series'], cap['latent_dims'], FIG_DIR,
                     name=f'figure8_capacity{tag}')

    # ── figure9: cell-4 failure both envs (THE MAIN FIGURE) ──────────────────
    q_fail_file = f'quadrant_matrix_cell4_failure_quadrant{tag}.json'
    g_fail_file = f'quadrant_matrix_cell4_failure_gridworld{tag}.json'
    # fall back to non-smoke if smoke variant missing
    def load_fallback(fname, fallback):
        try:
            return load(fname)
        except FileNotFoundError:
            print(f"  {fname} not found, falling back to {fallback}")
            return load(fallback)

    data_q = load_fallback(q_fail_file, 'quadrant_matrix_cell4_failure_quadrant.json')
    data_g = load_fallback(g_fail_file, 'quadrant_matrix_cell4_failure_gridworld.json')
    fig9_name = f'figure9_cell4_failure_both_envs{tag}'
    print(f"Generating figure9 (cell-4 failure both envs -> {fig9_name})...")
    figure9_cell4_failure_both_envs(data_q, data_g, FIG_DIR, name=fig9_name)

    # ── summary ───────────────────────────────────────────────────────────────
    figs = [f for f in sorted(os.listdir(FIG_DIR)) if f.endswith('.pdf') or f.endswith('.png')]
    print(f"\nAll figures written to {FIG_DIR}/")
    for f in figs:
        print(f"  {f}")


if __name__ == '__main__':
    main()
