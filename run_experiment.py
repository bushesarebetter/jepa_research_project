#!/usr/bin/env python
"""Master script for the JEPA control-loss experiment.

Tests: does JEPA's self-prediction objective discard information that is
control-relevant but temporally unpredictable (the switch_color feature),
even though autoencoders and inverse-dynamics-augmented JEPA retain it?
"""
import argparse
import json
import os

import numpy as np
import torch

from src.env import SwitchColorEnv
from src.training import train_model, get_latents
from src.evaluation import (
    linear_probe_accuracy, nonlinear_probe_accuracy,
    effective_rank, centroid_distance,
    control_regret_probe, control_regret_rollout,
)
from src.information import estimate_mi_infonce
from src.figures import generate_all_figures
from src.aggregate import print_summary
from src.provenance import provenance

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

TRAIN_EPISODES = 200
EVAL_EPISODES = 80
EPISODE_LENGTH = 100

# Dataset/rollout sizing + a few knobs shared by every preset; presets override the rest.
DEFAULTS = dict(
    train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES, episode_length=EPISODE_LENGTH,
    contrasts=[1.0], rollout_episodes=50, rollout_episode_length=100,
    batch_size=256, lr=1e-3, mi_epochs=300,
)

# Named configs (selected with --config; legacy --quick maps to 'quick', default is 'full').
PRESETS = {
    'quick': dict(p_values=[0.5, 1.0], seeds=[0],
                  model_types=['jepa', 'ae', 'jepa_invdyn'], n_steps=2000),
    'full': dict(p_values=[0.5, 0.65, 0.8, 1.0], seeds=[0, 1, 2],
                 model_types=['jepa', 'ae', 'jepa_invdyn'], n_steps=30_000),
    # salience sweep: how low can switch_contrast go before each objective drops the bit.
    'salience_sweep': dict(contrasts=[0.1, 0.3, 0.5, 0.7, 0.9], p_values=[0.5, 1.0],
                           seeds=[0, 1, 2], model_types=['jepa', 'ae'], n_steps=30_000),
    # robustness: many seeds at the two hardest predictability settings.
    'robustness': dict(p_values=[0.5, 0.65], seeds=[0, 1, 2, 3, 4, 5, 6],
                       model_types=['jepa', 'ae', 'jepa_invdyn'], n_steps=30_000),
}

# Shrink any preset to a CPU-seconds smoke while keeping its sweep structure (p/seeds/contrasts).
SMOKE_OVERRIDE = dict(n_steps=80, batch_size=64, train_episodes=10, eval_episodes=8,
                      episode_length=15, rollout_episodes=6, rollout_episode_length=15,
                      mi_epochs=150)


def build_config(args):
    name = args.config or ('quick' if args.quick else 'full')
    if name not in PRESETS:
        raise SystemExit(f"Unknown --config {name!r}; choices: {sorted(PRESETS)}")
    cfg = dict(DEFAULTS)
    cfg.update(PRESETS[name])
    cfg['preset'] = name
    if args.smoke:
        cfg.update(SMOKE_OVERRIDE)
    cfg['tag'] = '_smoke' if args.smoke else ''  # isolates smoke artifacts from full-size ones
    if args.p_values is not None:
        cfg['p_values'] = args.p_values
    if args.seeds is not None:
        cfg['seeds'] = args.seeds
    if args.n_steps is not None:
        cfg['n_steps'] = args.n_steps
    if args.contrasts is not None:
        cfg['contrasts'] = args.contrasts
    if args.switch_contrast is not None:
        cfg['contrasts'] = [args.switch_contrast]
    return cfg


def _tag(p, seed, contrast, suffix):
    base = f"p{p:.2f}_seed{seed}"
    if abs(contrast - 1.0) > 1e-9:  # contrast 1.0 keeps legacy names -> 36-run cache/skip intact
        base += f"_c{contrast:.2f}"
    return base + suffix


def dataset_path(p, seed, contrast, suffix):
    return os.path.join(RESULTS_DIR, f"{_tag(p, seed, contrast, suffix)}_dataset.npz")


def result_path(p, seed, model_type, contrast, suffix):
    return os.path.join(RESULTS_DIR, f"{_tag(p, seed, contrast, suffix)}_{model_type}_result.json")


def get_or_make_dataset(p, seed, contrast, cfg):
    path = dataset_path(p, seed, contrast, cfg['tag'])
    if os.path.exists(path):
        npz = np.load(path)
        return {k: npz[k] for k in npz.files}

    train_env = SwitchColorEnv(p_repeat=p, seed=seed * 2 + 1, switch_contrast=contrast)
    eval_env = SwitchColorEnv(p_repeat=p, seed=seed * 2 + 2 + 100000, switch_contrast=contrast)
    train_data = train_env.collect_dataset(cfg['train_episodes'], cfg['episode_length'])
    eval_data = eval_env.collect_dataset(cfg['eval_episodes'], cfg['episode_length'])

    bundle = {}
    for k, v in train_data.items():
        bundle[f'train_{k}'] = v
    for k, v in eval_data.items():
        bundle[f'eval_{k}'] = v
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(path, **bundle)
    return bundle


def evaluate_model(model, model_type, eval_data, p, seed, device, cfg, contrast):
    latents = get_latents(model, eval_data['eval_obs'], device=device)
    labels = eval_data['eval_switch_color']

    lin_mean, lin_std = linear_probe_accuracy(latents, labels)
    nonlin_mean, nonlin_std = nonlinear_probe_accuracy(latents, labels)
    eff_rank = effective_rank(latents)
    cdist = centroid_distance(latents, labels)
    # Estimated I(Z; switch_color) -- the de-circularized x-axis for figure 3.
    mi = estimate_mi_infonce(latents, labels, epochs=cfg['mi_epochs'], device=device)

    # Legacy probe-based regret keeps the historical schema keys (control_regret/policy_reward).
    regret, policy_reward = control_regret_probe(
        model, p, device=device, train_seed=seed * 2 + 3 + 200000,
        eval_seed=seed * 2 + 4 + 300000, switch_contrast=contrast)
    # Independent rollout-based regret (additive; this is what the figures now use).
    roll = control_regret_rollout(
        model, 'switch_color', p, seed=seed * 2 + 5 + 400000,
        n_episodes=cfg['rollout_episodes'], device=device,
        episode_length=cfg['rollout_episode_length'], switch_contrast=contrast)

    return {
        'model_type': model_type, 'p_repeat': p, 'seed': seed, 'switch_contrast': contrast,
        'linear_probe_acc': lin_mean, 'linear_probe_std': lin_std,
        'nonlinear_probe_acc': nonlin_mean, 'nonlinear_probe_std': nonlin_std,
        'effective_rank': eff_rank, 'centroid_dist_norm': cdist,
        'mi_infonce': mi,
        'control_regret': regret, 'policy_reward': policy_reward,
        'control_regret_rollout': roll['regret'], 'rollout_reward': roll['rollout_reward'],
        'oracle_reward': roll['oracle_reward'],
    }


def run_sweep(cfg, device):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    all_results = []

    for contrast in cfg['contrasts']:
        for p in cfg['p_values']:
            for seed in cfg['seeds']:
                data = get_or_make_dataset(p, seed, contrast, cfg)
                train_data = {
                    'obs': data['train_obs'], 'next_obs': data['train_next_obs'],
                    'switch_color': data['train_switch_color'], 'action': data['train_action'],
                    'reward': data['train_reward'],
                }
                for model_type in cfg['model_types']:
                    rpath = result_path(p, seed, model_type, contrast, cfg['tag'])
                    if os.path.exists(rpath):
                        print(f"[skip] {rpath} already exists")
                        with open(rpath) as f:
                            all_results.append(json.load(f))
                        continue

                    print(f"=== Training {model_type} | p={p} seed={seed} c={contrast} ===")
                    torch.manual_seed(seed)
                    model, _ = train_model(model_type, train_data, n_steps=cfg['n_steps'],
                                            lr=cfg['lr'], batch_size=cfg['batch_size'], device=device)

                    result = evaluate_model(model, model_type, data, p, seed, device, cfg, contrast)
                    result['_provenance'] = provenance(
                        config=cfg, p_repeat=p, seed=seed, model_type=model_type,
                        switch_contrast=contrast, n_steps=cfg['n_steps'], lr=cfg['lr'],
                        batch_size=cfg['batch_size'])
                    with open(rpath, 'w') as f:
                        json.dump(result, f, indent=2)
                    torch.save(model.state_dict(), os.path.join(
                        CKPT_DIR, f"{_tag(p, seed, contrast, cfg['tag'])}_{model_type}.pt"))
                    all_results.append(result)
                    print(f"  -> linear_probe_acc={result['linear_probe_acc']:.3f} "
                          f"regret_rollout={result['control_regret_rollout']:.3f} "
                          f"eff_rank={result['effective_rank']:.1f}")
                    if abs(p - 0.5) < 1e-9 and result['effective_rank'] < 5:
                        print(f"  !!! WARNING: effective_rank={result['effective_rank']:.2f} < 5 "
                              f"at p=0.5 for {model_type} (c={contrast}) -- possible collapse")

    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    return all_results


def make_decision(results, p_values):
    def get(model_type, p, key):
        vals = [r[key] for r in results if r['model_type'] == model_type and r['p_repeat'] == p]
        return float(np.mean(vals)) if vals else None

    p_lo, p_hi = min(p_values), max(p_values)
    jepa_lo = get('jepa', p_lo, 'linear_probe_acc')
    jepa_hi = get('jepa', p_hi, 'linear_probe_acc')
    ae_lo = get('ae', p_lo, 'linear_probe_acc')

    decision = {'p_lo': p_lo, 'p_hi': p_hi, 'jepa_lo': jepa_lo, 'jepa_hi': jepa_hi, 'ae_lo': ae_lo}

    if None in (jepa_lo, jepa_hi, ae_lo):
        decision['verdict'] = 'INCOMPLETE'
        decision['reason'] = 'Missing required results for decision.'
        return decision

    gap_jepa = jepa_hi - jepa_lo
    gap_ae_jepa = ae_lo - jepa_lo
    decision['gap_jepa_predictability'] = gap_jepa
    decision['gap_ae_minus_jepa_at_lo'] = gap_ae_jepa

    cond_drop = gap_jepa > 0.05
    cond_ae_high = ae_lo > 0.82
    cond_gap = gap_ae_jepa > 0.10

    if not cond_ae_high:
        decision['verdict'] = 'STOP (architecture issue)'
        decision['reason'] = f"AE linear probe at p={p_lo} is {ae_lo:.3f} <= 0.82; encoder/decoder likely broken."
    elif jepa_lo > 0.82:
        decision['verdict'] = 'STOP (hypothesis false)'
        decision['reason'] = (f"JEPA retains switch_color (acc={jepa_lo:.3f}) even at the hardest "
                               f"predictability setting p={p_lo}.")
    elif cond_drop and cond_ae_high and cond_gap:
        decision['verdict'] = 'GO'
        decision['reason'] = "JEPA selectively drops switch_color as predictability decreases; AE retains it."
    else:
        decision['verdict'] = 'REFRAME'
        decision['reason'] = (f"Effect exists but criteria not all met "
                               f"(drop={gap_jepa:.3f}, ae_lo={ae_lo:.3f}, gap_ae_jepa={gap_ae_jepa:.3f}).")

    return decision


def print_and_save_decision(results, p_values):
    decision = make_decision(results, p_values)
    with open(os.path.join(RESULTS_DIR, 'decision.json'), 'w') as f:
        json.dump(decision, f, indent=2)
    print("\n" + "=" * 60)
    print(f"VERDICT: {decision['verdict']}")
    print(decision.get('reason', ''))
    print(json.dumps(decision, indent=2))
    print("=" * 60)
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=sorted(PRESETS), default=None,
                        help="named preset (default: full; --quick selects 'quick')")
    parser.add_argument('--smoke', action='store_true',
                        help="shrink the chosen preset to a CPU-seconds run (isolated artifacts)")
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--only-figures', action='store_true')
    parser.add_argument('--only-decision', action='store_true')
    parser.add_argument('--only-aggregate', action='store_true')
    parser.add_argument('--p-values', type=float, nargs='+', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--n-steps', type=int, default=None)
    parser.add_argument('--contrasts', type=float, nargs='+', default=None,
                        help="switch_contrast sweep values")
    parser.add_argument('--switch-contrast', type=float, default=None,
                        help="single switch_contrast value (overrides --contrasts)")
    args = parser.parse_args()

    cfg = build_config(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Config: {cfg}  device={device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    all_results_path = os.path.join(RESULTS_DIR, 'all_results.json')

    if args.only_figures:
        with open(all_results_path) as f:
            results = json.load(f)
        generate_all_figures(results, cfg['p_values'], FIG_DIR)
        return

    if args.only_decision:
        with open(all_results_path) as f:
            results = json.load(f)
        print_and_save_decision(results, cfg['p_values'])
        return

    if args.only_aggregate:
        with open(all_results_path) as f:
            results = json.load(f)
        print_summary(results)
        return

    results = run_sweep(cfg, device)
    print_and_save_decision(results, cfg['p_values'])
    generate_all_figures(results, cfg['p_values'], FIG_DIR)
    print_summary(results)


if __name__ == '__main__':
    main()
