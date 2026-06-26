#!/usr/bin/env python
"""M3/M4 orchestrator: objective x cell retention matrix + the M4 sweeps, over 3 environments.

Default (--config matrix) reproduces the M3 objective x cell matrix on the 2x2 QuadrantEnv.
M4 adds:
  --config cell4_failure      jepa_ctrl + jepa_invdyn on RANDOM actions on cell 4 -> they fail;
                              jepa_reward + oracle keep it (the existing fixes-fail demo).
  --config min_reward_signal  sweep reward_label_fraction for jepa_reward on cell 4; find the
                              smallest fraction whose retention crosses the pass threshold.
  --config capacity_sweep     sweep latent_dim {16..1024} on cell 4 (quadrant + switch) -> flat
                              curves show latent capacity is not the bottleneck.
  --env {switch_color,gridworld,quadrant}   the surface form (cell 4 replicates in all three).

Invariant 1 (encoder byte-identical across objectives) is asserted across every trained model.

  python quadrant_experiment.py                              # CPU smoke: matrix on quadrant
  python quadrant_experiment.py --config cell4_failure --env gridworld
  python quadrant_experiment.py --config min_reward_signal --full
  python quadrant_experiment.py --config capacity_sweep --full
"""
import argparse
import json
import os

import numpy as np
import torch

from src.quadrant_env import QuadrantEnv, make_cell
from src.env import SwitchColorEnv
from src.gridworld_env import GridWorldHiddenRuleEnv
from src.training import train_objective, get_latents
from src.models import OBJECTIVES, encoder_signature
from src.evaluation import linear_probe_accuracy
from src.information import estimate_mi_infonce
from src.figures import quadrant_heatmap, figure7_min_reward_signal, figure8_capacity
from src.provenance import provenance

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'quadrant')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'results', 'figures')

OBJ_ORDER = ['recon', 'jepa', 'jepa_ctrl', 'jepa_invdyn', 'jepa_reward', 'oracle']
CELLS = [1, 2, 3, 4]
# jepa_invdyn is the only objective trained on informative-action data in the default matrix
# (its rescue should track action informativeness); cell4_failure overrides this to random.
INFORMATIVE = {'jepa_invdyn'}

SMOKE = dict(n_steps=150, train_n=2500, eval_n=800, batch_size=128, lr=1e-3,
             latent_dim=64, mi_epochs=150, seeds=[0], predictability=0.5,
             reward_fractions=[0.02, 0.1, 0.5, 1.0], latent_dims=[16, 64])
FULL = dict(n_steps=4000, train_n=20000, eval_n=4000, batch_size=128, lr=1e-3,
            latent_dim=128, mi_epochs=300, seeds=[0, 1, 2], predictability=0.5,
            reward_fractions=[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
            latent_dims=[16, 64, 128, 256, 512, 1024])
IMG_SIZE = 12
THRESHOLD = 0.75  # linear probe acc (chance 0.5) above which we call the feature "retained"


def make_env(env_name, cell, predictability, seed):
    """Build a cell-style env. switch_color/gridworld only instantiate cell 4 (exo+relevant)."""
    if env_name == 'quadrant':
        return QuadrantEnv(make_cell(cell, predictability), IMG_SIZE, seed=seed)
    if cell != 4:
        raise ValueError(f"{env_name} only instantiates cell 4 (exo+relevant), got cell {cell}")
    if env_name == 'switch_color':
        return SwitchColorEnv(p_repeat=predictability, img_size=IMG_SIZE, seed=seed)
    if env_name == 'gridworld':
        return GridWorldHiddenRuleEnv(predictability=predictability, img_size=IMG_SIZE, seed=seed)
    raise ValueError(f"unknown env {env_name}")


def make_datasets(env_name, cell, seed, cfg):
    """Random-policy train, informative-policy train, and a DISJOINT random-policy eval."""
    p = cfg['predictability']
    rand = make_env(env_name, cell, p, seed).sample_transitions(cfg['train_n'], 'random')
    info = make_env(env_name, cell, p, seed + 10000).sample_transitions(cfg['train_n'], 'informative')
    ev = make_env(env_name, cell, p, seed + 100000).sample_transitions(cfg['eval_n'], 'random')
    feature_name = make_env(env_name, cell, p, seed).feature_name
    return feature_name, rand, info, ev


def with_reward_fraction(data, fraction, seed):
    """Add a reward_mask keeping a `fraction` of transitions labelled (rest masked out)."""
    rng = np.random.RandomState(seed + 777)
    n = len(data['obs'])
    mask = (rng.random_sample(n) < fraction).astype(np.float32)
    if mask.sum() == 0:
        mask[rng.randint(n)] = 1.0  # ponytail: guarantee >=1 label so BCE denom != 0
    d = dict(data)
    d['reward_mask'] = mask
    return d


def probe(model, ev, feature_name, device):
    latents = get_latents(model, ev['obs'], device=device)
    acc, _ = linear_probe_accuracy(latents, ev['labels'][feature_name])
    return float(acc)


def result_path(config_name, env_name, cell, seed, obj, tag, extra=''):
    # Default matrix on quadrant keeps the legacy M3 filename so skip-if-done reuses those JSONs.
    if config_name == 'matrix' and env_name == 'quadrant' and not extra:
        base = f"cell{cell}_seed{seed}_{obj}{tag}"
    else:
        base = f"{config_name}_{env_name}_cell{cell}_seed{seed}_{obj}{extra}{tag}"
    return os.path.join(RESULTS_DIR, base + ".json")


# --------------------------------------------------------------------------- matrix / cell4_failure

def run_matrix(cfg, env_name, device, smoke, cells, objectives, informative, config_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = '_smoke' if smoke else ''
    results = []
    enc_sigs = {}

    for cell in cells:
        for seed in cfg['seeds']:
            feature_name, rand, info, ev = make_datasets(env_name, cell, seed, cfg)
            eval_labels = ev['labels'][feature_name]
            for obj in objectives:
                rpath = result_path(config_name, env_name, cell, seed, obj, tag)
                if os.path.exists(rpath):
                    with open(rpath) as f:
                        results.append(json.load(f))
                    print(f"[skip] {os.path.basename(rpath)}")
                    continue

                print(f"=== {config_name} | {env_name} | {obj} | cell {cell} | seed {seed} ===")
                torch.manual_seed(seed)
                policy = 'informative' if obj in informative else 'random'
                if obj == 'oracle':
                    latents = np.eye(2, dtype=np.float32)[eval_labels]
                    acc, _ = linear_probe_accuracy(latents, eval_labels)
                    mi = estimate_mi_infonce(latents, eval_labels, epochs=cfg['mi_epochs'], device=device)
                else:
                    train_data = info if policy == 'informative' else rand
                    model = train_objective(obj, train_data, n_steps=cfg['n_steps'],
                                            img_size=IMG_SIZE, latent_dim=cfg['latent_dim'],
                                            device=device, lr=cfg['lr'], batch_size=cfg['batch_size'])
                    enc_sigs[obj] = encoder_signature(model)
                    latents = get_latents(model, ev['obs'], device=device)
                    acc, _ = linear_probe_accuracy(latents, eval_labels)
                    mi = estimate_mi_infonce(latents, eval_labels, epochs=cfg['mi_epochs'], device=device)

                result = {
                    'objective': obj, 'cell': cell, 'seed': seed, 'env': env_name,
                    'config': config_name, 'feature_name': feature_name,
                    'predictability': cfg['predictability'], 'signal': OBJECTIVES[obj]['signal'],
                    'action_policy': policy, 'linear_probe_acc': float(acc), 'mi_infonce': float(mi),
                    'retained': bool(acc >= THRESHOLD),
                    '_provenance': provenance(config=cfg, cell=cell, seed=seed, objective=obj,
                                              env=env_name, config_name=config_name,
                                              img_size=IMG_SIZE, threshold=THRESHOLD),
                }
                with open(rpath, 'w') as f:
                    json.dump(result, f, indent=2)
                results.append(result)
                print(f"  -> probe_acc={acc:.3f} mi={mi:.3f} retained={result['retained']}")

    sigs = set(enc_sigs.values())
    assert len(sigs) <= 1, f"INVARIANT 1 VIOLATED: encoders differ across {list(enc_sigs)}"
    if enc_sigs:
        print(f"\n[invariant 1 OK] identical encoder across {sorted(enc_sigs)} "
              f"({len(next(iter(sigs)))} params)")
    return results, tag


def build_matrix(results, objectives, cells):
    m = np.full((len(objectives), len(cells)), np.nan)
    for i, obj in enumerate(objectives):
        for j, cell in enumerate(cells):
            vals = [r['linear_probe_acc'] for r in results
                    if r['objective'] == obj and r['cell'] == cell]
            if vals:
                m[i, j] = float(np.mean(vals))
    return m


# --------------------------------------------------------------------------- min_reward_signal

def run_min_reward(cfg, env_name, device, smoke):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = '_smoke' if smoke else ''
    cell, fracs = 4, cfg['reward_fractions']
    means = []
    for frac in fracs:
        accs = []
        for seed in cfg['seeds']:
            feature_name, rand, info, ev = make_datasets(env_name, cell, seed, cfg)
            rpath = result_path('minreward', env_name, cell, seed, 'jepa_reward', tag,
                                extra=f"_frac{frac:g}")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    accs.append(json.load(f)['linear_probe_acc'])
                print(f"[skip] {os.path.basename(rpath)}")
                continue
            print(f"=== min_reward | {env_name} | frac={frac:g} | seed {seed} ===")
            torch.manual_seed(seed)
            data = with_reward_fraction(rand, frac, seed)
            model = train_objective('jepa_reward', data, n_steps=cfg['n_steps'],
                                    img_size=IMG_SIZE, latent_dim=cfg['latent_dim'],
                                    device=device, lr=cfg['lr'], batch_size=cfg['batch_size'])
            acc = probe(model, ev, feature_name, device)
            n_labels = int(data['reward_mask'].sum())
            result = {'objective': 'jepa_reward', 'cell': cell, 'seed': seed, 'env': env_name,
                      'config': 'min_reward_signal', 'reward_label_fraction': frac,
                      'n_reward_labels': n_labels, 'linear_probe_acc': acc,
                      'retained': bool(acc >= THRESHOLD),
                      '_provenance': provenance(config=cfg, cell=cell, seed=seed, env=env_name,
                                                reward_label_fraction=frac, img_size=IMG_SIZE)}
            with open(rpath, 'w') as f:
                json.dump(result, f, indent=2)
            accs.append(acc)
            print(f"  -> n_labels={n_labels} probe_acc={acc:.3f} retained={result['retained']}")
        means.append(float(np.mean(accs)))

    crossing = [f for f, m in zip(fracs, means) if m >= THRESHOLD]
    min_frac = min(crossing) if crossing else None
    summary = {'config': 'min_reward_signal', 'env': env_name, 'cell': cell,
               'threshold': THRESHOLD, 'reward_fractions': fracs, 'mean_acc': means,
               'min_fraction_passing': min_frac}
    with open(os.path.join(os.path.dirname(RESULTS_DIR), f'min_reward_signal_{env_name}{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    figure7_min_reward_signal(fracs, means, FIG_DIR, threshold=THRESHOLD, min_frac=min_frac,
                              env_name=env_name, name=f'figure7_min_reward_signal_{env_name}{tag}')
    print(f"\nmin reward_label_fraction passing (acc>={THRESHOLD}): {min_frac}")
    for f_, m_ in zip(fracs, means):
        print(f"  frac={f_:<6g} mean_acc={m_:.3f} {'PASS' if m_ >= THRESHOLD else 'fail'}")


# --------------------------------------------------------------------------- capacity_sweep

def run_capacity(cfg, device, smoke):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = '_smoke' if smoke else ''
    cell, dims = 4, cfg['latent_dims']
    envs = ['quadrant', 'switch_color']      # cell 4 in both surface forms
    objectives = ['jepa', 'jepa_reward']     # jepa drops it (flat-low); jepa_reward keeps it
    series = {}
    for env_name in envs:
        for obj in objectives:
            accs_by_dim = []
            for ld in dims:
                accs = []
                for seed in cfg['seeds']:
                    feature_name, rand, info, ev = make_datasets(env_name, cell, seed, cfg)
                    rpath = result_path('capacity', env_name, cell, seed, obj, tag, extra=f"_ld{ld}")
                    if os.path.exists(rpath):
                        with open(rpath) as f:
                            accs.append(json.load(f)['linear_probe_acc'])
                        print(f"[skip] {os.path.basename(rpath)}")
                        continue
                    print(f"=== capacity | {env_name} | {obj} | latent_dim={ld} | seed {seed} ===")
                    torch.manual_seed(seed)
                    model = train_objective(obj, rand, n_steps=cfg['n_steps'], img_size=IMG_SIZE,
                                            latent_dim=ld, device=device, lr=cfg['lr'],
                                            batch_size=cfg['batch_size'])
                    acc = probe(model, ev, feature_name, device)
                    result = {'objective': obj, 'cell': cell, 'seed': seed, 'env': env_name,
                              'config': 'capacity_sweep', 'latent_dim': ld, 'linear_probe_acc': acc,
                              'retained': bool(acc >= THRESHOLD),
                              '_provenance': provenance(config=cfg, cell=cell, seed=seed, env=env_name,
                                                        latent_dim=ld, img_size=IMG_SIZE)}
                    with open(rpath, 'w') as f:
                        json.dump(result, f, indent=2)
                    accs.append(acc)
                    print(f"  -> probe_acc={acc:.3f} retained={result['retained']}")
                accs_by_dim.append(float(np.mean(accs)))
            series[f"{env_name}/{obj}"] = accs_by_dim

    summary = {'config': 'capacity_sweep', 'cell': cell, 'latent_dims': dims,
               'threshold': THRESHOLD, 'series': series}
    with open(os.path.join(os.path.dirname(RESULTS_DIR), f'capacity_sweep{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    figure8_capacity(series, dims, FIG_DIR, threshold=THRESHOLD, name=f'figure8_capacity{tag}')
    print("\ncapacity sweep (mean cell-4 probe acc, flat => not capacity-bound):")
    print(" " * 18 + "".join(f"{d:<8}" for d in dims))
    for label, accs in series.items():
        print(f"{label:<18}" + "".join(f"{a:<8.2f}" for a in accs))


# --------------------------------------------------------------------------- entrypoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='matrix',
                    choices=['matrix', 'cell4_failure', 'min_reward_signal', 'capacity_sweep'])
    ap.add_argument('--env', default='quadrant',
                    choices=['switch_color', 'gridworld', 'quadrant'])
    ap.add_argument('--full', action='store_true', help='bigger sweep (default: CPU smoke)')
    args = ap.parse_args()
    cfg = dict(FULL if args.full else SMOKE)
    smoke = not args.full
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Quadrant {args.config} | env={args.env} | {'full' if args.full else 'smoke'} "
          f"| device={device}")

    if args.config in ('matrix', 'cell4_failure'):
        cells = CELLS if (args.config == 'matrix' and args.env == 'quadrant') else [4]
        objectives = OBJ_ORDER
        informative = INFORMATIVE if args.config == 'matrix' else set()  # cell4_failure: all random
        results, tag = run_matrix(cfg, args.env, device, smoke, cells, objectives,
                                  informative, args.config)
        matrix = build_matrix(results, objectives, cells)
        out = {'objectives': objectives, 'cells': cells, 'threshold': THRESHOLD, 'env': args.env,
               'config': args.config, 'config_params': cfg,
               'matrix_linear_probe_acc': matrix.tolist(), 'results': results}
        mpath = os.path.join(os.path.dirname(RESULTS_DIR),
                             f'quadrant_matrix_{args.config}_{args.env}{tag}.json'
                             if args.config != 'matrix' or args.env != 'quadrant'
                             else f'quadrant_matrix{tag}.json')
        with open(mpath, 'w') as f:
            json.dump(out, f, indent=2)
        fig_name = (f'figure6_quadrant_matrix{tag}' if args.config == 'matrix' and args.env == 'quadrant'
                    else f'figure6_{args.config}_{args.env}{tag}')
        quadrant_heatmap(matrix, objectives, cells, FIG_DIR, threshold=THRESHOLD, name=fig_name)
        print(f"\nMatrix JSON -> {mpath}")
        print("\n" + " " * 14 + "".join(f"cell{c:<5}" for c in cells))
        for i, obj in enumerate(objectives):
            print(f"{obj:<14}" + "".join(f"{matrix[i, j]:<9.2f}" for j in range(len(cells))))
    elif args.config == 'min_reward_signal':
        run_min_reward(cfg, args.env, device, smoke)
    elif args.config == 'capacity_sweep':
        run_capacity(cfg, device, smoke)


if __name__ == '__main__':
    main()
