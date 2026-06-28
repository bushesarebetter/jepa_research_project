#!/usr/bin/env python
"""De-circularized regret-vs-information experiment (addresses PAPER_FACTS GAP / D5).

Trains the switch-color predictability sweep and, for each run, records BOTH:
  - mi_infonce            : InfoNCE lower bound on I(Z; switch_color), via a separate critic
  - control_regret_rollout: regret of an INDEPENDENT behaviour-cloned MLP policy rolled out
                            in the actual environment (distinct from the linear probe)
Neither quantity is derived from the linear probe, so plotting regret (y) vs MI (x) is
non-circular -- unlike the shipped figure3 (probe-acc vs probe-derived regret).

This is a NEW, reduced-budget run (CPU); it is deliberately separate from the full-training
numbers in PAPER_FACTS.md. Results -> results/regret_mi_sweep.json (does NOT touch all_results.json).
"""
import argparse, json, os, time
import numpy as np
import torch

from src.training import train_model, get_latents
from src.evaluation import linear_probe_accuracy, effective_rank, control_regret_rollout
from src.information import estimate_mi_infonce
from src.provenance import provenance

ROOT = os.path.dirname(__file__)
RESULTS = os.path.join(ROOT, 'results')


def dataset_path(p, seed):
    return os.path.join(RESULTS, f"p{p:.2f}_seed{seed}_dataset.npz")


def get_dataset(p, seed, train_episodes, eval_episodes, episode_length):
    path = dataset_path(p, seed)
    if os.path.exists(path):
        npz = np.load(path)
        return {k: npz[k] for k in npz.files}
    from src.env import SwitchColorEnv
    tr = SwitchColorEnv(p_repeat=p, seed=seed * 2 + 1).collect_dataset(train_episodes, episode_length)
    ev = SwitchColorEnv(p_repeat=p, seed=seed * 2 + 2 + 100000).collect_dataset(eval_episodes, episode_length)
    b = {}
    for k, v in tr.items(): b[f'train_{k}'] = v
    for k, v in ev.items(): b[f'eval_{k}'] = v
    np.savez_compressed(path, **b)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-steps', type=int, default=4000)
    ap.add_argument('--p-values', type=float, nargs='+', default=[0.5, 0.65, 0.8, 1.0])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--models', nargs='+', default=['jepa', 'ae', 'jepa_invdyn'])
    ap.add_argument('--mi-epochs', type=int, default=250)
    ap.add_argument('--rollout-episodes', type=int, default=40)
    ap.add_argument('--rollout-length', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--out', default=os.path.join(RESULTS, 'regret_mi_sweep.json'))
    args = ap.parse_args()

    torch.set_num_threads(6)  # measured fastest for this conv net on this CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  n_steps={args.n_steps}  models={args.models}  "
          f"p={args.p_values}  seeds={args.seeds}", flush=True)

    rows = []
    t0 = time.time()
    for p in args.p_values:
        for seed in args.seeds:
            data = get_dataset(p, seed, 200, 80, 100)
            train = {'obs': data['train_obs'], 'next_obs': data['train_next_obs'],
                     'switch_color': data['train_switch_color'], 'action': data['train_action'],
                     'reward': data['train_reward']}
            for mt in args.models:
                torch.manual_seed(seed)
                model, _ = train_model(mt, train, n_steps=args.n_steps, lr=1e-3,
                                       batch_size=args.batch_size, device=device, verbose=False)
                lat = get_latents(model, data['eval_obs'], device=device)
                lab = data['eval_switch_color']
                acc, _ = linear_probe_accuracy(lat, lab)
                er = effective_rank(lat)
                mi = estimate_mi_infonce(lat, lab, epochs=args.mi_epochs, device=device)
                roll = control_regret_rollout(model, 'switch_color', p, seed=seed * 2 + 5 + 400000,
                                              n_episodes=args.rollout_episodes, device=device,
                                              episode_length=args.rollout_length)
                row = {'model_type': mt, 'p_repeat': p, 'seed': seed,
                       'linear_probe_acc': float(acc), 'effective_rank': float(er),
                       'mi_infonce': float(mi),
                       'control_regret_rollout': float(roll['regret']),
                       'rollout_reward': float(roll['rollout_reward'])}
                rows.append(row)
                print(f"[{time.time()-t0:6.0f}s] {mt:12s} p={p:.2f} s={seed} "
                      f"acc={acc:.3f} mi={mi:.3f} regret={roll['regret']:.3f} er={er:.1f}", flush=True)

    out = {'config': {'n_steps': args.n_steps, 'p_values': args.p_values, 'seeds': args.seeds,
                      'models': args.models, 'mi_epochs': args.mi_epochs,
                      'rollout_episodes': args.rollout_episodes, 'rollout_length': args.rollout_length,
                      'note': 'NEW reduced-budget CPU run; separate from PAPER_FACTS full-training numbers'},
           '_provenance': provenance(experiment='regret_mi_sweep', n_steps=args.n_steps),
           'results': rows}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}  ({len(rows)} runs, {time.time()-t0:.0f}s total)", flush=True)


if __name__ == '__main__':
    main()
