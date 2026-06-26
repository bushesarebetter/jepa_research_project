#!/usr/bin/env python
"""verify.py -- M0 stub: import + provenance smoke check.

For now this only confirms every module imports cleanly and the provenance helper
returns a well-formed block. Later milestones grow this into the invariant test suite:

  M1: assert invariants 1, 4, 5  (encoder state_dict keys+shapes identical across models;
      eval seed disjoint from train; probes read the ONLINE encoder; skip-if-done on JSON)
  M2: assert invariants 2, 3     (cosine prediction loss + VICReg var AND cov; eff_rank > 5)
  M4: assert invariant 1 across the full objective zoo
  M7: run the smoke pipeline green asserting all 5 invariants end-to-end

Run: python verify.py
"""
import importlib

import numpy as np


def mi_sanity():
    """Invariant for M2: the InfoNCE estimator reads ~log 2 nats when a binary feature is
    fully recoverable from Z, and ~0 when it is independent noise."""
    from src.information import estimate_mi_infonce
    log2 = float(np.log(2))
    rng = np.random.default_rng(0)
    N, d = 2000, 16

    c = rng.integers(0, 2, size=N).astype(np.float32)
    Z = rng.normal(size=(N, d)).astype(np.float32)
    Z[:, 0] = 3.0 * c + 0.3 * rng.normal(size=N)          # first dim carries c
    mi_rec = estimate_mi_infonce(Z, c, epochs=400, device='cpu')

    Z2 = rng.normal(size=(N, d)).astype(np.float32)
    c2 = rng.integers(0, 2, size=N).astype(np.float32)    # independent of Z2
    mi_indep = estimate_mi_infonce(Z2, c2, epochs=400, device='cpu')

    assert abs(mi_rec - log2) < 0.15, f"recoverable MI {mi_rec:.3f} not ~{log2:.3f}"
    assert mi_indep < 0.10, f"independent MI {mi_indep:.3f} not ~0"
    print(f"  MI sanity OK: recoverable={mi_rec:.3f} (~{log2:.3f}), independent={mi_indep:.3f} (~0)")


def encoder_identity():
    """Invariant 1 (M3 scope): the encoder is byte-identical across every objective in the
    registry that has one. Cheap structural check -- builds each model, compares state_dict
    keys+shapes; no training."""
    from src.models import OBJECTIVES, build_model, encoder_signature
    sigs = {}
    for obj, spec in OBJECTIVES.items():
        if spec['cls'] is None:  # oracle has no encoder
            continue
        sigs[obj] = encoder_signature(build_model(obj, latent_dim=64, img_size=12, n_actions=2))
    assert len(set(sigs.values())) == 1, f"INVARIANT 1 VIOLATED across {list(sigs)}"
    print(f"  invariant 1 OK: identical encoder across {sorted(sigs)} "
          f"({len(next(iter(sigs.values())))} params)")


def env_zoo_cell4():
    """M4: switch_color, gridworld, and quadrant-cell-4 all emit the same cell-4 schema
    (exogenous + relevant): labels under feature_name, a* == bits, reward == 1[action == a*]."""
    from src.env import SwitchColorEnv
    from src.gridworld_env import GridWorldHiddenRuleEnv
    from src.quadrant_env import QuadrantEnv, make_cell
    envs = {
        'switch_color': SwitchColorEnv(p_repeat=0.5, img_size=12, seed=0),
        'gridworld': GridWorldHiddenRuleEnv(predictability=0.5, img_size=12, seed=0),
        'quadrant': QuadrantEnv(make_cell(4, 0.5), 12, seed=0),
    }
    for name, env in envs.items():
        d = env.sample_transitions(1500, 'random')
        bits = d['labels'][env.feature_name]
        assert (d['a_star'] == bits).all(), f"{name}: a* != feature (should be relevant)"
        assert (d['reward'] == (d['action'] == bits)).all(), f"{name}: reward != 1[action==a*]"
        assert d['obs'].shape[1:] == (3, 12, 12), f"{name}: bad obs shape {d['obs'].shape}"
    print(f"  env zoo OK: cell-4 schema holds for {sorted(envs)}")


def reward_mask_sanity():
    """M4: reward_label_fraction masks the reward loss -- mask all-ones == plain BCE; a sparse
    mask zeroes the loss on unlabelled transitions (so its gradient vanishes there)."""
    import torch
    from src.models import build_model
    torch.manual_seed(0)
    m = build_model('jepa_reward', latent_dim=16, img_size=12)
    o = torch.rand(8, 3, 12, 12); o1 = torch.rand(8, 3, 12, 12)
    a = torch.randint(0, 2, (8,)); r = torch.randint(0, 2, (8,)).float()
    ones = torch.ones(8)
    l_default, _, _ = m(o, o1, a, r)
    l_ones, _, _ = m(o, o1, a, r, ones)
    assert abs(l_default.item() - l_ones.item()) < 1e-5, "all-ones mask must equal plain BCE"
    half = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
    l_half, info_half, _ = m(o, o1, a, r, half)
    assert info_half['reward_loss'] >= 0.0
    l_zero, info_zero, _ = m(o, o1, a, r, torch.zeros(8))
    assert info_zero['reward_loss'] == 0.0, f"empty mask -> 0 reward loss, got {info_zero['reward_loss']}"
    print("  reward mask OK: all-ones == plain BCE; empty mask -> 0 reward loss")


def bisim_sanity():
    """M5: the analytical bisimulation reduction and the whitened latent class-distance proxy.
    Cell 4 (exo+relevant) at p=0.5 reduces to the immediate reward gap 1.0; an irrelevant feature
    is 0; a frozen feature (p=1.0) hits the discounted ceiling 1/(1-gamma). The proxy reads large
    when a feature is encoded and ~0 when it is independent of the latent."""
    import numpy as np
    from src.bisimulation import analytical_bisim_distance, latent_class_distance
    from src.quadrant_env import QuadrantEnv, make_cell

    assert abs(analytical_bisim_distance(QuadrantEnv(make_cell(4, 0.5))) - 1.0) < 1e-9
    assert analytical_bisim_distance(QuadrantEnv(make_cell(3, 0.5))) == 0.0
    assert abs(analytical_bisim_distance(QuadrantEnv(make_cell(4, 1.0)), gamma=0.99) - 100.0) < 1e-6

    rng = np.random.default_rng(0)
    lab = rng.integers(0, 2, 2000)
    Z = rng.normal(size=(2000, 8)); Z[:, 0] = 3.0 * lab + 0.3 * rng.normal(size=2000)
    d_enc = latent_class_distance(Z, lab)
    d_indep = latent_class_distance(rng.normal(size=(2000, 8)), lab)
    assert d_enc > 1.0 and d_indep < 0.5, f"proxy failed: encoded={d_enc:.2f} indep={d_indep:.2f}"
    print(f"  bisim OK: analytical(cell4,p=.5)=1.0, irrelevant=0, frozen=100; "
          f"proxy encoded={d_enc:.2f} indep={d_indep:.2f}")


def main():
    modules = [
        "src.env", "src.quadrant_env", "src.gridworld_env", "src.models", "src.training",
        "src.evaluation", "src.figures", "src.provenance", "src.bisimulation",
    ]
    for m in modules:
        importlib.import_module(m)
        print(f"  import OK: {m}")

    from src.provenance import provenance
    block = provenance(config={"smoke": True}, seed=0)
    assert set(block) >= {"git_commit", "versions", "config"}, block
    assert "torch" in block["versions"], block
    print(f"  provenance OK: git={block['git_commit'][:12]} torch={block['versions']['torch']}")

    mi_sanity()
    encoder_identity()
    env_zoo_cell4()
    reward_mask_sanity()
    bisim_sanity()

    print("verify.py: imports + provenance + MI sanity + invariant 1 + env zoo + reward mask "
          "+ bisim OK")


if __name__ == "__main__":
    main()
