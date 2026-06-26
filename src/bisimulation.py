"""Analytical bisimulation distance for the cell-style envs + a whitened latent class distance.

Bisimulation metric (Ferns et al. 2004; the object DBC / Zhang et al. 2021 approximate):

    d(s, t) = max_a [ |R(s,a) - R(t,a)| + gamma * W_1^d( P(.|s,a), P(.|t,a) ) ]

For the switch / quadrant / gridworld cell-4 envs the only reward-relevant state variable is the
binary feature value (the predictable background never touches reward and is shared, so two states
differing only in background are bisimilar => distance 0). Take s0 (value=0), s1 (value=1):

  * Immediate reward term. reward = 1[a == a*(v)], with a*(v) = v for a RELEVANT feature and a
    constant for an irrelevant one. So
        max_a |R(s0,a) - R(s1,a)| = 1[a*(0) != a*(1)] = 1  if relevant, else 0.
    (This is |r(0,a*) - r(1,a*)| for a* the optimal action of either state: that action scores 1
    in its own state and 0 in the other.)

  * Future term. The feature is EXOGENOUS: next value repeats w.p. p (= predictability), else
    flips, independent of the action. Backgrounds are bisim-distance 0, so the only ground
    distance is d itself, and with P0 = (p @ v0, 1-p @ v1), P1 = (1-p @ v0, p @ v1),
        W_1^d(P0, P1) = |2p - 1| * d.
    The metric is the fixed point  d = g + gamma*|2p-1|*d  =>  d = g / (1 - gamma*|2p-1|),
    where g is the immediate reward gap. At p = 0.5 the feature is i.i.d. so |2p-1| = 0 and the
    future term vanishes: d = g = 1 for a relevant feature.

So a relevant-but-exogenous feature (cell 4) has bisim distance >= 1 > 0 at every p, yet a pure
latent-prediction objective collapses it to ~0 in latent space (see proposition_numbers.py):
Observation 1 (empirical) — the bisimulation error stays bounded away from 0 at JEPA convergence.
The future-term reduction assumes an exogenous feature; for the p=0.5 case only the immediate term
survives, so it is exact regardless.
"""
import numpy as np


def _predictability(env):
    for attr in ('p_repeat', 'predictability'):     # SwitchColorEnv / GridWorldHiddenRuleEnv
        if hasattr(env, attr):
            return float(getattr(env, attr))
    if hasattr(env, 'spec'):                         # QuadrantEnv: per-cell FeatureSpec
        return float(env.spec.predictability)
    raise AttributeError(f"{type(env).__name__} exposes no predictability knob")


def _relevant(env):
    # QuadrantEnv tags relevance per cell; switch_color/gridworld are cell 4 (always relevant).
    return bool(env.spec.relevant) if hasattr(env, 'spec') else True


def analytical_bisim_distance(env, gamma=0.99):
    """On-policy bisimulation distance between the feature=0 and feature=1 states of a cell-style
    env. Reduces to the immediate reward gap (1 if the feature is relevant, else 0) plus a
    discounted exogenous-transition term that vanishes at p=0.5. See the module docstring."""
    reward_gap = 1.0 if _relevant(env) else 0.0      # max_a |R(0,a) - R(1,a)|
    if reward_gap == 0.0:
        return 0.0
    p = _predictability(env)
    shrink = gamma * abs(2.0 * p - 1.0)              # W_1 coupling of the exogenous next feature
    return reward_gap / (1.0 - shrink)               # fixed point d = g + gamma|2p-1| d


def latent_class_distance(Z, labels, ridge=1e-3, seed=0):
    """Whitened (Mahalanobis) distance between the two class centroids in latent space, CROSS-FIT.

    Whitening by the pooled covariance makes the number comparable across objectives whose latents
    live at different scales (JEPA's LayerNorm'd vector vs the oracle's one-hot). ~0 means the
    feature is collapsed (the classes coincide); large means it is linearly laid out. The ridge
    keeps it finite when the covariance is singular (e.g. the oracle's one-hot latent).

    Cross-fitting is essential: an in-sample Mahalanobis distance inverts a d*d covariance against a
    noisy centroid difference, so in high d / finite n it floors well above 0 even for a collapsed
    feature (a 64-dim JEPA latent reads ~1 from noise alone). We instead fit the whitened axis
    w = Sigma^-1 (mu1 - mu0) on one half and measure the centroid gap (mu1 - mu0) . w on the held-out
    half; under H0 (feature absent) the held-out difference is independent of w, so the estimate is
    ~0. Averaged over both folds, (mu1 - mu0)_te . w_tr estimates the SQUARED distance; sqrt gives
    the distance (clipped at 0)."""
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels)
    if not (labels == 0).any() or not (labels == 1).any():
        return 0.0
    idx = np.random.default_rng(seed).permutation(len(Z))
    folds = (idx[:len(Z) // 2], idx[len(Z) // 2:])
    gsq = []
    for tr, te in (folds, folds[::-1]):
        ytr, yte = labels[tr], labels[te]
        if len({*ytr.tolist()}) < 2 or len({*yte.tolist()}) < 2:
            continue
        Ztr = Z[tr]
        cov = np.cov(Ztr, rowvar=False) + ridge * np.eye(Z.shape[1])
        w = np.linalg.solve(cov, Ztr[ytr == 1].mean(0) - Ztr[ytr == 0].mean(0))   # whitened axis
        diff_te = Z[te][yte == 1].mean(0) - Z[te][yte == 0].mean(0)
        gsq.append(float(diff_te @ w))                                            # ~ squared dist
    if not gsq:
        return 0.0
    return float(np.sqrt(max(np.mean(gsq), 0.0)))


if __name__ == "__main__":  # runnable check: the reduction, irrelevance collapse, and the proxy
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.quadrant_env import QuadrantEnv, make_cell

    d4 = analytical_bisim_distance(QuadrantEnv(make_cell(4, 0.5)))
    assert abs(d4 - 1.0) < 1e-9, f"cell4 p=0.5 bisim should be 1.0, got {d4}"
    d3 = analytical_bisim_distance(QuadrantEnv(make_cell(3, 0.5)))
    assert d3 == 0.0, f"cell3 (irrelevant) bisim should be 0, got {d3}"
    d4_frozen = analytical_bisim_distance(QuadrantEnv(make_cell(4, 1.0)), gamma=0.99)
    assert abs(d4_frozen - 100.0) < 1e-6, f"cell4 p=1.0 -> 1/(1-0.99)=100, got {d4_frozen}"

    rng = np.random.default_rng(0)
    N = 2000
    lab = rng.integers(0, 2, N)
    Z = rng.normal(size=(N, 8)); Z[:, 0] = 3.0 * lab + 0.3 * rng.normal(size=N)
    assert latent_class_distance(Z, lab) > 1.0, "encoded feature should give large class distance"
    Zi = rng.normal(size=(N, 8))
    assert latent_class_distance(Zi, lab) < 0.5, "independent feature should give ~0 distance"
    print("bisimulation OK: reduction (1.0 @ p=.5, 0 irrelevant, 100 @ p=1), proxy separates")
