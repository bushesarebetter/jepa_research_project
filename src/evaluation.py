"""Evaluation metrics: probes, effective rank, bisimulation proxy, control regret.

All evaluation is performed on held-out data with a frozen (eval-mode, no-grad) encoder.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier  # also the rollout policy head
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .env import SwitchColorEnv
from .training import get_latents


def linear_probe_accuracy(latents, labels, cv=5, seed=0):
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    scores = cross_val_score(clf, latents, labels, cv=cv)
    return float(scores.mean()), float(scores.std())


def nonlinear_probe_accuracy(latents, labels, cv=5, seed=0):
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=seed)
    scores = cross_val_score(clf, latents, labels, cv=cv)
    return float(scores.mean()), float(scores.std())


def effective_rank(latents):
    scaled = StandardScaler().fit_transform(latents)
    _, sv, _ = np.linalg.svd(scaled, full_matrices=False)
    sv_norm = sv / sv.sum()
    eff_rank = np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-10)))
    if eff_rank < 3:
        print(f"  WARNING: effective_rank={eff_rank:.2f} < 3 -- possible representation collapse")
    return float(eff_rank)


def centroid_distance(latents, labels):
    z_c0 = latents[labels == 0].mean(0)
    z_c1 = latents[labels == 1].mean(0)
    centroid_dist = np.linalg.norm(z_c0 - z_c1)
    within_std = np.sqrt(
        (latents[labels == 0].var(0).mean() + latents[labels == 1].var(0).mean()) / 2
    )
    return float(centroid_dist / (within_std + 1e-8))


def control_regret_probe(model, p_repeat, device='cpu', n_episodes=50, episode_length=100,
                         train_seed=1000, eval_seed=2000, switch_contrast=1.0):
    """Legacy probe-based regret: fit a linear policy on oracle (obs, a*) pairs, then score it
    against the SAME pre-collected eval trajectory (no acting). Kept for the historical schema;
    NOT used for the regret-vs-information figure (use control_regret_rollout for that)."""
    train_env = SwitchColorEnv(p_repeat=p_repeat, seed=train_seed, switch_contrast=switch_contrast)
    train_data = train_env.collect_dataset(n_episodes, episode_length)
    eval_env = SwitchColorEnv(p_repeat=p_repeat, seed=eval_seed, switch_contrast=switch_contrast)
    eval_data = eval_env.collect_dataset(n_episodes, episode_length)

    z_train = get_latents(model, train_data['obs'], device=device)
    z_eval = get_latents(model, eval_data['obs'], device=device)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(z_train, train_data['switch_color'])
    policy_action = clf.predict(z_eval)

    policy_reward = float((policy_action == eval_data['switch_color']).mean())
    oracle_reward = 1.0
    regret = oracle_reward - policy_reward
    return regret, policy_reward


_ENVS = {'switch_color': SwitchColorEnv}


def control_regret_rollout(model, env_name, p, *, seed, n_episodes, device,
                           episode_length=100, switch_contrast=1.0):
    """Independent rollout-based regret.

    Trains a SEPARATE 1-hidden-layer MLP policy head on the frozen latent by behaviour cloning
    on the oracle action a* (distinct from the linear probe object used for retention), then rolls
    out *actual* episodes acting greedily and measures mean achieved reward. Returns
    {rollout_reward, oracle_reward, regret}. Probes the ONLINE encoder via get_latents.
    """
    env_cls = _ENVS[env_name]

    # Behaviour-cloning set from oracle rollouts (collect_dataset acts with action == a*).
    bc_env = env_cls(p_repeat=p, seed=seed, switch_contrast=switch_contrast)
    bc_data = bc_env.collect_dataset(n_episodes, episode_length)
    z_bc = get_latents(model, bc_data['obs'], device=device)
    policy = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed)
    policy.fit(z_bc, bc_data['action'])

    # Vectorized greedy rollout: switch_color evolves independently of the action, so we step
    # all episodes in lockstep and batch-encode each timestep (episode_length forward passes).
    envs = [env_cls(p_repeat=p, seed=seed + 1 + i, switch_contrast=switch_contrast)
            for i in range(n_episodes)]
    obs = np.stack([e.reset() for e in envs])
    total, count = 0.0, 0
    for _ in range(episode_length):
        actions = policy.predict(get_latents(model, obs, device=device))
        next_obs = []
        for e, a in zip(envs, actions):
            o, r, _ = e.step(int(a))
            total += r
            count += 1
            next_obs.append(o)
        obs = np.stack(next_obs)

    rollout_reward = total / count
    oracle_reward = 1.0
    return {'rollout_reward': rollout_reward, 'oracle_reward': oracle_reward,
            'regret': oracle_reward - rollout_reward}
