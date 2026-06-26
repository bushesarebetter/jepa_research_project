# results/ schema

## What lives here

| Artifact | Role | Tracked in git |
|---|---|---|
| `p{p:.2f}_seed{seed}_{model}_result.json` | **Source of truth.** One trained-model evaluation. skip-if-done keys on this file. | yes |
| `all_results.json` | List of every result dict from the last sweep. | yes |
| `decision.json` | GO / STOP / REFRAME verdict from `make_decision`. | yes |
| `config.json` | Config of the **most recent** run (overwritten each run). | yes |
| `figures/figure*.{png,pdf}` | Generated figures. | yes |
| `p{p:.2f}_seed{seed}_dataset.npz` | Cached env rollouts. Regenerable. | no (gitignored) |
| `../checkpoints/*.pt` | Model weights. Regenerable. skip-if-done does **not** key on these. | no (gitignored) |

## Per-result JSON schema

Scientific keys (stable since the 36-run sweep — never removed or renamed):

```json
{
  "model_type": "jepa|ae|jepa_invdyn",
  "p_repeat": 0.5,
  "seed": 0,
  "linear_probe_acc": 0.0, "linear_probe_std": 0.0,
  "nonlinear_probe_acc": 0.0, "nonlinear_probe_std": 0.0,
  "effective_rank": 0.0,
  "centroid_dist_norm": 0.0,
  "control_regret": 0.0, "policy_reward": 0.0
}
```

Robustness instrumentation (added M1, additive — present on every run from M1 onward;
legacy keys above are untouched, so `make_decision`/figures still index by them):

```json
{
  "switch_contrast": 1.0,            // salience knob; 1.0 == legacy block colors (0.95/0.05)
  "control_regret_rollout": 0.0,     // independent rollout regret (oracle - achieved); figures use this
  "rollout_reward": 1.0,             // mean reward of the BC policy acting greedily in real episodes
  "oracle_reward": 1.0
}
```

Information estimate (added M2, additive — present on every run from M2 onward; pre-M2 JSONs
lack it and figure 3 skips those points rather than back-filling from the probe):

```json
{
  "mi_infonce": 0.0                   // estimated I(Z; switch_color) in nats (InfoNCE lower bound,
                                      // held-out critic); the de-circularized x-axis for figure 3
}
```

`control_regret` / `policy_reward` remain the legacy *probe-based* regret (kept for the historical
schema; no longer used by the regret-vs-information figure). Smoke runs (`--smoke`) write artifacts
with a `_smoke` filename suffix, and non-default contrasts add a `_c{contrast}` tag, so neither
collides with the full-size contrast-1.0 results.

Provenance (added M0, additive, namespaced — present on every run from M0 onward;
legacy pre-M0 JSONs predate it and are left as historical record):

```json
{
  "_provenance": {
    "git_commit": "<sha or 'uncommitted'>",
    "versions": {"python": "...", "platform": "...", "torch": "...", "numpy": "...", "sklearn": "...", "matplotlib": "..."},
    "config": { "...full sweep config dict..." },
    "p_repeat": 0.5, "seed": 0, "model_type": "jepa", "n_steps": 30000, "lr": 0.001, "batch_size": 256
  }
}
```

Readers (`make_decision`, `figures.py`) index by scientific key only, so `_provenance`
is invisible to them — the 36-run schema is unchanged.
