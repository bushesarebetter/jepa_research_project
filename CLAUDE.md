# CLAUDE.md — JEPA control-loss experiment

**Re-read this file at the start of every prompt.** It is the project's single source of truth
for the claim, the spine, the invariants, the module map, and the milestone plan.

## The claim under test

A JEPA-style latent self-prediction objective **discards a feature that is control-relevant but
hard to predict**, even though the feature is trivially encodable — whereas a reconstruction
baseline with an *identical encoder* keeps it. The decisive manipulation is a **predictability
knob** (`p_repeat ∈ [0.5, 1.0]`) that varies temporal predictability while holding
control-relevance fixed.

## The spine — a 2×2 over features

- **Axis A**: controllable (the agent's action moves it) vs uncontrollable (exogenous).
- **Axis B**: relevant (sets reward / optimal action) vs irrelevant.

| cell | feature | who keeps it |
|---|---|---|
| 1 controllable + relevant | everyone keeps it |
| 2 controllable + irrelevant | controllability methods waste capacity; bisimulation drops |
| 3 uncontrollable + irrelevant | the classic distractor; reconstruction keeps it (bad) |
| 4 **uncontrollable + relevant** | **THE HARD CASE** — pure prediction drops it; reconstruction keeps it but pays full pixel cost; controllability drops it; inverse dynamics drops it unless actions encode it; only a reward-grounded signal reliably keeps it |

The current `SwitchColorEnv` already instantiates **cell 3** (predictable sinusoidal background =
uncontrollable+irrelevant distractor) and **cell 4** (`switch_color` = uncontrollable+relevant,
predictability tunable via `p_repeat`). Cells 1–2 (controllable features) arrive with
`GridWorldHiddenRuleEnv` in M3.

## The 5 scientific invariants (encode as asserts/tests; never optimize away)

1. **Encoder byte-identical across every objective** — assert `state_dict` keys+shapes match.
2. **Anti-collapse uses BOTH VICReg variance AND covariance** — assert JEPA latent effective rank > 5 on a smoke run.
3. **JEPA uses BYOL-style normalized (cosine) prediction loss + EMA target** (decay 0.996, updated every step).
4. **Probes/regret on a DISJOINT eval seed; always probe the ONLINE/context encoder, never the EMA target.**
5. **`latent_dim` configurable; skip-if-done keyed on result JSONs, not checkpoints.**

### Current state vs invariants (as of M0 — to reconcile in M1/M2)

| invariant | status | gap |
|---|---|---|
| 1 shared encoder | ✅ met | shared `Encoder` class across JEPA/AE/InvDyn |
| 2 var **and** cov | ❌ **not met** | `models.py` has the variance term only, no covariance → fix in **M2** |
| 3 cosine prediction | ❌ **not met** | `JEPAModel.forward` uses `F.mse_loss`, not cosine → fix in **M2** (EMA part ✅) |
| 4 disjoint seed + online encoder | ✅ met | eval seeds offset `+100000`; `get_latent` returns `online_encoder` |
| 5 configurable latent_dim + JSON skip | ⚠️ partial | skip-on-JSON ✅; `latent_dim` hardcoded, not config-wired → wire in **M1** |

> The prompt's prose described the baseline as already cosine+var+cov and a 12×12 frame.
> The actual code uses MSE+var-only and a **32×32** frame (switch block top-left 8×8). The plan
> brings JEPA up to the invariants in M2 with the legacy behavior kept config-selectable, so the
> prior 36-run sweep stays reproducible.

## Module map

```
run_experiment.py     thin orchestrator: CLI, sweep, GO/NO-GO decision, figures entrypoint
verify.py             import + provenance smoke (grows into the invariant test suite)
configs/              YAML configs — smoke.yaml (CPU) + full.yaml (GPU)        [populated M1]
src/
  env.py              SwitchColorEnv (predictability knob p_repeat)            [+GridWorld M3]
  models.py           Encoder, Decoder, JEPAModel, AutoencoderModel, JEPAInvDynModel, build_model
  training.py         PairDataset/ObsDataset, make_dataloader, train_model, get_latents
  evaluation.py       linear/nonlinear probe, effective_rank, centroid_distance, control_regret_{probe,rollout}
  aggregate.py        mean ±95% CI per (model,p); flags non-monotonic JEPA-vs-p curve        [added M1]
  figures.py          figure1..5, generate_all_figures
  provenance.py       provenance() — git commit + full config + lib versions for every result JSON
results/              result JSONs (source of truth), all_results.json, decision.json, figures/, SCHEMA.md
checkpoints/          *.pt weights (regenerable cache; gitignored)
```

## How to work (whole project, 8 prompts)

- Small, single-responsibility modules + a thin CLI/config layer. **Never** one monolithic script.
- Everything runs two ways: a tiny **CPU smoke** config (seconds) and a full **GPU** config. Default `--device` to **auto**.
- **Do NOT download large model weights during the build** — that is M6, behind an explicit guard.
- **Never break existing behavior**: the prior 36-run sweep must still run and keep its result schema (additions are additive only).

## The 8 milestones (see PLAN.md for done-criteria; STATUS.md for checkboxes)

- **M0** scaffolding + plan + provenance + git  ← *this prompt*
- **M1** config/CLI layer (`--config`, `--device auto`, wire `latent_dim`); assert invariants 1,4,5
- **M2** invariant-compliant JEPA (cosine + var+cov, legacy selectable); assert invariants 2,3
- **M3** complete the 2×2 env (`GridWorldHiddenRuleEnv`, cells 1–2; per-cell labels)
- **M4** objective zoo (bisimulation, reward-grounded, controllability) on the shared encoder; assert invariant 1 across all
- **M5** full evaluation matrix (objective × cell × predictability) + generalized decision + figures
- **M6** pretrained-encoder comparison — GUARDED download, never during build/smoke
- **M7** full reproduction + report; verify.py asserts all 5 invariants end-to-end

> M1–M7 details are provisional; each later prompt refines its own milestone. Update STATUS.md
> when a milestone completes.
