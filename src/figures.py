"""All figure generation for the JEPA control-loss experiment."""
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style='ticks', context='paper')
except ImportError:
    plt.style.use('seaborn-v0_8-paper')

MODEL_COLORS = {'jepa': '#d62728', 'ae': '#1f77b4', 'jepa_invdyn': '#2ca02c'}
MODEL_LABELS = {'jepa': 'JEPA', 'ae': 'Autoencoder', 'jepa_invdyn': 'JEPA + InvDyn'}


def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)


def _val(r, key):
    # regret-vs-information figures use the rollout regret; fall back to legacy probe regret
    # so the historical 36-run JSONs (which predate the rollout metric) still plot.
    if key == 'control_regret':
        return r.get('control_regret_rollout', r.get('control_regret'))
    return r[key]


def _agg(results, model_type, p, key):
    vals = [_val(r, key) for r in results if r['model_type'] == model_type and r['p_repeat'] == p]
    return np.mean(vals), np.std(vals)


def figure1_main_result(results, p_values, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, 'linear_probe_acc')
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', color=MODEL_COLORS[mt], label=MODEL_LABELS[mt])
        ax.fill_between(p_values, means - stds, means + stds, color=MODEL_COLORS[mt], alpha=0.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='chance')
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Linear probe accuracy')
    ax.set_title('Switch color decodability vs predictability')
    ax.legend(fontsize=8)
    _clean(ax)

    ax = axes[1]
    for mt, key, label in [('jepa', 'linear_probe_acc', 'JEPA linear'),
                            ('jepa', 'nonlinear_probe_acc', 'JEPA nonlinear'),
                            ('ae', 'linear_probe_acc', 'AE linear (ref)')]:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, key)
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', label=label)
        ax.fill_between(p_values, means - stds, means + stds, alpha=0.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Probe accuracy')
    ax.set_title('Linear vs nonlinear accessibility (JEPA)')
    ax.legend(fontsize=8)
    _clean(ax)

    fig.tight_layout()
    _save(fig, out_dir, 'figure1_main_result')


def figure2_bisimulation(results, p_values, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, 'centroid_dist_norm')
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', color=MODEL_COLORS[mt], label=MODEL_LABELS[mt])
        ax.fill_between(p_values, means - stds, means + stds, color=MODEL_COLORS[mt], alpha=0.2)
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Normalized centroid distance')
    ax.set_title('Bisimulation proxy: c=0 vs c=1 separation')
    ax.legend(fontsize=8)
    _clean(ax)

    ax = axes[1]
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, 'control_regret')
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', color=MODEL_COLORS[mt], label=MODEL_LABELS[mt])
        ax.fill_between(p_values, means - stds, means + stds, color=MODEL_COLORS[mt], alpha=0.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='max regret (random)')
    ax.axhline(0.0, color='black', linestyle=':', linewidth=1, label='zero regret')
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Control regret')
    ax.set_title('Control regret vs predictability')
    ax.legend(fontsize=8)
    _clean(ax)

    fig.tight_layout()
    _save(fig, out_dir, 'figure2_bisimulation')


def figure3_regret_vs_information(results, out_dir):
    """De-circularized: x = estimated I(Z; c) (nats, InfoNCE), y = rollout regret.

    Probe accuracy no longer appears on either axis -- we never plot regret = 1 - probe
    against probe accuracy again. Points without an `mi_infonce` key (pre-M2 JSONs) are
    skipped rather than back-filled from the probe.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        xs, ys = [], []
        for r in results:
            if r['model_type'] != mt:
                continue
            mi = r.get('mi_infonce')
            reg = _val(r, 'control_regret')
            if mi is None or reg is None:
                continue
            xs.append(mi); ys.append(reg)
        if xs:
            ax.scatter(xs, ys, color=MODEL_COLORS[mt], label=MODEL_LABELS[mt], alpha=0.8)

    ax.axhline(0.0, color='black', linestyle=':', linewidth=1, label='zero regret')
    ax.axvline(np.log(2), color='gray', linestyle='--', linewidth=1, label='I = log 2 (1 bit)')
    ax.set_xlabel('Estimated I(Z; c)  [nats, InfoNCE]')
    ax.set_ylabel('Rollout control regret')
    ax.set_title('Control regret vs retained information')
    ax.legend(fontsize=8)
    _clean(ax)
    fig.tight_layout()
    _save(fig, out_dir, 'figure3_regret_vs_information')


def figure4_ablations(results, p_values, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    model_types = ['jepa', 'ae', 'jepa_invdyn']
    means = []
    stds = []
    for mt in model_types:
        m, s = _agg(results, mt, 0.5, 'linear_probe_acc')
        means.append(m); stds.append(s)
    ax.bar(model_types, means, yerr=stds, color=[MODEL_COLORS[mt] for mt in model_types])
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xticklabels([MODEL_LABELS[mt] for mt in model_types])
    ax.set_ylabel('Linear probe accuracy at p=0.5')
    ax.set_title('Ablation: model type at hardest setting')
    _clean(ax)

    ax = axes[1]
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, 'effective_rank')
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', color=MODEL_COLORS[mt], label=MODEL_LABELS[mt])
        ax.fill_between(p_values, means - stds, means + stds, color=MODEL_COLORS[mt], alpha=0.2)
    ax.axhline(5, color='gray', linestyle='--', linewidth=1, label='collapse threshold (rank=5)')
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Effective rank')
    ax.set_title('Representation rank (capacity check)')
    ax.legend(fontsize=8)
    _clean(ax)

    fig.tight_layout()
    _save(fig, out_dir, 'figure4_ablations')


def figure5_environment(out_dir, img_size=32):
    from .env import SwitchColorEnv

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for row, p in enumerate([0.5, 1.0]):
        env = SwitchColorEnv(p_repeat=p, img_size=img_size, seed=42)
        obs = env.reset()
        for col in range(5):
            ax = axes[row, col]
            ax.imshow(np.transpose(obs, (1, 2, 0)))
            ax.set_title(f"t={env.t}  c={env.switch_color}", fontsize=9)
            ax.axis('off')
            obs, _, _ = env.step(env.switch_color)
    fig.suptitle('Environment frames: p=0.5 (top, i.i.d.) vs p=1.0 (bottom, constant)')
    fig.tight_layout()
    _save(fig, out_dir, 'figure5_environment')


def quadrant_heatmap(matrix, objectives, cells, out_dir, threshold=0.75, name='figure6_quadrant_matrix'):
    """Heatmap of retained switch-feature decodability: rows = objectives, cols = cells.

    `matrix[i, j]` is mean linear-probe accuracy of objective i on cell j (chance = 0.5).
    Each cell is annotated with the accuracy and a pass/fail mark vs `threshold` (retained
    if acc >= threshold). The colormap is centered so the threshold sits at the midpoint.
    """
    m = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 * len(cells) + 2, 0.7 * len(objectives) + 2))
    im = ax.imshow(m, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"cell {c}" for c in cells])
    ax.set_yticks(range(len(objectives)))
    ax.set_yticklabels(objectives)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            mark = 'PASS' if m[i, j] >= threshold else 'fail'
            ax.text(j, i, f"{m[i, j]:.2f}\n{mark}", ha='center', va='center', fontsize=8)
    ax.set_title(f"Retained I(Z; feature) proxy (linear probe acc)\npass = acc >= {threshold}")
    fig.colorbar(im, ax=ax, label='linear probe acc')
    fig.tight_layout()
    _save(fig, out_dir, name)


def figure7_min_reward_signal(fractions, mean_accs, out_dir, threshold=0.75,
                              min_frac=None, env_name='quadrant', name='figure7_min_reward_signal'):
    """M4: cell-4 retention (linear probe acc) vs reward_label_fraction. Marks the smallest
    fraction whose mean retention crosses the pass threshold."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fractions, mean_accs, marker='o', color='#9467bd')
    ax.axhline(threshold, color='gray', linestyle='--', linewidth=1, label=f'retain threshold {threshold}')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1, label='chance')
    if min_frac is not None:
        ax.axvline(min_frac, color='green', linestyle='-.', linewidth=1,
                   label=f'min fraction {min_frac:g}')
    ax.set_xscale('log')
    ax.set_xlabel('reward_label_fraction')
    ax.set_ylabel('cell-4 linear probe acc')
    ax.set_title(f'Minimum reward signal for cell-4 retention\n(jepa_reward, {env_name})')
    ax.legend(fontsize=8)
    _clean(ax)
    fig.tight_layout()
    _save(fig, out_dir, name)


def figure8_capacity(series, latent_dims, out_dir, threshold=0.75, name='figure8_capacity'):
    """M4: cell-4 retention vs latent_dim. `series` maps a label (env/objective) -> list of
    mean accs aligned with latent_dims. Flat curves => latent capacity is not the bottleneck."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, accs in series.items():
        ax.plot(latent_dims, accs, marker='o', label=label)
    ax.axhline(threshold, color='gray', linestyle='--', linewidth=1, label=f'retain {threshold}')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1, label='chance')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('latent_dim')
    ax.set_ylabel('cell-4 linear probe acc')
    ax.set_title('Capacity is not the bottleneck (cell 4, p=0.5)')
    ax.legend(fontsize=8)
    _clean(ax)
    fig.tight_layout()
    _save(fig, out_dir, name)


def generate_all_figures(results, p_values, out_dir):
    figure1_main_result(results, p_values, out_dir)
    figure2_bisimulation(results, p_values, out_dir)
    figure3_regret_vs_information(results, out_dir)
    figure4_ablations(results, p_values, out_dir)
    figure5_environment(out_dir)
    print(f"Figures saved to {out_dir}")
