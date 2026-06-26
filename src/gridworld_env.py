"""GridWorldHiddenRuleEnv: cell-4 (uncontrollable + relevant) logic in a gridworld skin.

A second surface form to check the JEPA control-loss replicates beyond SwitchColorEnv /
QuadrantEnv. The reward-relevant feature is a HIDDEN RULE BIT shown as the colour of a fixed
goal cell (red=1 / blue=0). It evolves EXOGENOUSLY: repeats with prob `predictability`, else
flips -- identical dynamics to the switch/quadrant exogenous feature, so it is cell 4. An agent
token walks a deterministic (predictable, reward-irrelevant) path as a visual distractor.

Same interface as SwitchColorEnv/QuadrantEnv: reset, a_star, step -> (obs, reward, bits_t),
sample_transitions -> standard transition dict via env.roll_transitions.
"""
import numpy as np

from .env import roll_transitions


class GridWorldHiddenRuleEnv:
    feature_name = 'rule_bit'
    n_actions = 2

    def __init__(self, predictability=0.5, img_size=12, seed=None, grid=4):
        self.predictability = predictability
        self.img_size = img_size
        self.grid = grid
        self.obs_shape = (3, img_size, img_size)
        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.value = None

    def reset(self):
        self.t = 0
        self.value = int(self.rng.randint(0, 2))
        return self._render()

    def a_star(self):
        return self.value  # rule bit is relevant: optimal action == bit

    def sample_transitions(self, n, action_policy='random', informative_eps=0.2):
        return roll_transitions(self, n, action_policy, informative_eps)

    def step(self, action):
        v = self.value
        reward = 1.0 if action == v else 0.0
        self.t += 1
        if self.rng.random_sample() < self.predictability:
            self.value = v              # exogenous: repeat
        else:
            self.value = 1 - v          # exogenous: flip
        return self._render(), reward, v

    def _fill_cell(self, img, i, j, cell, rgb):
        y0, x0 = i * cell + 1, j * cell + 1
        y1, x1 = (i + 1) * cell, (j + 1) * cell
        for c in range(3):
            img[c, y0:y1, x0:x1] = rgb[c]

    def _render(self):
        s, g = self.img_size, self.grid
        cell = s // g
        img = np.full((3, s, s), 0.45, dtype=np.float32)
        for k in range(g + 1):                          # grid lines
            p = min(k * cell, s - 1)
            img[:, p, :] = 0.30
            img[:, :, p] = 0.30
        # agent: deterministic walk depends only on t -> predictable + reward-irrelevant.
        self._fill_cell(img, self.t % g, (self.t * 2) % g, cell, (0.15, 0.15, 0.15))
        # goal cell drawn LAST (always visible), coloured by the hidden rule bit.
        color = (0.95, 0.05, 0.05) if self.value == 1 else (0.05, 0.05, 0.95)
        self._fill_cell(img, g - 1, g - 1, cell, color)
        return np.clip(img, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":  # runnable check: it really is cell 4 (exogenous + relevant)
    N = 4000
    d = GridWorldHiddenRuleEnv(0.5, seed=0).sample_transitions(N, 'random')
    bits, act = d['labels']['rule_bit'], d['action']
    # relevant: a* == bit and reward == 1[action == a*]
    assert (d['a_star'] == bits).all(), "rule bit should be the relevant feature (a* == bit)"
    assert (d['reward'] == (act == bits)).all(), "reward != 1[action == a*]"
    # uncontrollable: XOR-predicting next from (bit, action) should be ~chance.
    nxt = np.concatenate([bits[1:], bits[-1:]])
    acc_ctrl = ((bits ^ act)[:-1] == nxt[:-1]).mean()
    assert 0.4 < acc_ctrl < 0.6, f"rule bit should NOT be action-controllable, got {acc_ctrl:.2f}"
    # predictability knob: p=1.0 freezes the bit within a stream.
    dp = GridWorldHiddenRuleEnv(1.0, seed=1).sample_transitions(200, 'random')
    assert len(set(dp['labels']['rule_bit'].tolist())) == 1, "p=1.0 should freeze the rule bit"
    print("gridworld_env OK: exogenous + relevant (cell 4), predictability knob holds")
