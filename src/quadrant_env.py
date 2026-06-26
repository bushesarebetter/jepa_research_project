"""QuadrantEnv: a single tagged feature placed in one cell of the 2x2 spine.

The 2x2 (CLAUDE.md):
  Axis A controllable (action moves it) vs uncontrollable (exogenous).
  Axis B relevant (sets reward / optimal action) vs irrelevant.

  cell 1 controllable + relevant     (everyone keeps it)
  cell 2 controllable + irrelevant   (controllability methods waste capacity)
  cell 3 uncontrollable + irrelevant (the classic distractor; recon keeps it)
  cell 4 uncontrollable + relevant   (THE HARD CASE)

Frame: 12x12, a predictable sinusoidal background (always control-irrelevant) plus one
3x3 tagged-feature patch top-left whose binary value is the thing we measure retention of.

Dynamics:
  controllable feature:   value_{t+1} = value_t XOR action_t   (the action toggles it)
  uncontrollable feature: repeats with prob `predictability`, else flips (the M1 p_repeat knob),
                          independent of the action.

Reward/rule: a*(t) = value_t if the feature is RELEVANT else 0 (a fixed constant). reward =
1 if action == a*(t) else 0. So for relevant cells reward depends on the feature; for
irrelevant cells the feature is reward-decoupled (a pure distractor) yet still rendered.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureSpec:
    name: str
    controllable: bool      # does action_t move it?
    relevant: bool          # does it determine reward / a*?
    predictability: float = 0.5  # p_repeat for exogenous features; ignored if controllable


# The 2x2 templates; make_cell() stamps a predictability onto the chosen one.
_CELLS = {
    1: FeatureSpec('ctrl_relevant',   controllable=True,  relevant=True),
    2: FeatureSpec('ctrl_irrelevant', controllable=True,  relevant=False),
    3: FeatureSpec('exo_irrelevant',  controllable=False, relevant=False),
    4: FeatureSpec('exo_relevant',    controllable=False, relevant=True),
}


def make_cell(cell, predictability):
    """Return the single-feature spec for cell in 1..4 at the given predictability."""
    s = _CELLS[cell]
    return FeatureSpec(s.name, s.controllable, s.relevant, predictability)


class QuadrantEnv:
    def __init__(self, spec, img_size=12, seed=None):
        self.spec = spec
        self.img_size = img_size
        self.n_actions = 2
        self.obs_shape = (3, img_size, img_size)
        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.value = None

    @property
    def feature_name(self):
        return self.spec.name

    def reset(self):
        self.t = 0
        self.value = int(self.rng.randint(0, 2))
        return self._render()

    def a_star(self):
        return self.value if self.spec.relevant else 0

    def step(self, action):
        v = self.value
        reward = 1.0 if action == self.a_star() else 0.0
        self.t += 1
        if self.spec.controllable:
            self.value = v ^ int(action)                       # action toggles
        elif self.rng.random_sample() < self.spec.predictability:
            self.value = v                                     # exogenous: repeat
        else:
            self.value = 1 - v                                 # exogenous: flip
        return self._render(), reward, v                       # v == bits_t (value in obs_t)

    def _render(self):
        s = self.img_size
        img = np.zeros((3, s, s), dtype=np.float32)
        xx, yy = np.meshgrid(np.linspace(0, 2 * np.pi, s), np.linspace(0, 2 * np.pi, s))
        phase = self.t * 0.15
        img[0] = 0.5 + 0.25 * np.sin(xx + phase) * np.cos(yy * 0.5 + phase * 0.3)
        img[1] = 0.5 + 0.25 * np.cos(xx * 2 + phase * 1.3) * np.sin(yy + phase * 0.7)
        img[2] = 0.5 + 0.25 * np.sin(xx * 0.5 + yy + phase * 0.5)
        b = s // 4
        if self.value == 1:
            img[0, :b, :b], img[1, :b, :b], img[2, :b, :b] = 0.95, 0.05, 0.05
        else:
            img[0, :b, :b], img[1, :b, :b], img[2, :b, :b] = 0.05, 0.05, 0.95
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def sample_transitions(self, n, action_policy='random', informative_eps=0.2):
        """Roll one length-n stream and return transition tensors.

        action_policy='random'      -> action ~ uniform (uninformative about exogenous features).
        action_policy='informative' -> action = a* with prob 1-informative_eps else uniform; the
            taken action then ENCODES the feature, so inverse-dynamics can recover it. The eps
            keeps controllable features from collapsing under a deterministic policy. The point
            is that the rescue tracks action informativeness -- not that we supervise the label.
        """
        obs_t, obs_tp1, actions, a_stars, rewards, bits = [], [], [], [], [], []
        o = self.reset()
        for _ in range(n):
            astar = self.a_star()
            if action_policy == 'informative' and self.rng.random_sample() > informative_eps:
                a = astar
            else:
                a = int(self.rng.randint(0, self.n_actions))
            o_next, r, vt = self.step(a)
            obs_t.append(o); obs_tp1.append(o_next)
            actions.append(a); a_stars.append(astar); rewards.append(r); bits.append(vt)
            o = o_next
        return {
            'obs': np.stack(obs_t).astype(np.float32),
            'next_obs': np.stack(obs_tp1).astype(np.float32),
            'action': np.array(actions, dtype=np.int64),
            'a_star': np.array(a_stars, dtype=np.int64),
            'reward': np.array(rewards, dtype=np.float32),
            'labels': {self.spec.name: np.array(bits, dtype=np.int64)},
        }


if __name__ == "__main__":  # runnable check: each cell's defining property actually holds
    N = 4000
    # cell 1 controllable: feature must follow XOR of actions (action moves it).
    d = QuadrantEnv(make_cell(1, 0.5), seed=0).sample_transitions(N, 'random')
    bits, act = d['labels']['ctrl_relevant'], d['action']
    pred_next = bits ^ act                       # value_{t+1} predicted from (value_t, action_t)
    actual_next = np.concatenate([bits[1:], bits[-1:]])
    assert (pred_next[:-1] == actual_next[:-1]).all(), "cell1 not controllable by action"
    # relevant cells: reward is determined by (action == feature); a* == feature.
    assert (d['a_star'] == bits).all(), "cell1 a* should equal the relevant feature"
    assert (d['reward'] == (act == bits)).all(), "reward != 1[action == a*]"
    # cell 3 uncontrollable: feature independent of action -> XOR-predict should be ~chance.
    d3 = QuadrantEnv(make_cell(3, 0.5), seed=0).sample_transitions(N, 'random')
    b3, a3 = d3['labels']['exo_irrelevant'], d3['action']
    nxt3 = np.concatenate([b3[1:], b3[-1:]])
    acc_ctrl = ((b3 ^ a3)[:-1] == nxt3[:-1]).mean()
    assert 0.4 < acc_ctrl < 0.6, f"cell3 should NOT be action-controllable, got {acc_ctrl:.2f}"
    # irrelevant cells: a* is the constant 0 (feature does not set reward).
    assert (d3['a_star'] == 0).all(), "cell3 a* should be constant (feature irrelevant)"
    # predictability knob: p=1.0 means the exogenous feature never changes within a stream.
    dp = QuadrantEnv(make_cell(4, 1.0), seed=1).sample_transitions(200, 'random')
    assert len(set(dp['labels']['exo_relevant'].tolist())) == 1, "p=1.0 should freeze the feature"
    print("quadrant_env OK: controllability, relevance, and predictability knob all hold")
