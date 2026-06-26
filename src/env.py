"""SwitchColorEnv: a minimal control environment for testing JEPA's control-relevance blind spot.

The switch block (top-left 8x8) is control-relevant (determines optimal action /
reward) but its temporal predictability is tunable via p_repeat. The background
is a deterministic, predictable sinusoidal drift that is control-irrelevant.
This separates "control-relevant" from "temporally predictable" along two axes,
which JEPA's self-prediction objective conflates.
"""
import numpy as np


def roll_transitions(env, n, action_policy='random', informative_eps=0.2):
    """Roll one length-n stream off any cell-style env and return standard transition tensors.

    The env must expose: reset() -> obs, a_star() -> int, step(a) -> (obs, reward, bits_t),
    n_actions, feature_name, and an .rng. Shared by SwitchColorEnv, GridWorldHiddenRuleEnv,
    and (logically) QuadrantEnv so every --env emits the same schema.

    action_policy='random'      -> uniform action (uninformative about exogenous features).
    action_policy='informative' -> action = a* with prob 1-eps else uniform; the taken action
        then ENCODES the feature so inverse-dynamics can recover it.
    """
    obs_t, obs_tp1, actions, a_stars, rewards, bits = [], [], [], [], [], []
    o = env.reset()
    for _ in range(n):
        astar = env.a_star()
        if action_policy == 'informative' and env.rng.random_sample() > informative_eps:
            a = astar
        else:
            a = int(env.rng.randint(0, env.n_actions))
        o_next, r, vt = env.step(a)
        obs_t.append(o); obs_tp1.append(o_next)
        actions.append(a); a_stars.append(astar); rewards.append(r); bits.append(vt)
        o = o_next
    return {
        'obs': np.stack(obs_t).astype(np.float32),
        'next_obs': np.stack(obs_tp1).astype(np.float32),
        'action': np.array(actions, dtype=np.int64),
        'a_star': np.array(a_stars, dtype=np.int64),
        'reward': np.array(rewards, dtype=np.float32),
        'labels': {env.feature_name: np.array(bits, dtype=np.int64)},
    }


class SwitchColorEnv:
    feature_name = 'switch_color'
    n_actions = 2
    def __init__(self, p_repeat=0.5, img_size=32, seed=None, switch_contrast=1.0):
        self.p_repeat = p_repeat
        self.img_size = img_size
        # switch_contrast in (0, 1]: salience of the control-relevant bit. 1.0 == legacy
        # (block colors 0.95/0.05); lower contrast pulls the block toward background gray
        # (0.5), making the relevant feature less visually salient but still encodable.
        self.switch_contrast = switch_contrast
        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.switch_color = None

    def reset(self):
        self.t = 0
        self.switch_color = int(self.rng.randint(0, 2))
        return self._render()

    def a_star(self):
        return self.switch_color  # switch_color is uncontrollable+relevant == cell 4

    def sample_transitions(self, n, action_policy='random', informative_eps=0.2):
        return roll_transitions(self, n, action_policy, informative_eps)

    def step(self, action):
        reward = 1.0 if action == self.switch_color else 0.0
        prev_color = self.switch_color
        self.t += 1
        if self.rng.random_sample() < self.p_repeat:
            self.switch_color = prev_color
        else:
            self.switch_color = 1 - prev_color
        obs = self._render()
        return obs, reward, prev_color

    def _render(self):
        s = self.img_size
        img = np.zeros((3, s, s), dtype=np.float32)
        xx, yy = np.meshgrid(np.linspace(0, 2 * np.pi, s), np.linspace(0, 2 * np.pi, s))
        phase = self.t * 0.15
        img[0] = 0.5 + 0.25 * np.sin(xx + phase) * np.cos(yy * 0.5 + phase * 0.3)
        img[1] = 0.5 + 0.25 * np.cos(xx * 2 + phase * 1.3) * np.sin(yy + phase * 0.7)
        img[2] = 0.5 + 0.25 * np.sin(xx * 0.5 + yy + phase * 0.5)

        block = s // 4
        hi = 0.5 + 0.45 * self.switch_contrast  # 0.95 at contrast 1.0
        lo = 0.5 - 0.45 * self.switch_contrast  # 0.05 at contrast 1.0
        if self.switch_color == 1:
            img[0, :block, :block] = hi
            img[1, :block, :block] = lo
            img[2, :block, :block] = lo
        else:
            img[0, :block, :block] = lo
            img[1, :block, :block] = lo
            img[2, :block, :block] = hi
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def collect_dataset(self, n_episodes, episode_length):
        obs_list, next_obs_list, color_list, action_list, reward_list = [], [], [], [], []
        for _ in range(n_episodes):
            obs = self.reset()
            for _ in range(episode_length):
                action = self.switch_color  # oracle action == switch_color
                next_obs, reward, prev_color = self.step(action)
                obs_list.append(obs)
                next_obs_list.append(next_obs)
                color_list.append(prev_color)
                action_list.append(action)
                reward_list.append(reward)
                obs = next_obs
        data = {
            'obs': np.stack(obs_list).astype(np.float32),
            'next_obs': np.stack(next_obs_list).astype(np.float32),
            'switch_color': np.array(color_list, dtype=np.int64),
            'action': np.array(action_list, dtype=np.int64),
            'reward': np.array(reward_list, dtype=np.float32),
        }
        mean = data['switch_color'].mean()
        #assert 0.4 < mean < 0.6, f"Switch color not balanced: mean={mean}"
        return data


if __name__ == "__main__":  # runnable check: contrast=1.0 reproduces legacy pixels; lower = less salient
    e = SwitchColorEnv(p_repeat=1.0, seed=0); e.reset()
    b = e.img_size // 4
    blk = e._render()[:, :b, :b]
    assert abs(blk.max() - 0.95) < 1e-6 and abs(blk.min() - 0.05) < 1e-6, blk.max()
    lo = SwitchColorEnv(p_repeat=1.0, seed=0, switch_contrast=0.2); lo.reset()
    assert lo._render()[:, :b, :b].max() < blk.max(), "lower contrast should reduce salience"
    print("env OK: legacy contrast reproduced; salience knob monotone")
