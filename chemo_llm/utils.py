"""Shared helpers: config loading, seeding, building the env, and starting a run directory.

Reads these config sections (configs/base.yaml, overridden by the per-model file):

    env:
      id: AhnChemoEnv-continuous-setting2   # DTRGym env ID; setting1 is noise-free, 2-5 vary the
                                            # drug response per patient, 3-5 also add state and
                                            # observation noise.
                                            # Plain "AhnChemoEnv-continuous" is NOT noise-free.
      max_t: 600                            # steps per episode; one step is 6 hours, so 150 days
      history: 6                            # earlier steps shown to the policy (ObservationHistory)
      death_penalty: 300                    # training reward only (DeathPenalty)
    doses: [0.0, 0.25, 0.5, 0.75, 1.0]      # used by agent.py; continuous AhnChemoEnv takes doses in [0, 1]

AhnChemoEnv facts this module relies on (DTRGym 0.1.0, DTRGym/ahn_chemo_env.py):
- Gymnasium API: reset -> (obs, info), step -> (obs, reward, terminated, truncated, info).
- obs = [tumour cells, immune cells, drug concentration], noisy in settings 3-5. make_env
  stacks the last history + 1 of them into a 2-D array (rows oldest first).
- terminated = patient died (normal cells < 70% of initial), reward -100;
  truncated = reached max_t. ppo.py bootstraps on truncation, not on termination.
- reset(seed=None) reseeds NumPy's *global* RNG from entropy; see SeededResets below.

Used by:
- scripts/train.py: load_config, set_seed, make_env(..., training=True), start_run.
- scripts/evaluate.py and evaluation.py: load_config, make_env, then env.reset(seed=s) for each
  fixed evaluation seed, so every agent sees the same patients.
- ppo.py: expects the env from make_env (RecordEpisodeStatistics provides info["episode"]["r"]).
"""

import random
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

import DTRGym  # noqa: F401  (importing registers the AhnChemoEnv IDs with gymnasium)
import gymnasium as gym
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from chemo_llm import prompts


def load_config(path):
    """Load configs/base.yaml, then override it with the model config at `path`."""
    path = Path(path)
    cfg = yaml.safe_load((path.parent / "base.yaml").read_text())
    _merge(cfg, yaml.safe_load(path.read_text()))
    cfg["config_name"] = path.stem
    return cfg


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


class SeededResets(gym.Wrapper):
    """AhnChemoEnv reseeds NumPy's global RNG from entropy when reset() gets no seed, which
    makes runs unrepeatable. Seeds for unseeded resets are drawn from a generator seeded once."""

    def __init__(self, env, seed):
        super().__init__(env)
        self.seeds = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = int(self.seeds.integers(2**31))
        return self.env.reset(seed=seed, options=options)


class ScalarDose(gym.ActionWrapper):
    """The continuous env expects a shape-(1,) array; the agent gives one float dose."""

    def action(self, dose):
        return np.array([dose], dtype=np.float32)


class ObservationHistory(gym.Wrapper):
    """Return the last `length` observations as rows, oldest first, instead of only the current
    one. Every patient starts in the same state, and hard and easy patients differ only in how
    they respond to the drug, so the policy needs to see the trend to tell them apart.
    prompts.py turns the rows into trends and a short table of earlier steps."""

    def __init__(self, env, length):
        super().__init__(env)
        self.rows = deque(maxlen=length)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rows.clear()
        self.rows.append(obs)
        return np.array(self.rows), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rows.append(obs)
        return np.array(self.rows), reward, terminated, truncated, info


class DeathPenalty(gym.Wrapper):
    """Training-only reward fix. AhnChemoEnv charges -100 for death, but a patient kept alive
    with a lingering tumour can lose several hundred points over 150 days, so the raw reward
    can prefer letting a hard patient die early. On death this subtracts a further `penalty`.
    The penalty is constant (not based on the day) because the policy can't see the day.
    Evaluation uses the env's own reward, so reported returns stay comparable with DTR-Bench."""

    def __init__(self, env, penalty):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward -= self.penalty
        return obs, reward, terminated, truncated, info


def make_env(cfg, seed, training=False):
    """training=True adds the DeathPenalty reward, unless env.death_penalty is 0 (train on the
    env's own reward). Evaluation always leaves the env's reward as it is."""
    env_cfg = cfg["env"]
    assert env_cfg["history"] <= prompts.MAX_HISTORY, "the prompt can't show more earlier steps than MAX_HISTORY"
    env = gym.make(env_cfg["id"], max_t=env_cfg["max_t"])
    if isinstance(env.action_space, gym.spaces.Box):
        env = ScalarDose(env)
    env = SeededResets(env, seed)
    env = ObservationHistory(env, env_cfg["history"] + 1)  # earlier steps + the current one
    if training and env_cfg["death_penalty"]:
        env = DeathPenalty(env, env_cfg["death_penalty"])
    return gym.wrappers.RecordEpisodeStatistics(env)


def git_commit():
    """The repo's current commit hash, ending in "-dirty" if there are uncommitted changes,
    so each run records exactly which code produced it. "unknown" if git isn't available."""
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                                 text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if changes else "")


def start_run(cfg, seed):
    """Create runs/<config>-seed<seed>-<time>/, save the resolved config, seed and git commit
    there, and return (run_dir, TensorBoard writer)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"{cfg['config_name']}-seed{seed}-{stamp}"
    run_dir.mkdir(parents=True)
    record = {**cfg, "seed": seed, "git_commit": git_commit()}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    return run_dir, SummaryWriter(run_dir)
