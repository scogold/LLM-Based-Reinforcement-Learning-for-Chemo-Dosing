"""Shared helpers: config loading, seeding, building the env, and starting a run directory.

Reads these config sections (configs/base.yaml, overridden by the per-model file):

    env:
      id: AhnChemoEnv-continuous-setting2   # DTRGym env ID; setting1 is noise-free, 2-5 add noise.
                                            # Plain "AhnChemoEnv-continuous" is NOT noise-free.
      max_t: 600                            # steps per episode; one step is 6 hours, so 150 days
    doses: [0.0, 0.25, 0.5, 0.75, 1.0]      # used by agent.py; continuous AhnChemoEnv takes doses in [0, 1]

AhnChemoEnv facts this module relies on (DTRGym 0.1.0, DTRGym/ahn_chemo_env.py):
- Gymnasium API: reset -> (obs, info), step -> (obs, reward, terminated, truncated, info).
- obs = [tumour cells, immune cells, drug concentration], noisy in settings 2-5.
- terminated = patient died (normal cells < 70% of initial), reward -100;
  truncated = reached max_t. ppo.py bootstraps on truncation, not on termination.
- reset(seed=None) reseeds NumPy's *global* RNG from entropy; see SeededResets below.

Used by:
- scripts/train.py: load_config, set_seed, make_env, start_run.
- scripts/evaluate.py and evaluation.py: load_config, make_env, then env.reset(seed=s) for each
  fixed evaluation seed, so every agent sees the same patients.
- ppo.py: expects the env from make_env (RecordEpisodeStatistics provides info["episode"]["r"]).
"""

import random
from datetime import datetime
from pathlib import Path

import DTRGym  # noqa: F401  (importing registers the AhnChemoEnv IDs with gymnasium)
import gymnasium as gym
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


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


def make_env(cfg, seed):
    env = gym.make(cfg["env"]["id"], max_t=cfg["env"]["max_t"])
    if isinstance(env.action_space, gym.spaces.Box):
        env = ScalarDose(env)
    env = SeededResets(env, seed)
    return gym.wrappers.RecordEpisodeStatistics(env)


def start_run(cfg, seed):
    """Create runs/<config>-seed<seed>-<time>/, save the resolved config and seed there,
    and return (run_dir, TensorBoard writer)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"{cfg['config_name']}-seed{seed}-{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": seed}, sort_keys=False))
    return run_dir, SummaryWriter(run_dir)
