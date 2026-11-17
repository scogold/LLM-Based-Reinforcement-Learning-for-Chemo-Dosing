"""Read the true simulator state from AhnChemoEnv for use in evaluation metrics.

Purpose: the agent only sees a (possibly noisy or masked) observation of tumor, immune
and drug levels, and never sees normal cells. Metrics such as normal-cell damage and
death must be computed from the real state, so this module extracts it from the env.

`ahn_true_state` is passed to evaluation.run_episode as `true_state_fn`.
"""
from __future__ import annotations

# AhnChemoEnv keeps its ODE state in env.unwrapped.cur_state (also reported in info["state"]).
# Keys: N = normal cells, T = tumor cells, I = immune cells, B = drug concentration.
STATE_KEYS = {"normal": "N", "tumor": "T", "immune": "I"}


def _scalar(v) -> float:
    """float from a plain number, a 0-d array, or a shape-(1,) array."""
    try:
        return float(list(v)[0])
    except TypeError:
        return float(v)


def _obs_drug(obs) -> float:
    seq = list(obs)
    row = seq[-1] if seq and hasattr(seq[0], "__len__") else seq
    return float(row[2])


def ahn_true_state(env, obs, info) -> dict:
    """Return {'tumor', 'immune', 'drug', 'normal'} as floats from the simulator state."""
    state = getattr(getattr(env, "unwrapped", env), "cur_state", None)
    if state is None and isinstance(info, dict):
        state = info.get("state")
    if state is None or not hasattr(state, "keys"):
        raise RuntimeError(
            "could not find the AhnChemoEnv state (env.unwrapped.cur_state / info['state']); "
            "run with --use-obs-state to fall back to the observation")
    out = {name: _scalar(state[key]) for name, key in STATE_KEYS.items()}
    rest = [k for k in state.keys() if k not in STATE_KEYS.values()]
    out["drug"] = _scalar(state[rest[0]]) if len(rest) == 1 else _obs_drug(obs)
    return out
