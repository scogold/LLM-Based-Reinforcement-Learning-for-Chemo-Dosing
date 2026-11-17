"""Evaluation metrics for AhnChemoEnv episodes (pure functions, stdlib only).

Episode log format: a list of per-step dicts. Keys (extra keys are ignored):
  reward      float   env reward at this step                      (required)
  tumor       float   tumor cells AFTER the step                   (required)
  immune      float   immune cells after the step                  (required)
  drug        float   drug concentration after the step            (required)
  dose        float   dose actually sent to the env, in [0, 1]     (required)
  normal      float   normal cells after the step (optional)
  terminated  bool    True on the step where the patient died (env `terminated`)  (optional)
  dose_index  int     index of the chosen dose level (optional; used for histograms)
  log_prob    float   log-prob of the chosen action from Agent.act (optional)
  parse_ok    bool    only if some agent can fail to produce a valid action (optional;
                      the logit-readout Agent cannot, so you can leave it out)

IMPORTANT: for tumor/immune/drug/normal, log the TRUE simulator state, not the
noisy or masked observation, so metrics stay comparable across noise settings.
The agent's observation is a different thing. Check whether AhnChemoEnv exposes
the true state (e.g. in `info` or on `env.unwrapped`). Normal cells are hidden
from the agent but are still part of the simulator state, so use them here.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

HIGH_DOSE = 0.75        # dose >= this counts as "aggressive"
CLEARED_RATIO = 0.05    # tumor <= 5% of its starting size counts as cleared (tune)
DOSE_LEVELS_DEFAULT = (0.0, 0.25, 0.5, 0.75, 1.0)


def episode_metrics(steps: Sequence[dict], tumor0: Optional[float] = None,
                    normal0: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Summarise one episode. tumor0/normal0 = starting values if known
    (otherwise the first logged step is used, which is slightly off)."""
    if not steps:
        raise ValueError("empty episode")
    n = len(steps)
    tumor = [s["tumor"] for s in steps]
    immune = [s["immune"] for s in steps]
    doses = [s["dose"] for s in steps]
    t0 = tumor0 if tumor0 is not None else tumor[0]

    out: Dict[str, Optional[float]] = {
        "n_steps": float(n),
        "return": float(sum(s["reward"] for s in steps)),
        "tumor_final_ratio": tumor[-1] / t0 if t0 else None,
        "tumor_min_ratio": min(tumor) / t0 if t0 else None,
        "cleared": float(tumor[-1] <= CLEARED_RATIO * t0) if t0 else None,
        "immune_min": min(immune),
        "immune_final": immune[-1],
        "total_drug": float(sum(doses)),
        "mean_dose": float(sum(doses) / n),
        "aggressive_frac": sum(d >= HIGH_DOSE for d in doses) / n,
        "max_high_dose_streak": float(_max_streak(d >= HIGH_DOSE for d in doses)),
        "drug_conc_max": max(s["drug"] for s in steps),
    }

    if all("normal" in s for s in steps):
        normal = [s["normal"] for s in steps]
        n0 = normal0 if normal0 is not None else normal[0]
        out["normal_min"] = min(normal)
        out["normal_final"] = normal[-1]
        out["normal_min_ratio"] = min(normal) / n0 if n0 else None
    else:
        out["normal_min"] = out["normal_final"] = out["normal_min_ratio"] = None

    if any("terminated" in s for s in steps):
        # AhnChemoEnv ends the episode with reward -100 when normal cells fall below
        # 70% of their starting level, so 'died' is the key safety outcome.
        out["died"] = float(any(s.get("terminated") for s in steps))
    else:
        out["died"] = None

    if all("log_prob" in s for s in steps):
        # average probability the policy gave to the action it took (confidence)
        out["mean_action_prob"] = sum(math.exp(s["log_prob"]) for s in steps) / n
    else:
        out["mean_action_prob"] = None

    if all("parse_ok" in s for s in steps):
        out["parse_fail_rate"] = sum(not s["parse_ok"] for s in steps) / n
    else:
        out["parse_fail_rate"] = None
    return out


def _max_streak(flags: Iterable[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def aggregate(per_episode: Sequence[Dict[str, Optional[float]]]) -> Dict[str, Dict[str, float]]:
    """Mean, std, and 95% CI for each metric across episodes (None values skipped).

    Aggregate across SEEDS as well as episodes when you report: pass one entry
    per episode and also compute this per seed to see seed-to-seed spread.
    """
    keys = sorted({k for m in per_episode for k in m})
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [m[k] for m in per_episode if m.get(k) is not None and not math.isnan(m[k])]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ci = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        summary[k] = {"mean": mean, "std": std, "ci95": ci, "n": float(len(vals))}
    return summary


def consistency(levels: Sequence[int]) -> float:
    """Fraction of repeated queries (same state, same prompt) that agree with
    the most common answer. 1.0 = perfectly consistent."""
    if not levels:
        raise ValueError("no levels")
    top = max(set(levels), key=levels.count)
    return levels.count(top) / len(levels)


def dose_histogram(levels: Sequence[int], n_levels: int = len(DOSE_LEVELS_DEFAULT)) -> List[float]:
    """Share of steps spent at each dose level (for a bar chart)."""
    if not levels:
        return [0.0] * n_levels
    return [levels.count(i) / len(levels) for i in range(n_levels)]


def compare(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Simple comparison of two methods' per-episode values (e.g. returns) on the
    SAME patients/seeds in the same order. Returns mean paired difference and an
    approximate 95% CI. Good enough for a class report; say so."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equal-length sequences with >= 2 items")
    diffs = [x - y for x, y in zip(a, b)]
    mean = statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return {"mean_diff": mean, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se}


# ---------------------------------------------------------------------------
# Running episodes and logging (glue between the env, an agent, and the metrics)
# ---------------------------------------------------------------------------
OBS_NAMES = ("tumor", "immune", "drug")  # must match prompts.OBS_NAMES / the env


def _last_row(obs) -> List[float]:
    """Latest [tumor, immune, drug] from an obs that is 1-D or a stacked history."""
    seq = list(obs)
    row = seq[-1] if seq and hasattr(seq[0], "__len__") else seq
    return [float(x) for x in row]


def _state_from_obs(obs, last: Optional[Dict[str, float]]) -> Dict[str, float]:
    """FALLBACK when no true-state function is given: read the observation.
    NaN (missing value) is replaced by the previous value. In noisy settings (3-5)
    this is the NOISY observation, so prefer passing true_state_fn."""
    out = {}
    for name, v in zip(OBS_NAMES, _last_row(obs)):
        if math.isnan(v):
            v = last[name] if last else 0.0
        out[name] = v
    return out


def run_episode(env, act_fn: Callable, seed: int,
                true_state_fn: Optional[Callable] = None,
                max_steps: Optional[int] = None) -> Tuple[List[dict], Dict[str, float]]:
    """Play one episode and return (step_log, start_state).

    act_fn(obs) -> (env_action, action_index, log_prob_or_None, value), the same shape
    Agent.act returns, so you can pass `agent.act` or `lambda o: agent.act(o, greedy=True)`.

    true_state_fn(env, obs, info) -> dict with tumor/immune/drug (and optionally normal):
    the TRUE simulator state. Needed for clean metrics in noisy settings. If omitted,
    the observation is used instead.
    """
    obs, info = env.reset(seed=seed)
    get = true_state_fn or (lambda e, o, i: _state_from_obs(o, None))
    start = dict(get(env, obs, info))
    last = dict(start)
    steps: List[dict] = []
    terminated = truncated = False
    while not (terminated or truncated):
        env_action, index, log_prob, _value = act_fn(obs)
        obs, reward, terminated, truncated, info = env.step(env_action)
        state = true_state_fn(env, obs, info) if true_state_fn else _state_from_obs(obs, last)
        last = dict(state)
        step = {"reward": float(reward), "dose": float(env_action),
                "dose_index": int(index), "terminated": bool(terminated), **state}
        if log_prob is not None:
            step["log_prob"] = float(log_prob)
        steps.append(step)
        if max_steps is not None and len(steps) >= max_steps:
            break
    return steps, start


def evaluate_policy(env, act_fn: Callable, seeds: Sequence[int],
                    true_state_fn: Optional[Callable] = None,
                    max_steps: Optional[int] = None):
    """Run one episode per seed. Use the SAME seeds for every agent so they all see the
    same patients. Returns (per_episode_metrics, per_episode_logs)."""
    metrics, logs = [], []
    for seed in seeds:
        steps, start = run_episode(env, act_fn, seed, true_state_fn, max_steps)
        metrics.append(episode_metrics(steps, tumor0=start.get("tumor"),
                                       normal0=start.get("normal")))
        logs.append({"seed": seed, "start": start, "steps": steps})
    return metrics, logs


def baseline_policy(name: str, doses: Sequence[float], seed: int = 0) -> Callable:
    """Non-LLM reference policies with the same act signature as Agent.act.
    name: 'zero' (never dose), 'max' (always max), 'half' (middle level), 'random'."""
    rng = random.Random(seed)
    n = len(doses)
    pick = {"zero": lambda: 0, "max": lambda: n - 1, "half": lambda: n // 2,
            "random": lambda: rng.randrange(n)}
    if name not in pick:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(pick)}")

    def act(obs, greedy=False):
        i = pick[name]()
        return doses[i], i, None, 0.0
    return act


def save_episode_logs(path, logs) -> None:
    """Write the raw per-step logs as JSON so plots can be remade without re-running."""
    with open(path, "w") as f:
        json.dump(logs, f)