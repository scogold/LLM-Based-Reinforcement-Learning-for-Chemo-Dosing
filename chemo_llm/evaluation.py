"""Evaluation for the chemo-dosing agents: run episodes, log them, and compute metrics.

Everything is plain Python (standard library only) and works with any policy that has the
same call shape as Agent.act, so the LLM agent, the PPO-trained agent, the zero-shot agent
(adapter disabled) and the simple baselines are all scored the same way.

Running episodes:
    run_episode(env, act_fn, seed, true_state_fn)   play one episode, return the step log.
    evaluate_policy(env, act_fn, seeds, ...)         one episode per seed; returns per-episode
                                                     metrics and the raw logs. Use the same
                                                     seeds for every agent so they all treat
                                                     the same patients.
    baseline_policy(name, doses, seed)               non-LLM reference policies: "zero"
                                                     (never dose), "half", "max", "random".
    save_episode_logs(path, logs)                    write raw logs as JSON for plotting.
    act_fn(obs) returns (env_action, action_index, log_prob or None, value).

Step log: one dict per env step, with keys
    reward, dose, dose_index, terminated      always present
    tumor, immune, drug                       state after the step
    normal                                    normal cells, if a true_state_fn provides it
    log_prob                                  the agent's log-probability of its action, if any
The state values come from true_state_fn(env, obs, info) when given (the real simulator
state, unaffected by observation noise). Otherwise they are read from the observation, with
masked (NaN) values replaced by the previous value, and normal cells are not available.

Metrics, per episode (episode_metrics) and across episodes (aggregate: mean, std, 95% CI):
    return                                        total reward
    died                                          1 if the episode ended by patient death
                                                  (AhnChemoEnv terminates when normal cells
                                                  fall below 70% of their starting level)
    tumor_final_ratio, tumor_min_ratio, cleared   tumor size relative to the start; cleared
                                                  means at most CLEARED_RATIO of the start
    normal_min, normal_final, normal_min_ratio    toxicity (needs normal cells in the log)
    immune_min, immune_final                      immune cell levels
    total_drug, mean_dose, drug_conc_max          how much drug was used
    aggressive_frac, max_high_dose_streak         steps at or above HIGH_DOSE, longest run
    mean_action_prob                              how confident the policy was in its actions
    parse_fail_rate                               only for agents that can fail to give a
                                                  valid action; unused by the logit-based Agent

Used by scripts/evaluate.py, scripts/plot.py and tests/test_evaluation.py.
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