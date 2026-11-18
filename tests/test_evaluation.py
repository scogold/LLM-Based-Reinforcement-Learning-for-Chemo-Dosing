"""Tests for chemo_llm.evaluation."""
import math
from pathlib import Path

from chemo_llm import evaluation as M


# ---------- metrics ----------
def _episode(doses, tumors, with_normal=True, parse=True):
    steps = []
    for i, (d, t) in enumerate(zip(doses, tumors)):
        s = {"reward": 1.0, "tumor": t, "immune": 0.5, "drug": 0.1, "dose": d}
        if with_normal:
            s["normal"] = 1.0 - 0.1 * d
        if parse:
            s["parse_ok"] = (i != 1)
        steps.append(s)
    return steps


def test_episode_metrics_basic():
    m = M.episode_metrics(_episode([0.0, 1.0, 1.0, 0.25], [1.0, 0.8, 0.4, 0.02]), tumor0=1.0)
    assert m["return"] == 4.0
    assert math.isclose(m["tumor_final_ratio"], 0.02)
    assert m["cleared"] == 1.0
    assert math.isclose(m["total_drug"], 2.25)
    assert math.isclose(m["aggressive_frac"], 0.5)
    assert m["max_high_dose_streak"] == 2.0
    assert math.isclose(m["parse_fail_rate"], 0.25)
    assert math.isclose(m["normal_min"], 0.9)


def test_missing_optional_fields_give_none():
    m = M.episode_metrics(_episode([0.5, 0.5], [1.0, 0.9], with_normal=False, parse=False))
    assert m["normal_min"] is None and m["parse_fail_rate"] is None


def test_empty_episode_raises():
    try:
        M.episode_metrics([])
        assert False
    except ValueError:
        pass


def test_aggregate_skips_none_and_computes_stats():
    eps = [{"return": 1.0, "x": None}, {"return": 3.0, "x": None}]
    s = M.aggregate(eps)
    assert math.isclose(s["return"]["mean"], 2.0) and "x" not in s
    assert s["return"]["ci95"] > 0


def test_consistency_and_histogram():
    assert M.consistency([1, 1, 1, 2]) == 0.75
    assert M.dose_histogram([0, 0, 4, 4, 4]) == [0.4, 0.0, 0.0, 0.0, 0.6]


def test_compare_paired():
    r = M.compare([5, 6, 7, 8], [3, 4, 5, 6])
    assert math.isclose(r["mean_diff"], 2.0)


def test_mean_action_prob():
    steps = _episode([0.5, 0.5], [1.0, 0.9], parse=False)
    for s in steps:
        s["log_prob"] = math.log(0.5)
    assert math.isclose(M.episode_metrics(steps)["mean_action_prob"], 0.5)


# ---------- running episodes ----------
class FakeEnv:
    """Gymnasium-style env: tumor shrinks 10% per step; dies at `die_at`, else truncates at `horizon`."""
    def __init__(self, horizon=5, die_at=None, nan_at=None):
        self.horizon, self.die_at, self.nan_at = horizon, die_at, nan_at

    def reset(self, seed=None):
        self.t, self.tumor, self.seed = 0, 1.0, seed
        return [1.0, 0.5, 0.0], {}

    def step(self, action):
        self.t += 1
        self.tumor *= 0.9
        obs = [self.tumor, 0.5, float(action)]
        if self.nan_at == self.t:
            obs[0] = float("nan")
        died = self.die_at is not None and self.t >= self.die_at
        return obs, 1.0, died, (not died and self.t >= self.horizon), {}


def test_run_episode_logs_every_step():
    steps, start = M.run_episode(FakeEnv(5), M.baseline_policy("max", [0, 0.5, 1.0]), seed=3)
    assert len(steps) == 5 and start["tumor"] == 1.0
    assert all(s["dose"] == 1.0 and s["dose_index"] == 2 for s in steps)
    assert not any(s["terminated"] for s in steps)
    assert "log_prob" not in steps[0]


def test_run_episode_death_ends_episode_and_sets_died():
    steps, start = M.run_episode(FakeEnv(10, die_at=3), M.baseline_policy("zero", [0, 1]), seed=0)
    assert len(steps) == 3 and steps[-1]["terminated"]
    m = M.episode_metrics(steps, tumor0=start["tumor"])
    assert m["died"] == 1.0


def test_survivor_has_died_zero():
    steps, start = M.run_episode(FakeEnv(4), M.baseline_policy("half", [0, 0.5, 1.0]), seed=0)
    assert M.episode_metrics(steps)["died"] == 0.0


def test_nan_observation_is_carried_forward():
    steps, _ = M.run_episode(FakeEnv(4, nan_at=2), M.baseline_policy("zero", [0, 1]), seed=0)
    assert not any(math.isnan(s["tumor"]) for s in steps)
    assert steps[1]["tumor"] == steps[0]["tumor"]


def test_stacked_history_obs_uses_latest_row():
    assert M._last_row([[1, 2, 3], [4, 5, 6]]) == [4.0, 5.0, 6.0]
    assert M._last_row([4, 5, 6]) == [4.0, 5.0, 6.0]


def test_true_state_fn_is_used_and_normal_is_tracked():
    fn = lambda env, obs, info: {"tumor": 0.1, "immune": 0.2, "drug": 0.3, "normal": 0.9}
    steps, start = M.run_episode(FakeEnv(3), M.baseline_policy("zero", [0, 1]), 0, fn)
    assert steps[0]["normal"] == 0.9 and start["normal"] == 0.9
    assert M.episode_metrics(steps, normal0=start["normal"])["normal_min_ratio"] == 1.0


def test_agent_style_act_log_prob_is_logged():
    act = lambda obs: (0.5, 1, math.log(0.25), 0.0)
    steps, _ = M.run_episode(FakeEnv(3), act, 0)
    m = M.episode_metrics(steps)
    assert math.isclose(m["mean_action_prob"], 0.25)


def test_evaluate_policy_same_seeds_same_results():
    a, _ = M.evaluate_policy(FakeEnv(4), M.baseline_policy("random", [0, 0.5, 1.0], seed=1), [0, 1, 2])
    b, _ = M.evaluate_policy(FakeEnv(4), M.baseline_policy("random", [0, 0.5, 1.0], seed=1), [0, 1, 2])
    assert [m["total_drug"] for m in a] == [m["total_drug"] for m in b]
    assert len(a) == 3


def test_baseline_names():
    import pytest
    with pytest.raises(ValueError):
        M.baseline_policy("nope", [0, 1])
    assert M.baseline_policy("zero", [0, 1])(None)[0] == 0
    assert M.baseline_policy("max", [0, 1])(None)[0] == 1

def test_evaluate_script_merges_summary_across_calls(tmp_path):
    import importlib.util
    import json
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location("evaluate_script", root / "scripts" / "evaluate.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "eval"
    common = [str(root / "configs" / "qwen3_8b.yaml"), "--mode", "baseline", "--seeds", "100",
              "--max-t", "5", "--out", str(out)]
    script.main(common + ["--baseline", "zero"])
    script.main(common + ["--baseline", "max"])
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {"zero", "max"}
