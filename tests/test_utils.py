import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import re
import subprocess

import yaml

from chemo_llm import prompts
from chemo_llm import utils
from chemo_llm.utils import DeathPenalty, ObservationHistory, ScalarDose, git_commit, load_config, make_env, set_seed, start_run

CONFIG_DIR = Path(__file__).parent.parent / "configs"
# setting3 has observation noise, so different seeds give different patients.
NOISY_CFG = {"env": {"id": "AhnChemoEnv-continuous-setting3", "max_t": 3,
                     "history": 2, "death_penalty": 50.0}}


# ---- load_config ----

@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({
        "env": {"id": "base-id", "max_t": 10},
        "ppo": {"lr": 1e-5, "epochs": 4},
        "doses": [0.0, 1.0],
    }))
    (tmp_path / "mymodel.yaml").write_text(yaml.safe_dump({
        "model": {"name": "some/model"},
        "ppo": {"lr": 3e-4},
    }))
    return tmp_path


def test_load_config_model_file_overrides_nested_keys_and_keeps_the_rest(config_dir):
    cfg = load_config(config_dir / "mymodel.yaml")
    assert cfg["ppo"] == {"lr": 3e-4, "epochs": 4}   # lr overridden, epochs kept
    assert cfg["env"] == {"id": "base-id", "max_t": 10}
    assert cfg["doses"] == [0.0, 1.0]
    assert cfg["model"] == {"name": "some/model"}


def test_load_config_sets_config_name(config_dir):
    assert load_config(config_dir / "mymodel.yaml")["config_name"] == "mymodel"


@pytest.mark.parametrize("name", ["qwen3_8b", "medgemma_4b", "deepseek_r1_7b",
                                  "qwen3_8b_no_history"])
def test_real_configs_have_what_the_code_reads(name):
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    assert cfg["config_name"] == name
    assert "history" in cfg["env"] and "death_penalty" in cfg["env"]
    assert cfg["model"]["name"] and "lora" in cfg["model"]
    assert len(cfg["doses"]) == len(prompts.ACTION_LABELS)
    assert cfg["doses"] == list(prompts.DOSE_LEVELS)
    for key in ["rollout_steps", "epochs", "minibatch_size", "micro_batch_size", "gamma",
                "gae_lambda", "clip_range", "vf_coef", "ent_coef", "max_grad_norm", "lr",
                "reward_scale", "value_head_lr", "n_envs", "target_kl"]:
        assert key in cfg["ppo"]
    for key in ["prompt_variant", "updates", "save_every", "value_warmup_updates"]:
        assert key in cfg["train"]


# ---- seeding ----

def test_set_seed_makes_all_generators_repeatable():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()
    set_seed(7)
    first = draw()
    set_seed(7)
    assert draw() == first


# ---- env ----

def test_env_spaces_match_what_prompts_assume():
    env = make_env(NOISY_CFG, seed=0)
    assert env.observation_space.shape == (3,)  # tumour, immune, drug (per row)
    assert env.action_space.shape == (1,)       # one dose
    assert env.action_space.low[0] == 0.0 and env.action_space.high[0] == 1.0
    assert min(prompts.DOSE_LEVELS) >= 0.0 and max(prompts.DOSE_LEVELS) <= 1.0


def test_reset_with_same_seed_gives_same_observation():
    env = make_env(NOISY_CFG, seed=0)
    first, _ = env.reset(seed=5)
    other, _ = env.reset(seed=6)
    again, _ = env.reset(seed=5)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_unseeded_resets_repeat_across_envs_with_the_same_seed():
    env_a, env_b = make_env(NOISY_CFG, seed=3), make_env(NOISY_CFG, seed=3)
    seq_a = [env_a.reset()[0] for _ in range(3)]
    seq_b = [env_b.reset()[0] for _ in range(3)]
    for a, b in zip(seq_a, seq_b):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(seq_a[0], seq_a[1])  # but resets within one env differ


def test_unseeded_resets_differ_for_different_env_seeds():
    a = make_env(NOISY_CFG, seed=3).reset()[0]
    b = make_env(NOISY_CFG, seed=4).reset()[0]
    assert not np.array_equal(a, b)


def test_scalar_dose_wraps_float_into_shape_one_array():
    class Echo(gym.Env):
        action_space = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        observation_space = gym.spaces.Box(0.0, 1.0, (1,), np.float32)

        def step(self, action):
            self.received = action
            return np.zeros(1, np.float32), 0.0, False, False, {}

    env = ScalarDose(Echo())
    env.step(0.25)
    assert env.unwrapped.received.shape == (1,)
    assert env.unwrapped.received.dtype == np.float32
    assert env.unwrapped.received[0] == pytest.approx(0.25)


def test_episode_ends_by_truncation_at_max_t_and_reports_return():
    env = make_env(NOISY_CFG, seed=0)
    env.reset(seed=0)
    rewards = []
    for _ in range(3):
        _, reward, terminated, truncated, info = env.step(0.5)
        rewards.append(reward)
    assert truncated and not terminated
    assert float(np.sum(info["episode"]["r"])) == pytest.approx(sum(rewards), abs=1e-5)


# ---- start_run ----

def test_start_run_writes_config_with_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"config_name": "mymodel", "ppo": {"lr": 1e-5}}
    run_dir, writer = start_run(cfg, seed=4)
    writer.close()
    assert run_dir.parent == Path("runs")
    assert run_dir.name.startswith("mymodel-seed4-")
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved["seed"] == 4
    assert saved["git_commit"] == git_commit()
    assert saved["ppo"] == {"lr": 1e-5}
    assert not (tmp_path / "chemo_llm").exists()


# ---- git_commit ----

def _fake_git(head, porcelain):
    def run(cmd, **kwargs):
        out = head if cmd[1] == "rev-parse" else porcelain
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    return run


def test_git_commit_clean_is_the_head_hash(monkeypatch):
    head = "a" * 40
    monkeypatch.setattr(utils.subprocess, "run", _fake_git(head, ""))
    assert git_commit() == head


def test_git_commit_appends_dirty_when_status_is_not_empty(monkeypatch):
    head = "b" * 40
    monkeypatch.setattr(utils.subprocess, "run", _fake_git(head, " M chemo_llm/utils.py"))
    assert git_commit() == head + "-dirty"


@pytest.mark.parametrize("error", [OSError("no git"), subprocess.CalledProcessError(128, "git")])
def test_git_commit_is_unknown_when_git_fails(monkeypatch, error):
    def run(cmd, **kwargs):
        raise error
    monkeypatch.setattr(utils.subprocess, "run", run)
    assert git_commit() == "unknown"


def test_git_commit_real_repo_format():
    value = git_commit()
    assert value == "unknown" or re.fullmatch(r"[0-9a-f]{40}(-dirty)?", value)


# ---- ObservationHistory / DeathPenalty ----

class CountingEnv(gym.Env):
    """Observation = [k, k, k] after k steps. Dies (terminated) at step `die_at`, truncates at `max_t`."""
    action_space = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
    observation_space = gym.spaces.Box(-1e9, 1e9, (3,), np.float32)

    def __init__(self, die_at=None, max_t=100):
        self.die_at, self.max_t, self.k = die_at, max_t, 0

    def reset(self, *, seed=None, options=None):
        self.k = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.k += 1
        return (np.full(3, self.k, np.float32), -1.0,
                self.k == self.die_at, self.k == self.max_t, {})


def test_observation_history_stacks_rows_oldest_first_and_caps_length():
    env = ObservationHistory(CountingEnv(), 3)
    obs, _ = env.reset()
    assert obs.shape == (1, 3)
    obs, *_ = env.step(0.0)
    assert obs.shape == (2, 3)
    for _ in range(3):
        obs, *_ = env.step(0.0)  # now k = 4
    assert obs.shape == (3, 3)
    np.testing.assert_array_equal(obs[:, 0], [2, 3, 4])  # oldest first, step 0 and 1 dropped


def test_observation_history_reset_clears_rows():
    env = ObservationHistory(CountingEnv(), 3)
    env.reset()
    for _ in range(3):
        env.step(0.0)
    obs, _ = env.reset()
    assert obs.shape == (1, 3)
    obs, *_ = env.step(0.0)
    np.testing.assert_array_equal(obs[:, 0], [0, 1])


def test_death_penalty_is_constant_on_termination():
    env = DeathPenalty(CountingEnv(die_at=3, max_t=10), 50.0)
    env.reset()
    rewards = [env.step(0.0)[1] for _ in range(3)]
    assert rewards == [-1.0, -1.0, pytest.approx(-51.0)]  # death at step 3: -1 - 50


def test_death_penalty_leaves_truncation_alone():
    env = DeathPenalty(CountingEnv(max_t=3), 50.0)
    env.reset()
    results = [env.step(0.0) for _ in range(3)]
    assert [r[1] for r in results] == [-1.0, -1.0, -1.0]
    assert results[-1][3] and not results[-1][2]


REAL_CFG = {"env": {"id": "AhnChemoEnv-continuous-setting2", "max_t": 40,
                    "history": 2, "death_penalty": 50.0}}


def test_make_env_observation_shape_is_capped_at_history_plus_one():
    env = make_env(REAL_CFG, seed=0)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (1, 3)
    for _ in range(5):
        obs, *_ = env.step(0.5)
    assert obs.shape == (3, 3)


def test_make_env_training_flag_controls_death_penalty():
    def run(training):
        env = make_env(REAL_CFG, seed=0, training=training)
        env.reset(seed=105)  # setting2 seed 105 dies within ~25 steps when never dosed
        done, info = False, {}
        while not done:
            _, _, terminated, truncated, info = env.step(0.0)
            done = terminated or truncated
        return float(np.sum(info["episode"]["r"])), info["episode"]["l"], terminated

    eval_return, length, died = run(False)
    train_return, train_length, train_died = run(True)
    assert died and train_died and length == train_length
    # constant penalty of 50, included in the RecordEpisodeStatistics return
    assert train_return == pytest.approx(eval_return - 50.0, abs=1e-3)


def test_make_env_zero_death_penalty_trains_on_the_env_reward():
    cfg = {"env": {**REAL_CFG["env"], "death_penalty": 0}}

    def run(training):
        env = make_env(cfg, seed=0, training=training)
        env.reset(seed=105)  # dies when never dosed
        done, rewards = False, []
        while not done:
            _, r, terminated, truncated, info = env.step(0.0)
            rewards.append(float(r))
            done = terminated or truncated
        return rewards, float(np.sum(info["episode"]["r"])), terminated

    eval_rewards, eval_return, died = run(False)
    train_rewards, train_return, train_died = run(True)
    assert died and train_died
    assert train_rewards == pytest.approx(eval_rewards)
    assert train_return == pytest.approx(eval_return)


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_env_reward_ablation_config_differs_from_qwen3_only_in_penalty_and_name():
    base = _flatten(load_config(CONFIG_DIR / "qwen3_8b.yaml"))
    abl = _flatten(load_config(CONFIG_DIR / "qwen3_8b_env_reward.yaml"))
    differing = {k for k in base.keys() | abl.keys() if base.get(k, "<missing>") != abl.get(k, "<missing>")}
    assert differing == {"config_name", "env.death_penalty"}
    assert abl["env.death_penalty"] == 0
    assert base["env.death_penalty"]


def test_make_env_rejects_history_longer_than_the_prompt_can_show():
    cfg = {"env": {**REAL_CFG["env"], "history": prompts.MAX_HISTORY + 1}}
    with pytest.raises(AssertionError):
        make_env(cfg, seed=0)


def test_no_history_config_differs_from_qwen3_only_in_history_and_name():
    base = _flatten(load_config(CONFIG_DIR / "qwen3_8b.yaml"))
    nh = _flatten(load_config(CONFIG_DIR / "qwen3_8b_no_history.yaml"))
    differing = {k for k in base.keys() | nh.keys() if base.get(k, "<missing>") != nh.get(k, "<missing>")}
    assert differing == {"config_name", "env.history"}
    assert nh["env.history"] == 0


def test_no_history_config_gives_single_row_obs_and_same_prompt_as_current_obs():
    cfg = load_config(CONFIG_DIR / "qwen3_8b_no_history.yaml")
    env = make_env(cfg, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, 3)
    for _ in range(3):
        obs, *_ = env.step(0.5)
        assert obs.shape == (1, 3)
        text = prompts.user_prompt(obs)
        assert text == prompts.user_prompt(obs[-1])
        assert "Earlier steps" not in text and "oldest shown" not in text
