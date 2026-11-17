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
from chemo_llm.utils import ScalarDose, git_commit, load_config, make_env, set_seed, start_run

CONFIG_DIR = Path(__file__).parent.parent / "configs"
# setting3 has observation noise, so different seeds give different patients.
NOISY_CFG = {"env": {"id": "AhnChemoEnv-continuous-setting3", "max_t": 3}}


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


@pytest.mark.parametrize("name", ["qwen3_8b", "medgemma_4b", "deepseek_r1_7b"])
def test_real_configs_have_what_the_code_reads(name):
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    assert cfg["config_name"] == name
    assert cfg["model"]["name"] and "lora" in cfg["model"]
    assert len(cfg["doses"]) == len(prompts.ACTION_LABELS)
    assert cfg["doses"] == list(prompts.DOSE_LEVELS)
    for key in ["rollout_steps", "epochs", "minibatch_size", "micro_batch_size", "gamma",
                "gae_lambda", "clip_range", "vf_coef", "ent_coef", "max_grad_norm", "lr",
                "reward_scale", "value_head_lr"]:
        assert key in cfg["ppo"]
    for key in ["prompt_variant", "updates", "save_every"]:
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
    assert env.observation_space.shape == (3,)  # tumour, immune, drug
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
