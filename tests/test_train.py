"""Smoke test: scripts/train.py runs end to end with a stub model and a tiny config."""
import importlib.util
from pathlib import Path

import pytest
import yaml

from conftest import StubModel, StubTokenizer

ROOT = Path(__file__).parent.parent


def load_train_script():
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    base["env"]["max_t"] = 4
    base["ppo"].update(rollout_steps=6, epochs=1, minibatch_size=4, micro_batch_size=2)
    base["train"].update(updates=3, save_every=2)
    (config_dir / "base.yaml").write_text(yaml.safe_dump(base))
    (config_dir / "stub.yaml").write_text(yaml.safe_dump({"model": {"name": "stub"}}))
    return config_dir / "stub.yaml"


def test_train_writes_run_directory(tiny_config, tmp_path, monkeypatch, stub_prompt):
    train = load_train_script()
    monkeypatch.setattr(train, "load_model", lambda cfg: (StubModel(), StubTokenizer()))
    monkeypatch.chdir(tmp_path)

    train.main([str(tiny_config), "--seed", "3"])

    run_dirs = list((tmp_path / "runs").glob("stub-seed3-*"))
    assert len(run_dirs) == 1
    run = run_dirs[0]
    saved = yaml.safe_load((run / "config.yaml").read_text())
    assert saved["seed"] == 3 and saved["train"]["updates"] == 3
    assert list(run.glob("events.out.tfevents.*"))
    assert (run / "adapter-0002" / "value_head.pt").exists()  # save_every=2 -> update 2 only
    assert not (run / "adapter-0001").exists() and not (run / "adapter-0003").exists()
    assert (run / "adapter-final" / "value_head.pt").exists()
