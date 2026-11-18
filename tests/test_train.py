"""Smoke test: scripts/train.py runs end to end with a stub model and a tiny config."""
import importlib.util
from pathlib import Path

import pytest
import yaml

from conftest import stub_load_model

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
    base["ppo"].update(rollout_steps=6, n_envs=2, target_kl=1.0e9, epochs=1, minibatch_size=4, micro_batch_size=2)
    base["train"].update(updates=3, save_every=2, value_warmup_updates=1)
    (config_dir / "base.yaml").write_text(yaml.safe_dump(base))
    (config_dir / "stub.yaml").write_text(yaml.safe_dump({"model": {"name": "stub"}}))
    return config_dir / "stub.yaml"


def test_train_writes_run_directory(tiny_config, tmp_path, monkeypatch, stub_prompt):
    train = load_train_script()
    monkeypatch.setattr(train, "load_model", stub_load_model)
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


def test_train_builds_two_param_groups_with_configured_lrs(tiny_config, tmp_path, monkeypatch, stub_prompt):
    import torch

    train = load_train_script()
    monkeypatch.setattr(train, "load_model", stub_load_model)
    monkeypatch.chdir(tmp_path)
    captured = []
    real_adamw = torch.optim.AdamW

    def spy(*args, **kwargs):
        optimizer = real_adamw(*args, **kwargs)
        captured.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    train.main([str(tiny_config), "--seed", "3"])

    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())["ppo"]
    assert len(captured) == 1
    groups = captured[0].param_groups
    assert [g["lr"] for g in groups] == [cfg["lr"], cfg["value_head_lr"]]
    assert cfg["lr"] != cfg["value_head_lr"]
    assert all(len(g["params"]) > 0 for g in groups)


def test_resume_continues_in_same_run_directory(tiny_config, tmp_path, monkeypatch, stub_prompt):
    import torch

    train = load_train_script()
    monkeypatch.setattr(train, "load_model", stub_load_model)
    monkeypatch.chdir(tmp_path)

    train.main([str(tiny_config), "--seed", "3"])  # updates=3, save_every=2 -> adapter-0002
    run = next((tmp_path / "runs").glob("stub-seed3-*"))
    checkpoint = run / "adapter-0002"
    state = torch.load(checkpoint / "trainer.pt")
    assert state["update"] == 2 and "optimizer" in state
    assert torch.load(run / "adapter-final" / "trainer.pt")["update"] == 3

    # Same config, but train for 6 updates in total
    base = yaml.safe_load((tiny_config.parent / "base.yaml").read_text())
    base["train"]["updates"] = 6
    (tiny_config.parent / "base.yaml").write_text(yaml.safe_dump(base))

    train.main([str(tiny_config), "--seed", "3", "--resume", str(checkpoint)])

    assert len(list((tmp_path / "runs").iterdir())) == 1  # no new run directory
    assert (run / "adapter-0004" / "trainer.pt").exists()
    assert (run / "adapter-0006" / "trainer.pt").exists()
    assert not (run / "adapter-0003").exists()
    final = torch.load(run / "adapter-final" / "trainer.pt")
    assert final["update"] == 6
    assert len(list(run.glob("events.out.tfevents.*"))) >= 1


def test_train_logs_compute_and_ppo_scalars(tiny_config, tmp_path, monkeypatch, stub_prompt):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    train = load_train_script()
    monkeypatch.setattr(train, "load_model", stub_load_model)
    monkeypatch.chdir(tmp_path)
    train.main([str(tiny_config), "--seed", "3"])

    run = next((tmp_path / "runs").glob("stub-seed3-*"))
    assert yaml.safe_load((run / "config.yaml").read_text())["git_commit"]
    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    tags = events.Tags()["scalars"]
    assert len(events.Scalars("compute/minutes")) == 3  # updates=3
    assert all(p.value >= 0 for p in events.Scalars("compute/minutes"))
    assert "ppo/start_ratio_error" in tags and "ppo/explained_variance" in tags
    assert ("compute/peak_vram_gb" in tags) == torch.cuda.is_available()


def test_value_warmup_updates_leave_lora_unchanged(tiny_config, tmp_path, monkeypatch, stub_prompt):
    import torch

    train = load_train_script()
    initial = []

    def capturing_load_model(cfg, adapter_dir=None):
        model, tokenizer = stub_load_model(cfg, adapter_dir)
        initial.append(model.lora.weight.detach().clone())
        return model, tokenizer

    monkeypatch.setattr(train, "load_model", capturing_load_model)
    monkeypatch.chdir(tmp_path)
    base = yaml.safe_load((tiny_config.parent / "base.yaml").read_text())
    base["train"]["value_warmup_updates"] = 3  # all 3 updates are warm-up
    (tiny_config.parent / "base.yaml").write_text(yaml.safe_dump(base))

    train.main([str(tiny_config), "--seed", "3"])

    run = next((tmp_path / "runs").glob("stub-seed3-*"))
    final = torch.load(run / "adapter-final" / "adapter_model.pt")
    assert torch.equal(final["weight"], initial[0])
