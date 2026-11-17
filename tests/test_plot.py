"""scripts/plot.py: evaluation-log figures and the training learning curve."""
import importlib.util
import json
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent.parent


def load_plot_script():
    spec = importlib.util.spec_from_file_location("plot_script", ROOT / "scripts" / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_episode(seed):
    steps = [{"tumor": 1.0 - 0.1 * i, "immune": 0.5, "drug": 0.1 * i, "normal": 0.9,
              "dose": 0.5 * (i % 2), "dose_index": i % 2, "reward": -1.0, "terminated": False}
             for i in range(4)]
    return {"seed": seed, "start": {"tumor": 1.0, "normal": 1.0}, "steps": steps}


def test_main_without_logs_or_runs_is_an_error():
    with pytest.raises(SystemExit):
        load_plot_script().main([])


def test_main_runs_writes_learning_curve(tmp_path):
    run = tmp_path / "run"
    writer = SummaryWriter(run)
    for step, value in [(10, -50.0), (20, -40.0), (30, -30.0)]:
        writer.add_scalar("episode/return", value, step)
    writer.close()
    out = tmp_path / "out"

    load_plot_script().main(["--runs", str(run), "--out", str(out)])

    assert (out / "learning_curve.png").stat().st_size > 0
    assert not (out / "metrics.png").exists()  # no evaluation logs given


def test_main_with_evaluation_logs_writes_three_figures(tmp_path):
    logs = []
    for name in ("zero_shot", "ppo"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps([fake_episode(0), fake_episode(1)]))
        logs.append(str(path))
    (tmp_path / "summary.json").write_text(json.dumps({"not": "episodes"}))  # skipped
    out = tmp_path / "out"

    load_plot_script().main([*logs, str(tmp_path / "summary.json"), "--out", str(out)])

    for name in ("trajectories.png", "metrics.png", "doses.png"):
        assert (out / name).stat().st_size > 0
    assert not (out / "learning_curve.png").exists()
