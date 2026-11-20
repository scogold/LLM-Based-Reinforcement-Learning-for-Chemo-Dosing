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


def write_log(path):
    path.write_text(json.dumps([fake_episode(0)]))
    return path


def test_split_label_with_and_without_equals():
    plot = load_plot_script()
    default = lambda p: "DEFAULT:" + p.name
    assert plot.split_label("dir/a.json=My label", default) == (Path("dir/a.json"), "My label")
    assert plot.split_label("dir/a.json", default) == (Path("dir/a.json"), "DEFAULT:a.json")


def test_split_label_keeps_special_characters():
    path, label = load_plot_script().split_label("x/a.json=Qwen 8b (PPO) + history", lambda p: "d")
    assert path == Path("x/a.json")
    assert label == "Qwen 8b (PPO) + history"


def test_load_logs_uses_label_or_file_stem(tmp_path):
    plot = load_plot_script()
    labelled = write_log(tmp_path / "a.json")
    plain = write_log(tmp_path / "b.json")
    methods = plot.load_logs([f"{labelled}=Qwen 8b (PPO) + history", str(plain)])
    assert set(methods) == {"Qwen 8b (PPO) + history", "b"}
    assert methods["b"][0]["seed"] == 0


def test_load_logs_skips_labelled_non_episode_file(tmp_path):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"not": "episodes"}))
    assert load_plot_script().load_logs([f"{summary}=Label"]) == {}


def test_learning_curve_legend_uses_label_else_folder_name(tmp_path, monkeypatch):
    plot = load_plot_script()
    runs = []
    for name in ("run_a", "run_b"):
        writer = SummaryWriter(tmp_path / name)
        for step, value in [(10, -5.0), (20, -4.0)]:
            writer.add_scalar("episode/return", value, step)
        writer.close()
        runs.append(tmp_path / name)
    seen = []
    real_close = plot.plt.close

    def spy(fig):
        seen.append(fig.axes[0].get_legend_handles_labels()[1])
        real_close(fig)

    monkeypatch.setattr(plot.plt, "close", spy)
    out = tmp_path / "out"
    out.mkdir()
    plot.plot_learning_curve([f"{runs[0]}=PPO (long) + history", str(runs[1])], out)
    assert seen == [["PPO (long) + history", "run_b"]]
