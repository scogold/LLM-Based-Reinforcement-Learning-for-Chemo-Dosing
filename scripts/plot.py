"""Make the paper figures from logs written by scripts/evaluate.py and scripts/train.py.

    python scripts/plot.py runs/eval-main/*.json --out runs/eval-main/plots
    python scripts/plot.py --runs runs/qwen3_8b-seed0-<time> --out runs/plots

Each evaluation log file is one method (its name becomes the legend label). Files that are
not episode logs, such as summary.json, are skipped. Figures:
  trajectories.png    tumor, immune, drug and dose over time for one seed, all methods
                      (plus normal cells if the logs have them)
  metrics.png         mean with 95% CI across seeds for the headline metrics
  doses.png           share of steps spent at each dose level, per method (in .gitignore for now)
  learning_curve.png  (with --runs) episode return during training, one line per run,
                      read from the run's TensorBoard logs
Runs headless (no display needed).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chemo_llm.evaluation import aggregate, dose_histogram, episode_metrics  # noqa: E402

STEP_DAYS = 0.25  # 6 hours per env step
HEADLINE = (("return", "cost = -return (lower is better)"),
            ("died", "fraction of episodes ending in death"),
            ("tumor_final_ratio", "final tumor / starting tumor"),
            ("total_drug", "total drug given"))


def load_logs(paths):
    """{label: list of episode logs}, skipping files that are not episode logs."""
    methods = {}
    for p in map(Path, paths):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict) and "steps" in data[0]:
            methods[p.stem] = data
    return methods


def _metrics(episodes):
    return [episode_metrics(e["steps"], tumor0=e["start"].get("tumor"),
                            normal0=e["start"].get("normal")) for e in episodes]


def plot_trajectories(methods, out, seed=None):
    first = next(iter(methods.values()))
    seed = first[0]["seed"] if seed is None else seed
    panels = [("tumor", "tumor cells"), ("immune", "immune cells"), ("drug", "drug in blood"),
              ("dose", "dose given")]
    if all("normal" in ep["steps"][0] for eps in methods.values() for ep in eps):
        panels.insert(3, ("normal", "normal cells"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.4 * len(panels)), sharex=True)
    for label, episodes in methods.items():
        ep = next((e for e in episodes if e["seed"] == seed), None)
        if ep is None:
            continue
        days = [(i + 1) * STEP_DAYS for i in range(len(ep["steps"]))]
        for ax, (key, title) in zip(axes, panels):
            ax.plot(days, [s[key] for s in ep["steps"]], label=label, linewidth=1.3)
            ax.set_ylabel(title)
    axes[-1].set_xlabel("day")
    axes[0].set_title(f"One patient (seed {seed})")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=150)
    plt.close(fig)


def plot_metrics(methods, out):
    summaries = {label: aggregate(_metrics(eps)) for label, eps in methods.items()}
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(3.2 * len(HEADLINE), 3.6))
    for ax, (key, title) in zip(axes, HEADLINE):
        sign = -1 if key == "return" else 1
        labels = [l for l in summaries if key in summaries[l]]
        means = [sign * summaries[l][key]["mean"] for l in labels]
        errs = [summaries[l][key]["ci95"] for l in labels]
        ax.bar(range(len(labels)), means, yerr=errs, capsize=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=150)
    plt.close(fig)


def plot_doses(methods, out, n_levels=5):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / len(methods)
    for j, (label, episodes) in enumerate(methods.items()):
        levels = [s["dose_index"] for e in episodes for s in e["steps"]]
        hist = dose_histogram(levels, n_levels)
        ax.bar([i + j * width for i in range(n_levels)], hist, width, label=label)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(n_levels)])
    ax.set_xticklabels([chr(65 + i) for i in range(n_levels)])
    ax.set_xlabel("dose option (A = none ... E = maximum)")
    ax.set_ylabel("share of steps")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "doses.png", dpi=150)
    plt.close(fig)


def plot_learning_curve(run_dirs, out):
    """Episode return (env units) against training env steps, one line per training run."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for run_dir in map(Path, run_dirs):
        events = EventAccumulator(str(run_dir))  # reads every event file, so resumed runs too
        events.Reload()
        points = events.Scalars("episode/return")
        ax.plot([p.step for p in points], [p.value for p in points], marker=".", label=run_dir.name)
    ax.set_xlabel("training env steps")
    ax.set_ylabel("episode return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="JSON log files from scripts/evaluate.py")
    ap.add_argument("--runs", nargs="+", default=[], help="training run directories, for the learning curve")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--seed", type=int, default=None, help="patient seed for the trajectory plot")
    ap.add_argument("--levels", type=int, default=5, help="number of dose options")
    args = ap.parse_args(argv)
    if not args.logs and not args.runs:
        ap.error("give evaluation log files, --runs, or both")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.logs:
        methods = load_logs(args.logs)
        if not methods:
            sys.exit("no episode logs found in the given files")
        plot_trajectories(methods, out, args.seed)
        plot_metrics(methods, out)
        plot_doses(methods, out, args.levels)
        print(f"wrote 3 figures for {list(methods)} to {out}")
    if args.runs:
        plot_learning_curve(args.runs, out)
        print(f"wrote learning_curve.png for {len(args.runs)} run(s) to {out}")


if __name__ == "__main__":
    main()
