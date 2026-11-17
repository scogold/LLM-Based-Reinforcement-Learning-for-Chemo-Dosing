"""Score a policy on AhnChemoEnv with the same seeds and metrics for every method.

Modes
  baseline   non-LLM reference policies (no GPU, no model; runs on any laptop)
  zeroshot   the LLM with its LoRA adapter switched off (same weights and prompt as training)
  trained    the LLM with a PPO-trained adapter (--adapter <dir saved by Agent.save>)

Metrics are computed from the true simulator state (chemo_llm/true_state.py), including
normal cells, even though the agent only sees the (possibly noisy) observation.

Each policy writes <out>/<label>.json (raw per-step logs, read by scripts/plot.py) and the
summary table goes to the terminal and <out>/summary.json.
Use the same --seeds and --max-t for every method you want to compare.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chemo_llm.evaluation import (aggregate, baseline_policy, evaluate_policy,  # noqa: E402
                                  save_episode_logs)
from chemo_llm.true_state import ahn_true_state  # noqa: E402

BASELINES = ("zero", "half", "max", "random")
TABLE_KEYS = ("return", "died", "tumor_final_ratio", "normal_min_ratio", "total_drug",
              "aggressive_frac", "mean_action_prob")


def format_table(results, keys=TABLE_KEYS):
    def cell(summary, key):
        s = summary.get(key)
        return "-" if s is None else f"{s['mean']:.3g}±{s['ci95']:.2g}"
    width = max(len(n) for n in results) + 2
    lines = [f"{'policy':<{width}}" + "".join(f"{k:>20}" for k in keys)]
    for name, summary in results.items():
        lines.append(f"{name:<{width}}" + "".join(f"{cell(summary, k):>20}" for k in keys))
    return "\n".join(lines)


def _llm_policy(cfg, args):
    """Build (act_fn, context_manager) for zeroshot / trained modes. Needs a GPU."""
    from contextlib import nullcontext
    from chemo_llm.agent import Agent, load_model

    adapter = args.adapter if args.mode == "trained" else None
    model, tokenizer = load_model(cfg, adapter_dir=adapter)
    agent = Agent(model, tokenizer, args.variant, cfg["doses"])
    if adapter and (Path(adapter) / "value_head.pt").exists():
        agent.load_value_head(adapter)
    ctx = agent.model.disable_adapter() if args.mode == "zeroshot" else nullcontext()
    return (lambda obs: agent.act(obs, greedy=True)), ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--mode", choices=("baseline", "zeroshot", "trained"), required=True)
    ap.add_argument("--baseline", nargs="+", default=["all"],
                    help=f"names from {BASELINES}, or 'all'")
    ap.add_argument("--variant", default="expert", help="prompt variant: base | expert | reward")
    ap.add_argument("--adapter", default=None, help="adapter dir (trained mode)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--max-t", type=int, default=None, help="override episode length")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--label", default=None, help="name used for the output file")
    ap.add_argument("--use-obs-state", action="store_true",
                    help="score from the observation instead of the true simulator state "
                         "(noisy in settings 3-5, and no normal cells)")
    args = ap.parse_args(argv)
    if args.mode == "trained" and not args.adapter:
        ap.error("--mode trained needs --adapter")

    from chemo_llm.utils import load_config, make_env  # needs DTRGym

    cfg = load_config(args.config)
    if args.max_t:
        cfg["env"]["max_t"] = args.max_t
    env = make_env(cfg, seed=0)
    true_state_fn = None if args.use_obs_state else ahn_true_state

    out = Path(args.out or f"runs/eval-{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)

    jobs = []  # (label, act_fn, context)
    if args.mode == "baseline":
        names = BASELINES if "all" in args.baseline else tuple(args.baseline)
        for n in names:
            jobs.append((n, baseline_policy(n, cfg["doses"], seed=0), None))
    else:
        act, ctx = _llm_policy(cfg, args)
        label = args.label or f"{cfg['config_name']}-{args.mode}-{args.variant}"
        jobs.append((label, act, ctx))

    results = {}
    for label, act, ctx in jobs:
        if ctx is not None:
            with ctx:
                metrics, logs = evaluate_policy(env, act, args.seeds, true_state_fn)
        else:
            metrics, logs = evaluate_policy(env, act, args.seeds, true_state_fn)
        save_episode_logs(out / f"{label}.json", logs)
        results[label] = aggregate(metrics)
        print(f"finished {label}")

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print("\n" + format_table(results))
    print(f"\nsaved to {out}")
    return results


if __name__ == "__main__":
    main()
