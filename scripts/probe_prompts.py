"""Probe the LLM's dose choices on hand-made situations, under each prompt variant.

Purpose: check that a prompt produces sensible behavior before any PPO training, and
produce the prompt-comparison table for the paper. For each (prompt variant, situation)
it reports the probability of each dose option A-E and the probability-weighted dose.

Usage (GPU machine, repo root):
    python scripts/probe_prompts.py configs/qwen3_8b.yaml
    python scripts/probe_prompts.py configs/qwen3_8b.yaml --adapter runs/<run>/adapter

Without --adapter the LoRA adapter is switched off (plain zero-shot model).
Output: a table on screen and <out>/probe-<config>.csv
"""
import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chemo_llm import prompts

# (name, [tumor, immune, drug], what a sensible policy would do)
# AhnChemoEnv starts at about tumor 0.2, immune 0.005, drug 0.0. In baseline rollouts immune
# reaches ~0.6 within ~10 days, an untreated tumor grows to ~0.4, and drug follows the dose
# up to 1.0. Obs ranges: tumor 0-2, immune 0-2, drug 0-1.
STATES = [
    ("start of treatment",         [0.2,  0.005, 0.0], "begin dosing"),
    ("tumor growing, no drug",     [0.35, 0.6,   0.0], "dose (higher is fine)"),
    ("high drug already in blood", [0.25, 0.6,   0.9], "no extra drug"),
    ("big tumor, drug lingering",  [0.4,  0.6,   0.5], "moderate dose or wait"),
    ("tumor almost gone",          [0.05, 0.7,   0.3], "stop or very low dose"),
    ("tumor gone, no drug",        [0.0,  0.7,   0.0], "no drug"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--variants", nargs="+", default=list(prompts.PROMPT_VARIANTS))
    ap.add_argument("--adapter", default=None, help="probe a trained adapter instead of zero-shot")
    ap.add_argument("--out", default="runs/probe")
    args = ap.parse_args(argv)

    import torch
    from chemo_llm.agent import Agent, load_model
    from chemo_llm.utils import load_config

    cfg = load_config(args.config)
    model, tokenizer = load_model(cfg, adapter_dir=args.adapter)
    agent = Agent(model, tokenizer, args.variants[0], cfg["doses"])
    agent.model.eval()
    ctx = nullcontext() if args.adapter else agent.model.disable_adapter()

    labels = prompts.ACTION_LABELS
    rows = []
    with ctx, torch.no_grad():
        for variant in args.variants:
            agent.prompt_variant = variant
            for name, obs, expect in STATES:
                log_probs, _ = agent._forward([obs])
                probs = log_probs[0].exp().tolist()
                rows.append({
                    "variant": variant, "state": name, "tumor": obs[0], "immune": obs[1],
                    "drug": obs[2], **{f"p_{l}": p for l, p in zip(labels, probs)},
                    "expected_dose": sum(p * d for p, d in zip(probs, cfg["doses"])),
                    "top": labels[max(range(len(probs)), key=probs.__getitem__)],
                    "sensible": expect,
                })

    head = f"{'variant':<8}{'situation':<28}" + "".join(f"{l:>7}" for l in labels) \
        + f"{'E[dose]':>9}{'top':>5}   sensible"
    print(head)
    for r in rows:
        print(f"{r['variant']:<8}{r['state']:<28}"
              + "".join(f"{r['p_' + l]:>7.2f}" for l in labels)
              + f"{r['expected_dose']:>9.2f}{r['top']:>5}   {r['sensible']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"probe-{cfg['config_name']}{'-trained' if args.adapter else ''}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved to {path}")
    return rows


if __name__ == "__main__":
    main()
