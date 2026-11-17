"""Train the LLM dosing policy with PPO.

    python scripts/train.py configs/qwen3_8b.yaml --seed 0

Creates runs/<config>-seed<seed>-<time>/ containing:
    config.yaml          the resolved config and seed
    events.out.tfevents  TensorBoard logs (tensorboard --logdir runs)
    adapter-0020/ ...    checkpoints every train.save_every updates
    adapter-final/       the final LoRA adapter and value head

Evaluate a checkpoint with:
    python scripts/evaluate.py configs/qwen3_8b.yaml --mode trained --adapter <run>/adapter-final

Use training seeds that differ from the evaluation seeds (100+), so the agent is tested on
patients it did not train on.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from chemo_llm.agent import Agent, load_model  # noqa: E402
from chemo_llm.ppo import collect_rollout, ppo_update  # noqa: E402
from chemo_llm.utils import load_config, make_env, set_seed, start_run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the LLM dosing policy with PPO.")
    ap.add_argument("config", help="model config, e.g. configs/qwen3_8b.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ppo_cfg, train_cfg = cfg["ppo"], cfg["train"]

    set_seed(args.seed)
    env = make_env(cfg, args.seed)
    model, tokenizer = load_model(cfg)
    agent = Agent(model, tokenizer, train_cfg["prompt_variant"], cfg["doses"])
    optimizer = torch.optim.AdamW(agent.trainable_parameters(), lr=ppo_cfg["lr"])
    run_dir, writer = start_run(cfg, args.seed)
    print(f"run directory: {run_dir}")

    obs, _ = env.reset(seed=args.seed)
    for update in range(1, train_cfg["updates"] + 1):
        buffer, obs, episode_returns = collect_rollout(env, agent, obs, ppo_cfg)
        stats = ppo_update(agent, optimizer, buffer, ppo_cfg)

        # x-axis: total env steps so far
        step = update * ppo_cfg["rollout_steps"]
        for name, value in stats.items():
            writer.add_scalar(f"ppo/{name}", value, step)
        writer.add_scalar("rollout/mean_reward", sum(buffer.rewards) / len(buffer), step)
        writer.add_scalar("rollout/mean_dose",
                          sum(cfg["doses"][a] for a in buffer.actions) / len(buffer), step)
        for episode_return in episode_returns:
            writer.add_scalar("episode/return", episode_return, step)

        print(f"update {update}/{train_cfg['updates']}  "
              f"policy_loss {stats['policy_loss']:.4f}  value_loss {stats['value_loss']:.4f}  "
              f"approx_kl {stats['approx_kl']:.4f}  episodes finished {len(episode_returns)}")

        if update % train_cfg["save_every"] == 0:
            agent.save(run_dir / f"adapter-{update:04d}")

    agent.save(run_dir / "adapter-final")
    writer.close()


if __name__ == "__main__":
    main()
