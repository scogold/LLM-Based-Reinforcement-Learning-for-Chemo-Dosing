"""Train the LLM dosing policy with PPO.

    python scripts/train.py configs/qwen3_8b.yaml --seed 0

Creates runs/<config>-seed<seed>-<time>/ containing:
    config.yaml          the resolved config, seed and git commit
    events.out.tfevents  TensorBoard logs (tensorboard --logdir runs); plot the learning curve
                         with scripts/plot.py --runs <run directory>
    adapter-0020/ ...    checkpoints every train.save_every updates
    adapter-final/       the final checkpoint
Each checkpoint holds the LoRA adapter and value head (what evaluate.py loads) plus
trainer.pt (optimizer state and update number, what --resume loads).

Evaluate a checkpoint with:
    python scripts/evaluate.py configs/qwen3_8b.yaml --mode trained --adapter <run>/adapter-final

Resume an interrupted run from its latest checkpoint, with the same config and --seed:
    python scripts/train.py configs/qwen3_8b.yaml --seed 0 --resume runs/<run>/adapter-0200
It continues in the same run directory. A resumed run is not bit-identical to an
uninterrupted one: it starts a fresh episode and reseeds the random generators.

Use training seeds that differ from the evaluation seeds (100+), so the agent is tested on
patients it did not train on.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from chemo_llm.agent import Agent, load_model  # noqa: E402
from chemo_llm.ppo import collect_rollout, ppo_update  # noqa: E402
from chemo_llm.utils import load_config, make_env, set_seed, start_run  # noqa: E402


def save_checkpoint(directory, agent, optimizer, update):
    agent.save(directory)
    torch.save({"update": update, "optimizer": optimizer.state_dict()}, directory / "trainer.pt")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the LLM dosing policy with PPO.")
    ap.add_argument("config", help="model config, e.g. configs/qwen3_8b.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from, e.g. runs/<run>/adapter-0200")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ppo_cfg, train_cfg = cfg["ppo"], cfg["train"]

    first_update = 1
    if args.resume:
        trainer_state = torch.load(Path(args.resume) / "trainer.pt")
        first_update = trainer_state["update"] + 1

    # On resume, shift the seed so the run doesn't replay the patients it started with.
    seed = args.seed + first_update - 1
    set_seed(seed)
    # One env per patient. seed * 100 + i gives each its own stream of patients. The first
    # reset also draws its seed from that stream (a random number below 2**31), so it is very
    # unlikely to be one of the evaluation seeds (100+).
    envs = [make_env(cfg, seed * 100 + i, training=True) for i in range(ppo_cfg["n_envs"])]
    model, tokenizer = load_model(cfg, adapter_dir=args.resume)
    agent = Agent(model, tokenizer, train_cfg["prompt_variant"], cfg["doses"])
    optimizer = torch.optim.AdamW([
        {"params": agent.lora_parameters(), "lr": ppo_cfg["lr"]},
        {"params": agent.value_head.parameters(), "lr": ppo_cfg["value_head_lr"]},
    ])

    if args.resume:
        agent.load_value_head(args.resume)
        optimizer.load_state_dict(trainer_state["optimizer"])
        run_dir = Path(args.resume).parent
        writer = SummaryWriter(run_dir)
    else:
        run_dir, writer = start_run(cfg, args.seed)
    print(f"run directory: {run_dir}, starting at update {first_update}")

    start_time = time.time()
    observations = [env.reset()[0] for env in envs]
    for update in range(first_update, train_cfg["updates"] + 1):
        update_start = time.time()
        value_only = update <= train_cfg["value_warmup_updates"]
        buffer, observations, episodes = collect_rollout(envs, agent, observations, ppo_cfg)
        stats = ppo_update(agent, optimizer, buffer, ppo_cfg, value_only=value_only)

        # x-axis: total env steps so far
        step = update * ppo_cfg["rollout_steps"]
        for name, value in stats.items():
            writer.add_scalar(f"ppo/{name}", value, step)
        writer.add_scalar("rollout/mean_reward", sum(buffer.rewards) / len(buffer), step)
        writer.add_scalar("rollout/mean_dose",
                          sum(cfg["doses"][a] for a in buffer.actions) / len(buffer), step)
        for episode in episodes:
            # Training return includes the DeathPenalty; evaluate.py reports the env's own reward.
            writer.add_scalar("episode/return", episode["return"], step)
            writer.add_scalar("episode/length", episode["length"], step)
            writer.add_scalar("episode/died", float(episode["died"]), step)
        # Compute cost for the paper: total training time is the sum of compute/minutes
        # (this also works across a --resume).
        writer.add_scalar("compute/minutes", (time.time() - update_start) / 60, step)
        if torch.cuda.is_available():
            writer.add_scalar("compute/peak_vram_gb", torch.cuda.max_memory_reserved() / 2**30, step)

        minutes_per_update = (time.time() - start_time) / 60 / (update - first_update + 1)
        hours_left = minutes_per_update * (train_cfg["updates"] - update) / 60
        print(f"update {update}/{train_cfg['updates']}  "
              f"policy_loss {stats['policy_loss']:.4f}  value_loss {stats['value_loss']:.4f}  "
              f"approx_kl {stats['approx_kl']:.4f}  explained_var {stats['explained_variance']:.2f}  "
              f"steps {stats['optimizer_steps']}  episodes finished {len(episodes)}  "
              f"{minutes_per_update:.1f} min/update, about {hours_left:.1f} h left"
              + ("  (value warm-up)" if value_only else ""))
        if stats["start_ratio_error"] > 0.01:
            print(f"  WARNING: PPO ratio was off by {stats['start_ratio_error']:.4f} before any update; "
                  "act() and evaluate() are computing different log-probs for the same input")

        if update % train_cfg["save_every"] == 0:
            save_checkpoint(run_dir / f"adapter-{update:04d}", agent, optimizer, update)

    save_checkpoint(run_dir / "adapter-final", agent, optimizer, train_cfg["updates"])
    writer.close()


if __name__ == "__main__":
    main()
