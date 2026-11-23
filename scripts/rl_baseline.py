"""Train a conventional (non-LLM) RL baseline on AhnChemoEnv and score it like every other method.

    python scripts/rl_baseline.py configs/qwen3_8b.yaml --algo dqn --steps 200000

What it does
  1. Builds the same env as the LLM runs (chemo_llm.utils.make_env: setting 2, 600 steps,
     6-step history, training-only death penalty from base.yaml).
  2. Gives the agent the same 5 dose levels (cfg["doses"]) as a Discrete(5) action.
  3. Trains DQN or PPO from Stable-Baselines3 (allowed for this baseline; claude.md only bans
     it for the LLM agent). Training patients come from a generator seeded with --seed, so
     they are not the evaluation seeds 100-199.
  4. Evaluates the greedy policy with evaluation.evaluate_policy on --seeds (default 100-199),
     using the TRUE simulator state and the env's own reward, exactly like evaluate.py.
  5. Writes <out>/<label>.json (raw logs for plot.py) and updates <out>/summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

from chemo_llm.evaluation import aggregate, evaluate_policy, save_episode_logs  # noqa: E402
from chemo_llm.true_state import ahn_true_state  # noqa: E402
from chemo_llm.utils import load_config, make_env, set_seed  # noqa: E402
from evaluate import format_table  # noqa: E402  (scripts/ is on sys.path when run as a script)

REWARD_SCALE = 0.01  # same as ppo.reward_scale: returns of +-100 become +-1


def make_features(history_len, scale):
    """obs (k rows of [tumor, immune, drug], k <= history_len + 1) -> flat, scaled vector.
    Early in an episode there are fewer rows; the oldest row is repeated to fill the gap."""
    def features(obs):
        rows = np.nan_to_num(np.asarray(obs, dtype=np.float64))
        missing = history_len + 1 - len(rows)
        if missing > 0:
            rows = np.concatenate([np.repeat(rows[:1], missing, axis=0), rows])
        return np.clip(rows / scale, -10, 10).astype(np.float32).ravel()
    return features


def estimate_scale(cfg, episodes=10, seed=0):
    """Per-column max |value| seen under random dosing, used to scale observations to ~[-1, 1]."""
    env = make_env(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    top = np.full(3, 1e-6)
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            top = np.maximum(top, np.abs(np.nan_to_num(np.asarray(obs)[-1])))
            obs, _, terminated, truncated, _ = env.step(float(rng.choice(cfg["doses"])))
            done = terminated or truncated
    return top


class DiscreteDoseEnv(gym.Wrapper):
    """Discrete(5) action -> dose, flat scaled observation, and scaled reward for training."""

    def __init__(self, env, doses, features, n_features, reward_scale=1.0):
        super().__init__(env)
        self.doses, self.features, self.reward_scale = doses, features, reward_scale
        self.action_space = spaces.Discrete(len(doses))
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(n_features,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.features(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.doses[int(action)])
        return self.features(obs), reward * self.reward_scale, terminated, truncated, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--algo", choices=("dqn", "ppo"), default="dqn")
    ap.add_argument("--steps", type=int, default=200_000, help="training env steps")
    ap.add_argument("--seed", type=int, default=0, help="training seed (keep outside --seeds)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 200)),
                    help="evaluation seeds")
    ap.add_argument("--death-penalty", type=float, default=None,
                    help="override env.death_penalty for training (0 = env's own reward)")
    ap.add_argument("--max-t", type=int, default=None, help="override episode length")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--label", default=None, help="name used for the output file")
    args = ap.parse_args(argv)

    from stable_baselines3 import DQN, PPO

    cfg = load_config(args.config)
    if args.max_t:
        cfg["env"]["max_t"] = args.max_t
    if args.death_penalty is not None:
        cfg["env"]["death_penalty"] = args.death_penalty
    doses = cfg["doses"]
    history_len = cfg["env"]["history"]
    set_seed(args.seed)

    scale = estimate_scale(cfg)
    features = make_features(history_len, scale)
    n_features = 3 * (history_len + 1)
    print(f"observation scale (tumor, immune, drug): {scale}")

    train_env = DiscreteDoseEnv(make_env(cfg, seed=args.seed, training=True), doses, features,
                                n_features, REWARD_SCALE)
    if args.algo == "dqn":
        model = DQN("MlpPolicy", train_env, learning_rate=1e-4, buffer_size=100_000,
                    learning_starts=5_000, batch_size=64, gamma=0.99, train_freq=4,
                    target_update_interval=2_000, exploration_fraction=0.3,
                    exploration_final_eps=0.05, policy_kwargs={"net_arch": [128, 128]},
                    seed=args.seed, device="cpu", verbose=1)
    else:
        model = PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                    policy_kwargs={"net_arch": [128, 128]}, seed=args.seed, device="cpu",
                    verbose=1)
    model.learn(total_timesteps=args.steps)

    # Evaluation: the plain env (env's own reward, no death penalty), same as evaluate.py.
    def act(obs):
        action, _ = model.predict(features(obs), deterministic=True)
        index = int(action)
        return doses[index], index, None, 0.0

    label = args.label or args.algo
    out = Path(args.out or f"runs/{label}-s{args.seeds[0]}-{args.seeds[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    eval_env = make_env(cfg, seed=0)
    metrics, logs = evaluate_policy(eval_env, act, args.seeds, ahn_true_state)
    save_episode_logs(out / f"{label}.json", logs)
    model.save(str(out / f"{label}_model"))
    (out / f"{label}_train_config.json").write_text(json.dumps({
        "algo": args.algo, "steps": args.steps, "seed": args.seed, "eval_seeds": [args.seeds[0], args.seeds[-1]],
        "death_penalty": cfg["env"]["death_penalty"], "max_t": cfg["env"]["max_t"],
        "obs_scale": scale.tolist()}, indent=2))

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[label] = aggregate(metrics)
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n" + format_table({label: summary[label]}))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()