"""Hand-written PPO for the LLM agent: rollout collection, GAE and the clipped update.

Reads the `ppo` section of the config:

    ppo:
      rollout_steps: 64          # env steps collected per update, split evenly over the envs
      n_envs: 4                  # patients per rollout (used by scripts/train.py to build the envs)
      epochs: 4                  # passes over each rollout (at most; see target_kl)
      target_kl: 0.05            # stop the update early once approx_kl goes above this
      minibatch_size: 16         # samples per optimizer step
      micro_batch_size: 1        # samples per forward pass; keep 1 to match act() (see base.yaml)
      gamma: 0.99
      gae_lambda: 0.95
      clip_range: 0.2
      vf_coef: 0.5
      ent_coef: 0.01
      max_grad_norm: 1.0
      reward_scale: 0.01         # rewards are multiplied by this before GAE (death -100 -> -1)
      lr: 1.0e-5                 # LoRA learning rate, used by scripts/train.py
      value_head_lr: 1.0e-4      # value head learning rate, used by scripts/train.py

Needs from utils.py (the env builder):
    A list of single (not vectorised) Gymnasium-style AhnChemoEnvs, one per patient:
    reset() -> (obs, info) and step(action) -> (obs, reward, terminated, truncated, info),
    wrapped in gymnasium.wrappers.RecordEpisodeStatistics so that info["episode"] holds
    the episode return "r" and length "l" when an episode ends.

Needs from agent.py: Agent.act, Agent.value, Agent.evaluate, Agent.lora_parameters and
Agent.value_head (the two are gradient-clipped separately).

Used by scripts/train.py, roughly:

    observations = [env.reset()[0] for env in envs]
    for update in range(num_updates):
        buffer, observations, episodes = collect_rollout(envs, agent, observations, ppo_cfg)
        stats = ppo_update(agent, optimizer, buffer, ppo_cfg, value_only=update < warmup)
        # log stats and episodes to TensorBoard, save checkpoints
"""

import torch


class RolloutBuffer:
    """One rollout: a run of consecutive steps from each env, one env after another. Every run
    ends with done=1 and a bootstrap, so GAE never mixes steps from two envs."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []      # scaled env reward only, what gets logged
        self.bootstraps = []   # gamma * V(next obs) where a run is cut off, else 0; used by GAE
        self.dones = []

    def add(self, obs, action, log_prob, value, reward, done, bootstrap=0.0):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.bootstraps.append(bootstrap)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)

    def compute_advantages(self, last_value, gamma, gae_lambda):
        """Set self.advantages and self.returns with GAE. last_value is V of the obs after the rollout."""
        values = self.values + [last_value]
        advantages = [0.0] * len(self)
        gae = 0.0
        for t in reversed(range(len(self))):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.bootstraps[t] + gamma * values[t + 1] * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
        self.advantages = torch.tensor(advantages)
        self.returns = self.advantages + torch.tensor(self.values)


def collect_rollout(envs, agent, observations, cfg):
    """Collect cfg["rollout_steps"] steps: rollout_steps / len(envs) consecutive steps from
    each env, starting from observations[i]. Using several patients per rollout means one
    update doesn't only see one patient in one phase of their disease.

    Returns (buffer, next_observations, episodes). episodes has one dict per episode that
    finished during the rollout: "return" (total env reward, unscaled), "length" (steps) and
    "died" (True if the patient died rather than reaching the time limit).
    """
    assert cfg["rollout_steps"] % len(envs) == 0, "rollout_steps must divide evenly over the envs"
    steps_per_env = cfg["rollout_steps"] // len(envs)
    buffer = RolloutBuffer()
    episodes = []
    next_observations = []

    for env, obs in zip(envs, observations):
        for t in range(steps_per_env):
            env_action, action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            # Scale only what PPO learns from; info["episode"]["r"] keeps the env's own units.
            reward *= cfg["reward_scale"]

            # The time limit and the end of this env's run are not real ends of treatment,
            # so bootstrap from the next state. Death is a real end: no bootstrap.
            run_ends = t == steps_per_env - 1
            bootstrap = 0.0
            if (truncated or run_ends) and not terminated:
                bootstrap = cfg["gamma"] * agent.value(next_obs)

            buffer.add(obs, action, log_prob, value, reward,
                       float(terminated or truncated or run_ends), bootstrap)

            if terminated or truncated:
                episodes.append({
                    "return": info["episode"]["r"].item(),  # r and l are shape-(1,) arrays
                    "length": info["episode"]["l"].item(),
                    "died": bool(terminated),
                })
                next_obs, _ = env.reset()
            obs = next_obs
        next_observations.append(obs)

    # Every run ends with done=1 and its own bootstrap, so no value after the rollout is needed.
    buffer.compute_advantages(0.0, cfg["gamma"], cfg["gae_lambda"])
    return buffer, next_observations, episodes


def ppo_update(agent, optimizer, buffer, cfg, value_only=False):
    """Run the PPO epochs over one rollout. Returns a dict of statistics for logging:
    mean losses, entropy, approx_kl, clip_fraction, the gradient norms before clipping,
    explained_variance (how well the value head predicted the returns; 1 is perfect,
    0 is no better than a constant), start_ratio_error (see below) and optimizer_steps
    (fewer than epochs * minibatches when target_kl stopped the update early).

    value_only=True trains only the value head and leaves the policy (LoRA) unchanged. Used
    for the first few updates so the value head is a useful baseline before the policy moves.
    """
    actions = torch.tensor(buffer.actions)
    old_log_probs = torch.tensor(buffer.log_probs)
    returns = buffer.returns
    advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    # Clipped separately, so a large value-head gradient can't shrink the policy's update.
    lora_params = agent.lora_parameters()
    head_params = list(agent.value_head.parameters())
    device = head_params[0].device
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": [],
             "grad_norm_lora": [], "grad_norm_value_head": []}

    # Before the first optimizer step the policy hasn't changed, so the ratio new/old should be
    # 1. A larger error means act() and evaluate() compute different log-probs for the same input.
    start_ratios = []
    optimizer_steps = 0
    stop = False

    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(buffer))
        for start in range(0, len(buffer), cfg["minibatch_size"]):
            minibatch = order[start:start + cfg["minibatch_size"]]
            optimizer.zero_grad()
            minibatch_kl = []

            for micro in minibatch.split(cfg["micro_batch_size"]):
                observations = [buffer.observations[i] for i in micro]
                new_log_probs, entropy, values = agent.evaluate(observations, actions[micro])

                old = old_log_probs[micro].to(device)
                adv = advantages[micro].to(device)
                ret = returns[micro].to(device)

                log_ratio = new_log_probs - old
                ratio = log_ratio.exp()
                if epoch == 0 and start == 0:
                    start_ratios.append(ratio.detach())
                clipped = ratio.clamp(1 - cfg["clip_range"], 1 + cfg["clip_range"])
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = 0.5 * (values - ret).pow(2).mean()
                loss = cfg["vf_coef"] * value_loss
                if not value_only:
                    loss = loss + policy_loss - cfg["ent_coef"] * entropy.mean()

                # Scale so the accumulated gradient equals the minibatch mean.
                (loss * len(micro) / len(minibatch)).backward()

                with torch.no_grad():
                    # KL estimate that is never negative: mean of (ratio - 1) - log(ratio).
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    minibatch_kl.append(kl)
                    stats["approx_kl"].append(kl)
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["clip_fraction"].append(((ratio - 1).abs() > cfg["clip_range"]).float().mean().item())

            stats["grad_norm_lora"].append(
                torch.nn.utils.clip_grad_norm_(lora_params, cfg["max_grad_norm"]).item())
            stats["grad_norm_value_head"].append(
                torch.nn.utils.clip_grad_norm_(head_params, cfg["max_grad_norm"]).item())
            if value_only:
                # The value loss also reaches LoRA through the shared model; drop those
                # gradients so only the value head changes and the policy stays as it is.
                for p in lora_params:
                    p.grad = None
            optimizer.step()
            optimizer_steps += 1

            # The policy has moved far enough from the one that collected this data:
            # stop and collect a fresh rollout instead of taking more steps on stale data.
            if sum(minibatch_kl) / len(minibatch_kl) > cfg["target_kl"]:
                stop = True
                break
        if stop:
            break

    result = {name: sum(values) / len(values) for name, values in stats.items()}
    result["optimizer_steps"] = optimizer_steps
    result["start_ratio_error"] = (torch.cat(start_ratios) - 1).abs().max().item()
    old_values = torch.tensor(buffer.values)
    result["explained_variance"] = (1 - (returns - old_values).var() / (returns.var() + 1e-8)).item()
    return result
