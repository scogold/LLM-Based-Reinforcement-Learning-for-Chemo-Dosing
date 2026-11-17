"""Hand-written PPO for the LLM agent: rollout collection, GAE and the clipped update.

Reads the `ppo` section of the config:

    ppo:
      rollout_steps: 64          # env steps collected per update
      epochs: 4                  # passes over each rollout
      minibatch_size: 16         # samples per optimizer step
      micro_batch_size: 2        # samples per forward pass; gradients accumulate up to minibatch_size
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
    A single (not vectorised) Gymnasium-style AhnChemoEnv:
    reset() -> (obs, info) and step(action) -> (obs, reward, terminated, truncated, info),
    wrapped in gymnasium.wrappers.RecordEpisodeStatistics so that info["episode"] holds
    the episode return "r" and length "l" when an episode ends.

Needs from agent.py: Agent.act, Agent.value, Agent.evaluate, Agent.lora_parameters and
Agent.value_head (the two are gradient-clipped separately).

Used by scripts/train.py, roughly:

    obs, _ = env.reset(seed=seed)
    for update in range(num_updates):
        buffer, obs, episodes = collect_rollout(env, agent, obs, ppo_cfg)
        stats = ppo_update(agent, optimizer, buffer, ppo_cfg)
        # log stats and episodes to TensorBoard, save checkpoints
"""

import torch


class RolloutBuffer:
    """One rollout of single-env transitions, in order."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []      # scaled env reward only, what gets logged
        self.bootstraps = []   # gamma * V(next obs) after a time-limit cut, else 0; used by GAE
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


def collect_rollout(env, agent, obs, cfg):
    """Run the agent for cfg["rollout_steps"] steps starting from obs.

    Returns (buffer, next_obs, episodes). episodes has one dict per episode that finished during
    the rollout: "return" (total env reward, unscaled), "length" (steps) and "died" (True if
    the patient died rather than reaching the time limit).
    """
    buffer = RolloutBuffer()
    episodes = []

    for _ in range(cfg["rollout_steps"]):
        env_action, action, log_prob, value = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        # Scale only what PPO learns from; info["episode"]["r"] keeps the env's own units.
        reward *= cfg["reward_scale"]

        # A time-limit cut is not a real end of treatment, so bootstrap from the next state.
        bootstrap = 0.0
        if truncated and not terminated:
            bootstrap = cfg["gamma"] * agent.value(next_obs)

        buffer.add(obs, action, log_prob, value, reward, float(terminated or truncated), bootstrap)

        if terminated or truncated:
            episodes.append({
                "return": info["episode"]["r"].item(),  # r and l are shape-(1,) arrays
                "length": info["episode"]["l"].item(),
                "died": bool(terminated),
            })
            next_obs, _ = env.reset()
        obs = next_obs

    buffer.compute_advantages(agent.value(obs), cfg["gamma"], cfg["gae_lambda"])
    return buffer, obs, episodes


def ppo_update(agent, optimizer, buffer, cfg):
    """Run the PPO epochs over one rollout. Returns a dict of statistics for logging:
    mean losses, entropy, approx_kl, clip_fraction, the gradient norms before clipping,
    explained_variance (how well the value head predicted the returns; 1 is perfect,
    0 is no better than a constant) and start_ratio_error (see below)."""
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

    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(buffer))
        for start in range(0, len(buffer), cfg["minibatch_size"]):
            minibatch = order[start:start + cfg["minibatch_size"]]
            optimizer.zero_grad()

            for micro in minibatch.split(cfg["micro_batch_size"]):
                observations = [buffer.observations[i] for i in micro]
                new_log_probs, entropy, values = agent.evaluate(observations, actions[micro])

                old = old_log_probs[micro].to(device)
                adv = advantages[micro].to(device)
                ret = returns[micro].to(device)

                ratio = (new_log_probs - old).exp()
                if epoch == 0 and start == 0:
                    start_ratios.append(ratio.detach())
                clipped = ratio.clamp(1 - cfg["clip_range"], 1 + cfg["clip_range"])
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = 0.5 * (values - ret).pow(2).mean()
                loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy.mean()

                # Scale so the accumulated gradient equals the minibatch mean.
                (loss * len(micro) / len(minibatch)).backward()

                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["approx_kl"].append((old - new_log_probs).mean().item())
                    stats["clip_fraction"].append(((ratio - 1).abs() > cfg["clip_range"]).float().mean().item())

            stats["grad_norm_lora"].append(
                torch.nn.utils.clip_grad_norm_(lora_params, cfg["max_grad_norm"]).item())
            stats["grad_norm_value_head"].append(
                torch.nn.utils.clip_grad_norm_(head_params, cfg["max_grad_norm"]).item())
            optimizer.step()

    result = {name: sum(values) / len(values) for name, values in stats.items()}
    result["start_ratio_error"] = (torch.cat(start_ratios) - 1).abs().max().item()
    old_values = torch.tensor(buffer.values)
    result["explained_variance"] = (1 - (returns - old_values).var() / (returns.var() + 1e-8)).item()
    return result
