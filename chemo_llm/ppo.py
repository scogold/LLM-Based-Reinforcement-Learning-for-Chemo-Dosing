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
      lr: 1.0e-5                 # used by scripts/train.py to build the optimizer

Needs from utils.py (the env builder):
    A single (not vectorised) Gymnasium-style AhnChemoEnv:
    reset() -> (obs, info) and step(action) -> (obs, reward, terminated, truncated, info),
    wrapped in gymnasium.wrappers.RecordEpisodeStatistics so that info["episode"]["r"] holds
    the episode return when an episode ends.

Needs from agent.py: Agent.act, Agent.value and Agent.evaluate.

Used by scripts/train.py, roughly:

    obs, _ = env.reset(seed=seed)
    for update in range(num_updates):
        buffer, obs, episode_returns = collect_rollout(env, agent, obs, ppo_cfg)
        stats = ppo_update(agent, optimizer, buffer, ppo_cfg)
        # log stats and episode_returns to TensorBoard, save checkpoints
"""

import torch


class RolloutBuffer:
    """One rollout of single-env transitions, in order."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, obs, action, log_prob, value, reward, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
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
            delta = self.rewards[t] + gamma * values[t + 1] * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
        self.advantages = torch.tensor(advantages)
        self.returns = self.advantages + torch.tensor(self.values)


def collect_rollout(env, agent, obs, cfg):
    """Run the agent for cfg["rollout_steps"] steps starting from obs.

    Returns (buffer, next_obs, episode_returns), where episode_returns lists the total reward of
    each episode that finished during the rollout.
    """
    buffer = RolloutBuffer()
    episode_returns = []

    for _ in range(cfg["rollout_steps"]):
        env_action, action, log_prob, value = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(env_action)

        # A time-limit cut is not a real end of treatment, so bootstrap from the next state.
        if truncated and not terminated:
            reward += cfg["gamma"] * agent.value(next_obs)

        buffer.add(obs, action, log_prob, value, reward, float(terminated or truncated))

        if terminated or truncated:
            episode_returns.append(float(info["episode"]["r"]))
            next_obs, _ = env.reset()
        obs = next_obs

    buffer.compute_advantages(agent.value(obs), cfg["gamma"], cfg["gae_lambda"])
    return buffer, obs, episode_returns


def ppo_update(agent, optimizer, buffer, cfg):
    """Run the PPO epochs over one rollout. Returns a dict of mean losses for logging."""
    actions = torch.tensor(buffer.actions)
    old_log_probs = torch.tensor(buffer.log_probs)
    returns = buffer.returns
    advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    params = agent.trainable_parameters()
    device = params[0].device
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

    for _ in range(cfg["epochs"]):
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

            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            optimizer.step()

    return {name: sum(values) / len(values) for name, values in stats.items()}
