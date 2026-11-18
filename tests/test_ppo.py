import numpy as np
import pytest
import torch

from chemo_llm.agent import Agent
from chemo_llm.ppo import RolloutBuffer, collect_rollout, ppo_update

DOSES = [0.0, 0.25, 0.5, 0.75, 1.0]
OBS = [[0.2, 0.005, 0.1], [0.5, 0.1, 0.3], [0.9, 0.0, 0.0], [0.1, 0.4, 0.8]]


def make_buffer(rewards, values, dones, last_value, gamma=0.9, lam=0.8):
    buffer = RolloutBuffer()
    for r, v, d in zip(rewards, values, dones):
        buffer.add(None, 0, 0.0, v, r, d)
    buffer.compute_advantages(last_value, gamma, lam)
    return buffer


# ---- GAE ----

def test_gae_two_steps_no_done():
    # gamma=0.9, lambda=0.8, rewards [1,1], values [0.5,0.4], last value 0.3
    # delta1 = 1 + 0.9*0.3 - 0.4 = 0.87              -> A1 = 0.87
    # delta0 = 1 + 0.9*0.4 - 0.5 = 0.86              -> A0 = 0.86 + 0.9*0.8*0.87 = 1.4864
    buffer = make_buffer([1, 1], [0.5, 0.4], [0.0, 0.0], 0.3)
    torch.testing.assert_close(buffer.advantages, torch.tensor([1.4864, 0.87]))
    # returns = advantage + value
    torch.testing.assert_close(buffer.returns, torch.tensor([1.9864, 1.27]))


def test_gae_done_at_last_step_cuts_bootstrap():
    # done at t=1: delta1 = 1 - 0.4 = 0.6 (last_value ignored), A1 = 0.6
    # delta0 = 0.86, A0 = 0.86 + 0.72*0.6 = 1.292
    buffer = make_buffer([1, 1], [0.5, 0.4], [0.0, 1.0], last_value=100.0)
    torch.testing.assert_close(buffer.advantages, torch.tensor([1.292, 0.6]))


def test_gae_done_in_the_middle_stops_propagation():
    # done at t=0: delta0 = 1 - 0.5 = 0.5 and A0 = 0.5, nothing from step 1 leaks back.
    # step 1 is unaffected: delta1 = 0.87
    buffer = make_buffer([1, 1], [0.5, 0.4], [1.0, 0.0], 0.3)
    torch.testing.assert_close(buffer.advantages, torch.tensor([0.5, 0.87]))


def test_gae_lambda_zero_is_one_step_td():
    buffer = make_buffer([1, 1], [0.5, 0.4], [0.0, 0.0], 0.3, lam=0.0)
    torch.testing.assert_close(buffer.advantages, torch.tensor([0.86, 0.87]))


# ---- collect_rollout ----

class ToyEnv:
    """obs is a step counter. Every step gives reward 1. The episode ends at step `length`,
    by termination or by truncation."""

    def __init__(self, length, truncate):
        self.length, self.truncate, self.t = length, truncate, 0

    def reset(self):
        self.t = 0
        return 0, {}

    def step(self, action):
        self.t += 1
        end = self.t == self.length
        info = {"episode": {"r": np.array([float(self.t)]), "l": np.array([self.t])}} if end else {}
        return self.t, 1.0, end and not self.truncate, end and self.truncate, info


class FakeAgent:
    """Always acts with index 1, log-prob -0.5, and values every state as 10."""

    def act(self, obs):
        return DOSES[1], 1, -0.5, 10.0

    def value(self, obs):
        return 10.0


CFG = {"rollout_steps": 3, "gamma": 0.9, "gae_lambda": 0.8, "reward_scale": 1.0}


def test_truncation_adds_discounted_bootstrap_to_reward():
    buffer, next_obs, episodes = collect_rollout([ToyEnv(2, truncate=True)], FakeAgent(), [0], CFG)
    # step 2 is truncated: env reward stays 1; the bootstrap 0.9 * V(next) = 9 is stored apart.
    # step 3 ends the env's run: done=1 and bootstrap 9 too.
    assert buffer.rewards == [1.0, 1.0, 1.0]
    assert buffer.bootstraps == [0.0, 9.0, 9.0]
    assert buffer.dones == [0.0, 1.0, 1.0]
    assert [e["return"] for e in episodes] == [2.0]
    assert next_obs == [1]  # a new episode started after the cut, one step into it


def test_episode_died_is_true_on_termination_and_false_on_truncation():
    _, _, dead = collect_rollout([ToyEnv(2, truncate=False)], FakeAgent(), [0], CFG)
    _, _, cut = collect_rollout([ToyEnv(2, truncate=True)], FakeAgent(), [0], CFG)
    assert dead == [{"return": 2.0, "length": 2, "died": True}]
    assert cut == [{"return": 2.0, "length": 2, "died": False}]


def test_episode_length_is_reported_for_each_finished_episode():
    # 3 steps, episode length 1 -> three episodes of length 1
    _, _, episodes = collect_rollout([ToyEnv(1, truncate=False)], FakeAgent(), [0], CFG)
    assert [e["length"] for e in episodes] == [1, 1, 1]


def test_termination_does_not_bootstrap():
    buffer, _, episodes = collect_rollout([ToyEnv(2, truncate=False)], FakeAgent(), [0], CFG)
    # death at step 2: no bootstrap; step 3 is the run end (not a death): bootstrap 9
    assert buffer.rewards == [1.0, 1.0, 1.0]
    assert buffer.bootstraps == [0.0, 0.0, 9.0]
    assert buffer.dones == [0.0, 1.0, 1.0]
    assert [e["return"] for e in episodes] == [2.0]


def test_rollout_records_transitions():
    buffer, next_obs, episodes = collect_rollout([ToyEnv(100, truncate=False)], FakeAgent(), [0], CFG)
    assert len(buffer) == 3
    assert buffer.observations == [0, 1, 2]  # the obs the action was taken in
    assert buffer.actions == [1, 1, 1]
    assert buffer.log_probs == [-0.5] * 3
    assert buffer.dones == [0.0, 0.0, 1.0]       # the run end is a done
    assert buffer.bootstraps == [0.0, 0.0, 9.0]  # 0.9 * V = 9
    assert next_obs == [3] and episodes == []
    assert buffer.advantages.shape == buffer.returns.shape == (3,)


class RewardEnv(ToyEnv):
    """Like ToyEnv but with a chosen reward per step (the last one is used for later steps)."""

    def __init__(self, length, truncate, rewards):
        super().__init__(length, truncate)
        self.rewards = rewards

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        return obs, self.rewards[min(self.t, len(self.rewards)) - 1], terminated, truncated, info


def scaled_cfg(scale):
    return {**CFG, "reward_scale": scale}


def test_reward_scale_applies_to_stored_rewards():
    env = RewardEnv(100, False, [10.0, -20.0, 30.0])
    buffer, _, _ = collect_rollout([env], FakeAgent(), [0], scaled_cfg(0.1))
    assert buffer.rewards == pytest.approx([1.0, -2.0, 3.0])


def test_truncation_bootstrap_is_not_scaled_and_not_logged_as_reward():
    # reward 50 at scale 0.01 -> 0.5; V(next) = 10, gamma = 0.9 -> bootstrap 0.9*10 = 9.0
    # rewards hold only the scaled env reward; the bootstrap is unscaled and stored apart
    env = RewardEnv(2, True, [50.0, 50.0])
    buffer, _, _ = collect_rollout([env], FakeAgent(), [0], scaled_cfg(0.01))
    assert buffer.rewards == pytest.approx([0.5, 0.5, 0.5])
    assert buffer.bootstraps == pytest.approx([0.0, 9.0, 9.0])
    assert buffer.dones == [0.0, 1.0, 1.0]


def test_advantages_include_truncation_bootstrap():
    # Splitting the bootstrap out must not change GAE: same result as folding it into the reward.
    env = RewardEnv(2, True, [50.0, 50.0])
    buffer, _, _ = collect_rollout([env], FakeAgent(), [0], scaled_cfg(0.01))
    folded = make_buffer([0.5, 9.5, 9.5], [10.0, 10.0, 10.0], [0.0, 1.0, 1.0], 0.0)
    torch.testing.assert_close(buffer.advantages, folded.advantages)
    torch.testing.assert_close(buffer.returns, folded.returns)


def test_death_reward_is_scaled_without_bootstrap():
    # termination with reward -100 at scale 0.01 -> -1, no bootstrap
    env = RewardEnv(2, False, [1.0, -100.0])
    buffer, _, _ = collect_rollout([env], FakeAgent(), [0], scaled_cfg(0.01))
    assert buffer.rewards == pytest.approx([0.01, -1.0, 0.01])
    assert buffer.dones == [0.0, 1.0, 1.0]
    assert buffer.bootstraps == pytest.approx([0.0, 0.0, 9.0])  # none after death; run end bootstraps


def test_episode_returns_stay_unscaled():
    # ToyEnv reports info["episode"]["r"] = 2.0 for a 2-step episode, whatever the scale
    _, _, episodes = collect_rollout([ToyEnv(2, truncate=False)], FakeAgent(), [0], scaled_cfg(0.01))
    assert [e["return"] for e in episodes] == [2.0]


# ---- several envs ----

class ValueIsObsAgent(FakeAgent):
    """Like FakeAgent, but V(obs) = obs, so a bootstrap shows which obs it was computed from."""

    def act(self, obs):
        return DOSES[1], 1, -0.5, float(obs)

    def value(self, obs):
        return float(obs)


class OffsetEnv(ToyEnv):
    """ToyEnv whose observations are shifted by `offset`, to tell the envs apart."""

    def __init__(self, length, truncate, offset):
        super().__init__(length, truncate)
        self.offset = offset

    def reset(self):
        return self.offset, {}

    def step(self, action):
        obs, *rest = super().step(action)
        return (obs + self.offset, *rest)


def cfg_steps(n):
    return {**CFG, "rollout_steps": n}


def test_rollout_takes_consecutive_runs_from_each_env_in_order():
    envs = [OffsetEnv(100, False, 0), OffsetEnv(100, False, 1000)]
    buffer, next_obs, _ = collect_rollout(envs, ValueIsObsAgent(), [0, 1000], cfg_steps(4))
    # 4 steps / 2 envs = 2 steps each: env 0 saw obs 0, 1 and env 1 saw 1000, 1001
    assert buffer.observations == [0, 1, 1000, 1001]
    assert buffer.dones == [0.0, 1.0, 0.0, 1.0]  # each run ends with done
    assert next_obs == [2, 1002]                  # carried on per env


def test_run_end_bootstraps_from_next_obs():
    envs = [OffsetEnv(100, False, 0), OffsetEnv(100, False, 1000)]
    buffer, _, _ = collect_rollout(envs, ValueIsObsAgent(), [0, 1000], cfg_steps(4))
    # V(obs) = obs, gamma 0.9: env 0 ends with next obs 2 -> 1.8; env 1 with 1002 -> 901.8
    assert buffer.bootstraps == pytest.approx([0.0, 1.8, 0.0, 901.8])


def test_death_at_run_end_has_no_bootstrap():
    buffer, _, _ = collect_rollout([ToyEnv(2, False)], ValueIsObsAgent(), [0], cfg_steps(2))
    assert buffer.dones == [0.0, 1.0] and buffer.bootstraps == [0.0, 0.0]


def test_truncation_at_run_end_bootstraps_from_pre_reset_obs():
    buffer, next_obs, _ = collect_rollout([ToyEnv(2, True)], ValueIsObsAgent(), [0], cfg_steps(2))
    # truncated at step 2 (obs 2): bootstrap 0.9 * 2 = 1.8, not V of the reset obs (0)
    assert buffer.bootstraps == pytest.approx([0.0, 1.8])
    assert buffer.dones == [0.0, 1.0]
    assert next_obs == [0]  # but the env continues from the reset obs


def test_advantages_do_not_leak_across_env_boundaries():
    envs = [RewardEnv(100, False, [2.0]), RewardEnv(100, False, [100.0])]
    buffer, _, _ = collect_rollout(envs, FakeAgent(), [0, 0], cfg_steps(4))
    assert buffer.rewards == [2.0, 2.0, 100.0, 100.0]
    # V = 10 everywhere, gamma 0.9, lambda 0.8. Env 0: bootstrap 9 at its last step.
    # delta = 2 + 9 - 10 = 1 at step 1; delta0 = 2 + 0.9*10 - 10 = 1
    # A1 = 1, A0 = 1 + 0.9*0.8*1 = 1.72
    torch.testing.assert_close(buffer.advantages[:2], torch.tensor([1.72, 1.0]))
    # Env 1: deltas are 100 + 9 - 10 = 99 at both steps; A1 = 99, A0 = 99 + 0.72*99
    torch.testing.assert_close(buffer.advantages[2:], torch.tensor([99 * 1.72, 99.0]))


def test_rollout_steps_must_divide_evenly_over_envs():
    envs = [ToyEnv(100, False), ToyEnv(100, False)]
    with pytest.raises(AssertionError):
        collect_rollout(envs, FakeAgent(), [0, 0], cfg_steps(3))


# ---- PPO update ----

@pytest.fixture
def agent(stub_model_and_tokenizer):
    model, tokenizer = stub_model_and_tokenizer
    return Agent(model, tokenizer, "base", DOSES)


class RecordGrads(torch.optim.Optimizer):
    """Does not update anything; stores a copy of the gradients at each step()."""

    def __init__(self, params):
        super().__init__(params, {})
        self.recorded = []

    def step(self):
        self.recorded.append([p.grad.clone() for g in self.param_groups for p in g["params"]])


def ppo_cfg(**overrides):
    cfg = dict(epochs=1, minibatch_size=4, micro_batch_size=4, clip_range=0.2, vf_coef=0.5,
               ent_coef=0.01, max_grad_norm=1e6, target_kl=1e9)
    return {**cfg, **overrides}


def fixed_buffer(agent, actions, advantages, log_prob_shift=None):
    """A buffer over OBS with old log-probs from the current policy (plus an optional shift)."""
    with torch.no_grad():
        old, _, values = agent.evaluate(OBS, torch.tensor(actions))
    if log_prob_shift is not None:
        old = old + torch.tensor(log_prob_shift)
    buffer = RolloutBuffer()
    for obs, a, lp, v in zip(OBS, actions, old, values):
        buffer.add(obs, a, lp.item(), v.item(), 0.0, 0.0)
    buffer.advantages = torch.tensor(advantages)
    buffer.returns = values + torch.tensor([1.0, -1.0, 2.0, 0.5])  # arbitrary value targets
    return buffer


def test_gradient_accumulation_equals_full_minibatch(agent):
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -0.5, 2.0, 0.3], log_prob_shift=[0.1, -0.2, 0.3, 0.0])
    grads = {}
    for micro in (4, 1, 2):
        opt = RecordGrads(agent.trainable_parameters())
        ppo_update(agent, opt, buffer, ppo_cfg(micro_batch_size=micro))
        grads[micro] = opt.recorded[0]
    for micro in (1, 2):
        for full, accumulated in zip(grads[4], grads[micro]):
            torch.testing.assert_close(accumulated, full)


def test_update_increases_prob_of_good_actions_and_decreases_bad(agent):
    actions = [2, 2, 0, 0]
    buffer = fixed_buffer(agent, actions, [1.0, 1.0, -1.0, -1.0])
    before = agent.evaluate(OBS, torch.tensor(actions))[0].detach()
    opt = torch.optim.SGD(agent.trainable_parameters(), lr=0.1)
    stats = ppo_update(agent, opt, buffer, ppo_cfg(vf_coef=0.0, ent_coef=0.0))
    after = agent.evaluate(OBS, torch.tensor(actions))[0].detach()
    assert (after[:2] > before[:2]).all()
    assert (after[2:] < before[2:]).all()
    assert stats["clip_fraction"] == 0.0
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-6)  # ratio is 1 on the first step


def test_clipped_samples_give_no_policy_gradient_and_known_loss(agent):
    # Advantages [1,1,-1,-1] normalise to +-a with a = 1/std = 1/sqrt(4/3) = 0.8660.
    # Old log-probs are shifted so ratio = e^1 = 2.72 for the positive-advantage samples
    # and e^-1 = 0.37 for the negative ones. Both lie outside [0.8, 1.2] in the direction
    # where PPO clips, so min(ratio*A, clip(ratio)*A) = clip(ratio)*A: 1.2a, 1.2a, -0.8a, -0.8a.
    # policy_loss = -mean = -(0.2a) = -0.1732, and there is no gradient at all.
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, 1.0, -1.0, -1.0],
                          log_prob_shift=[-1.0, -1.0, 1.0, 1.0])
    opt = RecordGrads(agent.trainable_parameters())
    stats = ppo_update(agent, opt, buffer, ppo_cfg(vf_coef=0.0, ent_coef=0.0))
    assert stats["clip_fraction"] == 1.0
    assert stats["policy_loss"] == pytest.approx(-0.2 * 0.8660254, abs=1e-5)
    for g in opt.recorded[0]:
        assert torch.count_nonzero(g) == 0


def test_unclipped_loss_equals_negative_mean_ratio_times_advantage(agent):
    # Shifts of 0.1 give ratios exp(+-0.1) ~ 1.105 / 0.905, inside [0.8, 1.2], so no clipping.
    # Advantages [1,1,-1,-1] normalise to +-0.8660; loss = -mean(ratio * A).
    shift = [-0.1, -0.1, 0.1, 0.1]   # new - old = -shift, so ratio = exp(-shift)
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, 1.0, -1.0, -1.0], log_prob_shift=shift)
    a = 0.8660254
    ratios = torch.exp(-torch.tensor(shift))
    expected = -(ratios * torch.tensor([a, a, -a, -a])).mean().item()
    stats = ppo_update(agent, RecordGrads(agent.trainable_parameters()), buffer,
                       ppo_cfg(vf_coef=0.0, ent_coef=0.0))
    assert stats["policy_loss"] == pytest.approx(expected, abs=1e-5)
    assert stats["clip_fraction"] == 0.0


def test_value_loss_is_half_mse(agent):
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 1.0, -1.0])
    # value targets differ from the current values by [1, -1, 2, 0.5] (see fixed_buffer)
    # value_loss = 0.5 * mean([1, 1, 4, 0.25]) = 0.5 * 1.5625 = 0.78125
    stats = ppo_update(agent, RecordGrads(agent.trainable_parameters()), buffer, ppo_cfg())
    assert stats["value_loss"] == pytest.approx(0.78125, abs=1e-5)


def test_update_only_changes_lora_and_value_head(agent):
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 0.5, -0.5])
    frozen = list(agent.model.frozen_head.parameters()) + list(agent.model.embedding.parameters())
    frozen_before = [p.detach().clone() for p in frozen]
    lora_before = agent.model.lora.weight.detach().clone()
    head_before = agent.value_head.weight.detach().clone()
    opt = torch.optim.SGD(agent.trainable_parameters(), lr=0.1)
    ppo_update(agent, opt, buffer, ppo_cfg())
    for before, after in zip(frozen_before, frozen):
        torch.testing.assert_close(before, after.detach())
    assert not torch.equal(lora_before, agent.model.lora.weight.detach())
    assert not torch.equal(head_before, agent.value_head.weight.detach())


# ---- extra PPO statistics ----

def lora_grads_with_value_error(agent, value_error):
    """LoRA and value-head gradients seen at the optimizer step, for a given value target error."""
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -0.5, 2.0, 0.3], log_prob_shift=[0.1, -0.2, 0.3, 0.0])
    buffer.returns = torch.tensor(buffer.values) + value_error
    opt = RecordGrads(agent.trainable_parameters())
    stats = ppo_update(agent, opt, buffer, ppo_cfg(max_grad_norm=1.0))
    n = len(agent.lora_parameters())
    return opt.recorded[0][:n], opt.recorded[0][n:], stats


def test_value_head_gradient_is_clipped_separately_from_lora(agent):
    # The value head starts at zero, so the value loss sends no gradient into the LoRA weights.
    # With separate clipping the LoRA gradient is therefore the same whatever the value error;
    # with one joint clip, a huge value-head gradient would shrink it.
    lora_small, _, _ = lora_grads_with_value_error(agent, 0.0)
    lora_huge, head_huge, stats = lora_grads_with_value_error(agent, 1000.0)
    for a, b in zip(lora_small, lora_huge):
        torch.testing.assert_close(a, b)
    assert stats["grad_norm_value_head"] > 100          # reported before clipping
    assert torch.sqrt(sum(g.pow(2).sum() for g in head_huge)) <= 1.0 + 1e-4  # clipped to max_grad_norm


def test_reported_lora_grad_norm_matches_recorded_gradients(agent):
    # max_grad_norm is huge, so nothing is clipped and the norm is that of the gradient itself.
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -0.5, 2.0, 0.3], log_prob_shift=[0.1, -0.2, 0.3, 0.0])
    opt = RecordGrads(agent.trainable_parameters())
    stats = ppo_update(agent, opt, buffer, ppo_cfg())
    n = len(agent.lora_parameters())
    expected = torch.sqrt(sum(g.pow(2).sum() for g in opt.recorded[0][:n]))
    assert stats["grad_norm_lora"] == pytest.approx(expected.item(), rel=1e-5)


def test_start_ratio_error_is_zero_for_consistent_old_log_probs(agent):
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 0.5, -0.5])
    stats = ppo_update(agent, RecordGrads(agent.trainable_parameters()), buffer, ppo_cfg())
    assert stats["start_ratio_error"] == pytest.approx(0.0, abs=1e-5)


def test_start_ratio_error_detects_perturbed_old_log_probs(agent):
    # Old log-prob of sample 0 is 0.5 too high: ratio = exp(-0.5) = 0.6065, error = 0.3935
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 0.5, -0.5], log_prob_shift=[0.5, 0, 0, 0])
    stats = ppo_update(agent, RecordGrads(agent.trainable_parameters()), buffer, ppo_cfg())
    assert stats["start_ratio_error"] == pytest.approx(1 - np.exp(-0.5), abs=1e-5)


def explained_variance(agent, returns, values):
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 0.5, -0.5])
    buffer.returns = torch.tensor(returns)
    buffer.values = list(values)
    stats = ppo_update(agent, RecordGrads(agent.trainable_parameters()), buffer, ppo_cfg())
    return stats["explained_variance"]


def test_explained_variance_is_one_for_perfect_values(agent):
    returns = [1.0, 2.0, 3.0, 4.0]
    assert explained_variance(agent, returns, returns) == pytest.approx(1.0, abs=1e-5)


def test_explained_variance_is_zero_for_constant_mean_values(agent):
    returns = [1.0, 2.0, 3.0, 4.0]
    assert explained_variance(agent, returns, [2.5] * 4) == pytest.approx(0.0, abs=1e-5)


# ---- value_only ----

def test_value_only_update_trains_value_head_but_not_lora(agent):
    torch.manual_seed(0)
    torch.nn.init.normal_(agent.value_head.weight)  # a zero head would send no gradient to LoRA anyway
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 0.5, -0.5])
    lora_before = agent.model.lora.weight.detach().clone()
    head_before = agent.value_head.weight.detach().clone()
    opt = torch.optim.SGD(agent.trainable_parameters(), lr=0.1)
    ppo_update(agent, opt, buffer, ppo_cfg(), value_only=True)
    assert torch.equal(lora_before, agent.model.lora.weight.detach())
    assert not torch.equal(head_before, agent.value_head.weight.detach())


# ---- target_kl ----

SHIFT = [0.1, -0.2, 0.3, 0.0]


def test_no_early_stop_with_large_target_kl(agent):
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 0.5, -0.5], log_prob_shift=SHIFT)
    opt = torch.optim.SGD(agent.trainable_parameters(), lr=0.01)
    stats = ppo_update(agent, opt, buffer, ppo_cfg(epochs=3, minibatch_size=3, micro_batch_size=3))
    assert stats["optimizer_steps"] == 3 * 2  # epochs * ceil(4 / 3)


def test_tiny_target_kl_stops_after_first_optimizer_step(agent):
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 0.5, -0.5], log_prob_shift=SHIFT)
    opt = torch.optim.SGD(agent.trainable_parameters(), lr=0.01)
    stats = ppo_update(agent, opt, buffer,
                       ppo_cfg(epochs=3, minibatch_size=2, micro_batch_size=2, target_kl=1e-12))
    assert stats["optimizer_steps"] == 1


def test_approx_kl_is_non_negative_estimator(agent):
    buffer = fixed_buffer(agent, [0, 1, 2, 3], [1.0, -1.0, 0.5, -0.5], log_prob_shift=SHIFT)
    stats = ppo_update(agent, RecordGrads(agent.trainable_parameters()), buffer, ppo_cfg())
    # ratio = exp(-shift); kl = mean((ratio - 1) - log ratio) = mean(exp(-s) - 1 + s) >= 0
    s = torch.tensor(SHIFT)
    expected = (torch.exp(-s) - 1 + s).mean().item()
    assert expected > 0
    assert stats["approx_kl"] == pytest.approx(expected, abs=1e-5)
