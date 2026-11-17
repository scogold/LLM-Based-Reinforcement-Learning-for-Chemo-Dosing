import pytest
import torch

from chemo_llm import prompts
from chemo_llm.agent import Agent
from conftest import LABEL_IDS

DOSES = [0.0, 0.25, 0.5, 0.75, 1.0]
OBS = [[0.2, 0.005, 0.1], [0.5, 0.1, 0.3], [0.9, 0.0, 0.0]]


@pytest.fixture
def agent(stub_model_and_tokenizer):
    model, tokenizer = stub_model_and_tokenizer
    return Agent(model, tokenizer, "base", DOSES)


def manual_log_probs(agent, obs):
    """Recompute the policy by hand: run the stub layers, keep label logits, log-softmax."""
    x = torch.tensor(obs)
    logits = agent.model.frozen_head(torch.tanh(agent.model.lora(x)))
    return logits[:, LABEL_IDS].log_softmax(-1)


def test_action_tokens_are_the_label_tokens(agent):
    assert agent.action_token_ids.tolist() == LABEL_IDS


def test_dose_count_must_match_labels(stub_model_and_tokenizer):
    model, tokenizer = stub_model_and_tokenizer
    with pytest.raises(AssertionError):
        Agent(model, tokenizer, "base", [0.0, 1.0])


def test_doses_must_match_the_numbers_in_the_prompt(stub_model_and_tokenizer):
    """The prompt shows prompts.DOSE_LEVELS, so the env must receive the same values."""
    model, tokenizer = stub_model_and_tokenizer
    with pytest.raises(AssertionError, match="doses must equal prompts.DOSE_LEVELS"):
        Agent(model, tokenizer, "base", [0.0, 0.1, 0.5, 0.75, 1.0])


def test_doses_as_list_or_tuple_are_accepted(stub_model_and_tokenizer):
    model, tokenizer = stub_model_and_tokenizer
    Agent(model, tokenizer, "base", list(prompts.DOSE_LEVELS))
    Agent(model, tokenizer, "base", prompts.DOSE_LEVELS)


def test_multi_token_label_is_rejected(stub_model_and_tokenizer, monkeypatch):
    model, tokenizer = stub_model_and_tokenizer
    monkeypatch.setattr(tokenizer, "encode", lambda text, add_special_tokens=False: [1, 2])
    with pytest.raises(AssertionError):
        Agent(model, tokenizer, "base", DOSES)


def test_forward_does_not_add_special_tokens(agent):
    """The chat template already writes BOS for Gemma and DeepSeek, so the tokenizer must not add another."""
    agent._forward(OBS)
    assert agent.tokenizer.special_token_flags == [False]


def test_log_probs_match_manual_computation(agent):
    log_probs, _ = agent._forward(OBS)
    torch.testing.assert_close(log_probs, manual_log_probs(agent, OBS))
    assert log_probs.shape == (3, len(prompts.ACTION_LABELS))


def test_probabilities_sum_to_one_over_labels(agent):
    log_probs, _ = agent._forward(OBS)
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(3))


def test_other_tokens_cannot_change_the_policy(agent):
    """Huge logits on non-label tokens must not affect the action probabilities."""
    before, _ = agent._forward(OBS)
    with torch.no_grad():
        for token in [0, 1, 2, 8, 9]:
            agent.model.frozen_head.bias[token] = 1e4
    after, _ = agent._forward(OBS)
    torch.testing.assert_close(before, after)


def test_act_returns_dose_matching_index_and_log_prob(agent):
    torch.manual_seed(0)
    for _ in range(10):
        dose, index, log_prob, value = agent.act(OBS[0])
        assert dose == DOSES[index]
        assert log_prob == pytest.approx(manual_log_probs(agent, OBS)[0, index].item(), abs=1e-6)
        assert isinstance(value, float)


def test_greedy_picks_the_most_likely_label(agent):
    expected = manual_log_probs(agent, OBS).argmax(-1)
    for obs, best in zip(OBS, expected):
        dose, index, _, _ = agent.act(obs, greedy=True)
        assert index == best.item()
        assert dose == DOSES[best.item()]


def test_greedy_is_deterministic_and_sampling_is_seeded(agent):
    assert agent.act(OBS[1], greedy=True) == agent.act(OBS[1], greedy=True)
    torch.manual_seed(3)
    first = [agent.act(OBS[1])[1] for _ in range(20)]
    torch.manual_seed(3)
    second = [agent.act(OBS[1])[1] for _ in range(20)]
    assert first == second


def test_sampling_follows_the_policy(agent):
    """Frequencies of sampled actions are close to softmax probabilities."""
    torch.manual_seed(0)
    probs = manual_log_probs(agent, OBS)[0].exp()
    counts = torch.zeros(5)
    for _ in range(2000):
        counts[agent.act(OBS[0])[1]] += 1
    assert torch.allclose(counts / 2000, probs, atol=0.05)


def test_value_matches_act_value_and_head(agent):
    hidden = torch.tanh(agent.model.lora(torch.tensor(OBS[0])))
    expected = agent.value_head(hidden).item()
    assert agent.value(OBS[0]) == pytest.approx(expected, abs=1e-6)
    assert agent.act(OBS[0])[3] == pytest.approx(expected, abs=1e-6)


def test_evaluate_shapes_and_values(agent):
    actions = torch.tensor([0, 4, 2])
    chosen, entropy, values = agent.evaluate(OBS, actions)
    assert chosen.shape == entropy.shape == values.shape == (3,)

    log_probs = manual_log_probs(agent, OBS)
    torch.testing.assert_close(chosen, log_probs[torch.arange(3), actions])
    # entropy = -sum p log p, by hand
    torch.testing.assert_close(entropy, -(log_probs.exp() * log_probs).sum(-1))
    assert 0 < entropy.min() and entropy.max() <= torch.log(torch.tensor(5.0)) + 1e-6


def test_only_lora_and_value_head_get_gradients(agent):
    chosen, entropy, values = agent.evaluate(OBS, torch.tensor([0, 1, 2]))
    (chosen.sum() + values.sum()).backward()
    assert agent.model.lora.weight.grad is not None
    assert agent.value_head.weight.grad is not None
    assert agent.model.frozen_head.weight.grad is None
    assert agent.model.embedding.weight.grad is None


def test_trainable_parameters_are_lora_plus_value_head(agent):
    params = agent.trainable_parameters()
    ids = {id(p) for p in params}
    expected = {id(p) for p in agent.model.lora.parameters()} | {id(p) for p in agent.value_head.parameters()}
    assert ids == expected


def test_value_head_save_and_load_roundtrip(agent, tmp_path):
    agent.save(tmp_path)
    saved = agent.value_head.weight.detach().clone()
    with torch.no_grad():
        agent.value_head.weight.zero_()
    agent.load_value_head(tmp_path)
    torch.testing.assert_close(agent.value_head.weight, saved)


@pytest.mark.gpu
@pytest.mark.parametrize("config", ["qwen3_8b", "medgemma_4b", "deepseek_r1_7b"])
def test_real_model_labels_are_single_tokens_and_agent_builds(config):
    """Loads the real model in 4-bit; checks LoRA targets exist and A-E are single tokens."""
    from pathlib import Path

    from chemo_llm.agent import load_model
    from chemo_llm.utils import load_config

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cfg = load_config(Path(__file__).parent.parent / "configs" / f"{config}.yaml")
    model, tokenizer = load_model(cfg)  # raises if target_modules match no layer
    agent = Agent(model, tokenizer, cfg["train"]["prompt_variant"], cfg["doses"])
    obs = [0.2, 0.005, 0.0]
    dose, index, log_prob, value = agent.act(obs, greedy=True)
    assert dose == cfg["doses"][index]


def test_value_head_starts_at_zero(stub_model_and_tokenizer):
    model, tokenizer = stub_model_and_tokenizer
    agent = Agent(model, tokenizer, "base", [0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.count_nonzero(agent.value_head.weight) == 0
    assert torch.count_nonzero(agent.value_head.bias) == 0
    assert agent.value([0.2, 0.5, 0.1]) == 0.0
