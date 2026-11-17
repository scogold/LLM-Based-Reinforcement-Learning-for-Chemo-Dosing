"""Shared stubs: a tiny fake model and tokenizer, so Agent and PPO run on CPU in milliseconds.

The stub "prompt" is just the observation itself (build_prompt is patched), and the stub
tokenizer turns a batch of observations into a float tensor. The model is
    hidden = tanh(lora(x));  logits = frozen_head(hidden)
so `lora` plays the trainable LoRA weights and `frozen_head` plays the frozen base model.
Vocabulary: token 0-2 are other tokens, labels "A".."E" are tokens 3..7, 8-9 are other tokens.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chemo_llm import prompts

VOCAB = 10
HIDDEN = 6
OBS_DIM = 3
LABEL_IDS = [3, 4, 5, 6, 7]  # tokens of "A".."E"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU and a real 4-8B model")


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(text) - ord("A") + 3]  # "A" -> 3, ..., "E" -> 7

    def __call__(self, observations, return_tensors="pt", padding=True):
        x = torch.tensor(np.array(observations), dtype=torch.float32)
        return SimpleNamespace(to=lambda device: {"x": x})


class StubModel(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB, HIDDEN)  # only used for its size
        self.lora = torch.nn.Linear(OBS_DIM, HIDDEN)         # trainable
        self.frozen_head = torch.nn.Linear(HIDDEN, VOCAB)    # frozen
        for p in list(self.embedding.parameters()) + list(self.frozen_head.parameters()):
            p.requires_grad = False

    def save_pretrained(self, directory):
        """Like PEFT's save_pretrained: write the trainable weights into the directory."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.lora.state_dict(), Path(directory) / "adapter_model.pt")

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, x, output_hidden_states=True, logits_to_keep=1, use_cache=False):
        hidden = torch.tanh(self.lora(x))
        logits = self.frozen_head(hidden)
        return SimpleNamespace(logits=logits[:, None, :], hidden_states=(hidden[:, None, :],))


@pytest.fixture
def stub_prompt(monkeypatch, request):
    """The stub prompt is the observation itself (real prompts are used in gpu tests)."""
    if request.node.get_closest_marker("gpu"):
        return
    monkeypatch.setattr(prompts, "build_prompt", lambda obs, variant, tokenizer: obs)


@pytest.fixture
def stub_model_and_tokenizer(stub_prompt):
    torch.manual_seed(0)
    return StubModel(), StubTokenizer()
