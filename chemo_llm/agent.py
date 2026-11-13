"""The LLM policy: load the model, then map observation -> prompt -> dose action, plus a value.

`load_model` loads the base LLM in 4-bit and attaches LoRA adapters. It reads the `model`
section of the config:

    model:
      name: Qwen/Qwen3-8B            # any Hugging Face causal LM
      load_in_4bit: true
      lora:
        r: 16
        alpha: 32
        dropout: 0.0                 # keep 0: dropout makes PPO's old/new log-prob ratio noisy
        target_modules: [q_proj, k_proj, v_proj, o_proj]

MedGemma 4B is a Gemma 3 vision-language model. Recent Transformers versions map it to
Gemma3ForConditionalGeneration through AutoModelForCausalLM; if loading fails, check that first.

`Agent` makes one forward pass over the prompt, which gives both outputs:
- the next-token logits. Only the logits of the K action-label tokens are kept, and a softmax
  over them gives the policy. This is the action constraint: the model can only pick a label.
- the final hidden state of the last prompt token, which a small linear value head maps to V(s).
The value head shares the LoRA-adapted backbone with the policy.

Action k means: the model writes prompts.ACTION_LABELS[k], and the env receives doses[k].

Needs from prompts.py:
    build_prompt(obs, variant: str, tokenizer) -> str
        The full prompt text for one env observation, with the chat template already applied,
        ending right where the model should write its action. For Qwen3 and DeepSeek R1 the
        template must skip the thinking block (e.g. `enable_thinking=False` for Qwen3),
        because the action is read from the very next token. End the prompt so the next token
        is the bare label (e.g. right after the assistant turn header, not after "Answer: "):
        Agent scores the token for "A", and after a space many tokenizers would write " A".
    ACTION_LABELS: list[str]
        The answer options the prompt offers, e.g. ["A", "B", "C", "D", "E"]. Each must be a
        single token for every model we use; Agent checks this when it is built.

Needs from the config (cfg["doses"], passed in by scripts/train.py and scripts/evaluate.py):
    doses: one env action per label, in the same order as ACTION_LABELS. The values must be
    what AhnChemoEnv's action space expects (dose levels for a continuous action space,
    or indices for a discrete one).

Used by:
- ppo.py calls `act` during rollouts and `evaluate` during the PPO update.
- scripts/train.py calls `load_model(cfg)`, builds the Agent, gets the optimizer parameters
  from `trainable_parameters`, and calls `save` for checkpoints.
- scripts/evaluate.py calls `load_model(cfg, adapter_dir=...)` and `load_value_head`, then
  `act(obs, greedy=True)`.
- evaluation.py (zero-shot LLM baseline) should use the same model with the adapter switched
  off: `with agent.model.disable_adapter(): ...`, so it has exactly the same weights and prompt.
"""

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from chemo_llm import prompts


def load_model(cfg, adapter_dir=None):
    """Return (model, tokenizer). Loads trainable LoRA weights from `adapter_dir` if given."""
    model_cfg = cfg["model"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    # Left padding keeps the last prompt token at position -1, where Agent reads the
    # next-token logits and the hidden state for the value head.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = None
    if model_cfg["load_in_4bit"]:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    # Freezes the base weights and turns on gradient checkpointing to fit in 16 GB.
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        lora = model_cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ))

    return model, tokenizer


class Agent:
    def __init__(self, model, tokenizer, prompt_variant, doses):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_variant = prompt_variant
        self.doses = doses
        assert len(doses) == len(prompts.ACTION_LABELS), "need one dose per action label"

        token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in prompts.ACTION_LABELS]
        assert all(len(ids) == 1 for ids in token_ids), f"action labels must be single tokens: {token_ids}"
        self.action_token_ids = torch.tensor([ids[0] for ids in token_ids], device=model.device)

        hidden_size = model.get_input_embeddings().embedding_dim
        self.value_head = torch.nn.Linear(hidden_size, 1).to(model.device)

    def _forward(self, observations):
        """Return (log_probs [B, K], values [B]) for a batch of observations."""
        texts = [prompts.build_prompt(obs, self.prompt_variant, self.tokenizer) for obs in observations]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        # logits_to_keep=1 skips the full [B, T, vocab] logits tensor, which is large.
        out = self.model(**inputs, output_hidden_states=True, logits_to_keep=1, use_cache=False)
        log_probs = out.logits[:, -1, self.action_token_ids].float().log_softmax(-1)
        values = self.value_head(out.hidden_states[-1][:, -1, :].float()).squeeze(-1)
        return log_probs, values

    @torch.no_grad()
    def act(self, obs, greedy=False):
        """Choose an action for one observation.

        Returns (env_action, action_index, log_prob, value); the last three are Python numbers.
        """
        self.model.eval()
        log_probs, values = self._forward([obs])
        if greedy:
            index = log_probs[0].argmax()
        else:
            index = Categorical(logits=log_probs[0]).sample()
        index = index.item()
        return self.doses[index], index, log_probs[0, index].item(), values[0].item()

    @torch.no_grad()
    def value(self, obs):
        self.model.eval()
        return self._forward([obs])[1].item()

    def evaluate(self, observations, action_indices):
        """Log-probs of the given actions, policy entropy and values, with gradients.

        action_indices is a LongTensor of shape [B]. Returns three tensors of shape [B].
        """
        self.model.train()
        log_probs, values = self._forward(observations)
        action_indices = action_indices.to(log_probs.device)
        chosen = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return chosen, entropy, values

    def trainable_parameters(self):
        lora = [p for p in self.model.parameters() if p.requires_grad]
        return lora + list(self.value_head.parameters())

    def save(self, directory):
        directory = Path(directory)
        self.model.save_pretrained(directory)  # LoRA adapter weights only
        torch.save(self.value_head.state_dict(), directory / "value_head.pt")

    def load_value_head(self, directory):
        """Load the value head saved next to an adapter. Load the adapter itself with
        load_model(cfg, adapter_dir=directory)."""
        state = torch.load(Path(directory) / "value_head.pt", map_location=self.model.device)
        self.value_head.load_state_dict(state)
