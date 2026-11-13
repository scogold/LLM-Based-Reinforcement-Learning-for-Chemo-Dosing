# LLM-Based Reinforcement Learning for Chemo Dosing

Research project by Scott and Tasnim. We fine-tune a small open-weight LLM with PPO so that it acts as a chemotherapy dosing policy in the `AhnChemoEnv` simulator, and compare it against conventional RL agents and the same LLM without fine-tuning.

## How the system works

1. **Environment**: `AhnChemoEnv` from the DTR-Bench / DTRGym package, installed as a dependency and used unchanged. Do not copy or modify the simulator source; if behavior needs to change, wrap the env.
2. **State to prompt**: the env observation is summarized into a text prompt.
3. **LLM policy**: the LLM itself is the policy. It reads the prompt and produces a dose action from a constrained set of outputs.
4. **PPO**: a custom, hand-written PPO loop in this repo updates the LLM through LoRA adapter weights. We do not use Tianshou, Stable-Baselines3, or another RL library for the LLM agent.
5. **Evaluation**: metrics and graphs compare the trained LLM agent against the baselines.

## Repository layout

```
.
├── claude.md
├── README.md
├── requirements.txt
├── .gitignore                 # .venv/, runs/, caches
├── configs/
│   ├── base.yaml              # env settings, seeds, PPO hyperparameters
│   ├── qwen3_8b.yaml          # per-model overrides: model name, LoRA, quantization
│   ├── medgemma_4b.yaml
│   └── deepseek_r1_7b.yaml
├── chemo_llm/                 # the package
│   ├── env.py                 # builds AhnChemoEnv from DTRGym, any wrappers
│   ├── prompts.py             # observation -> prompt text, prompt variants   (Tasnim)
│   ├── actions.py             # constrained output <-> dose action            (Tasnim)
│   ├── model.py               # load model + tokenizer, 4-bit, attach LoRA    (Scott)
│   ├── agent.py               # LLM policy: prompt -> action, log-prob, value (Scott)
│   ├── ppo.py                 # rollout buffer, GAE, PPO update               (Scott)
│   ├── baselines.py           # zero-shot LLM and conventional RL agents
│   ├── metrics.py             # evaluation metrics                            (Tasnim)
│   └── utils.py               # seeding, config loading, TensorBoard writer
├── scripts/
│   ├── train.py               # python scripts/train.py --config configs/qwen3_8b.yaml
│   ├── evaluate.py            # run a checkpoint or baseline over fixed eval seeds
│   └── plot.py                # graphs for the paper from run outputs
├── tests/                     # one test file per chemo_llm module
├── runs/                      # gitignored: TensorBoard events, checkpoints, adapters
└── figures/                   # committed: final graphs used in the paper
```

Conventions:

- Keep the package flat, one module per pipeline stage. Split a module only when it gets unwieldy.
- `actions.py` owns everything about the action constraint. `agent.py` and `ppo.py` get valid actions and log-probs from it and must not depend on how the constraint is implemented.
- Prompt variants live in `prompts.py` and are selected by config, so a prompt experiment is a config change, not a code change.
- `train.py` writes a copy of the resolved config and seed into each run directory under `runs/`.
- Only final paper graphs go in `figures/`; everything else a run produces stays in `runs/`.

## Models

Candidates, all loaded with Hugging Face Transformers and PEFT (LoRA):

- Qwen3 8B
- MedGemma 4B
- DeepSeek R1-Distill 7B

Training more than one model is optional and happens only if time allows. Code should take the model name from config so that switching models needs no code change.

## Hardware

Training runs on a single local GPU with 16 GB of VRAM.

- Load the 7B and 8B models in 4-bit (bitsandbytes) with LoRA adapters; full-precision loading will not fit alongside PPO gradients.
- Keep rollout and minibatch sizes small and use gradient accumulation instead of large batches.
- Before proposing a change that raises memory use (longer prompts, larger batches, higher LoRA rank, more trainable modules), say what it costs.

## Baselines

- Conventional RL agents (non-LLM policies) on the same environment.
- Zero-shot LLM: the same model and prompt with no PPO fine-tuning.

Every comparison must use the same env configuration, seeds, and evaluation episodes across agents.

## Tooling

- Python virtual environment with `python -m venv` and pip; dependencies in `requirements.txt`. No conda, uv, or poetry.
- Tests with `pytest`.
- Metrics logged with TensorBoard. Run outputs (event files, checkpoints, adapter weights) are never committed.
- The development machine is Windows 11 with PowerShell. Use paths and commands that work there, and avoid dependencies that are Linux-only.

Commands (update this section as the repo takes shape):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
tensorboard --logdir <run directory>
```


