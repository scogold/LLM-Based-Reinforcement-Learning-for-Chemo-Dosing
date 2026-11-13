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
│   ├── prompts.py             # observation -> prompt text, prompt variants,
│   │                          #   ACTION_LABELS (answer options)              (Tasnim)
│   ├── agent.py               # load model (4-bit + LoRA); LLM policy:
│   │                          #   action constraint, label -> dose,
│   │                          #   prompt -> action, log-prob, value           (Scott)
│   ├── ppo.py                 # rollout buffer, GAE, PPO update               (Scott)
│   ├── evaluation.py          # metrics (Tasnim); zero-shot LLM and
│   │                          #   conventional RL baselines
│   └── utils.py               # builds AhnChemoEnv + wrappers, seeding,
│                              #   config loading, TensorBoard writer
├── scripts/
│   ├── train.py               # python scripts/train.py --config configs/qwen3_8b.yaml
│   ├── evaluate.py            # run a checkpoint or baseline over fixed eval seeds
│   └── plot.py                # graphs for the paper from run outputs
├── tests/                     # one test file per chemo_llm module
├── runs/                      # gitignored: TensorBoard events, checkpoints, adapters
└── figures/                   # committed: final graphs used in the paper
```

Conventions:

- Keep the package flat and small: few modules, each covering a whole pipeline stage. Don't add a new module for something a few dozen lines long; put it in the module it belongs with. Split a module only when it gets unwieldy.
- `agent.py` owns the action constraint: it keeps only the next-token logits of the action labels, and maps each label to a dose from config. `prompts.py` owns only the text, including `ACTION_LABELS`, the answer options the prompt offers; it is the single place those labels are defined. `ppo.py` works with action indices and log-probs and must not depend on how the constraint is implemented.
- Prompt variants live in `prompts.py` and are selected by config, so a prompt experiment is a config change, not a code change.
- `train.py` writes a copy of the resolved config and seed into each run directory under `runs/`.
- Only final paper graphs go in `figures/`; everything else a run produces stays in `runs/`.

## Subagents

Project subagents are defined in `.claude/agents/`:

| Agent | Model | Use it for |
| --- | --- | --- |
| `researcher` | Sonnet | Literature and docs questions behind design decisions; citations for related work. Read-only plus web. |
| `test-writer` | Sonnet | Writing pytest tests for a module once its interface exists, from the specified behaviour rather than the implementation. Edits only `tests/` and `pytest.ini`. |
| `test-runner` | Haiku | Running pytest after changes and before pushing to `main`. Reports failures, never fixes. |
| `code-reviewer` | Sonnet | Run after each piece of code is written in the main session. Simplifies it in place: removes unnecessary safeguards, over-abstraction and noise without changing behaviour, then runs the fast tests. Asks about anything that might change behaviour. |
| `rl-reviewer` | Opus | Reviewing PPO, policy, action, env and evaluation code for silent correctness bugs, especially before long training runs. Read-only. |
| `experiment-analyst` | Sonnet | Comparing runs in `runs/` across seeds and producing graphs through `scripts/plot.py`. |
| `basic-debugger` | Haiku | First stop for every bug. Fixes what it can and verifies; otherwise escalates. |
| `super-debugger` | Opus | Only after `basic-debugger` escalates: NaNs, PPO not learning or diverging, out-of-memory, nondeterminism, and anything else it couldn't fix. May run short reproduction runs in `runs/debug-*/`; proposes rather than applies changes to the experimental design. |

Debugging always goes through the two debuggers in order:

1. Send every bug to `basic-debugger` first, including ones that look hard. Don't go straight to `super-debugger`.
2. If `basic-debugger` reports that it fixed and verified the bug, stop there.
3. If its report ends with an escalation (it could not find or fix the cause), immediately call `super-debugger` and pass it the original problem plus `basic-debugger`'s full report, including what it changed and what it ruled out.

Subagents cannot call each other, so the main session does this handoff.

After writing each piece of code in the main session, call `code-reviewer` on it before moving on. Pass any questions in its report on to the user.

The paper is written by hand; there is no writing agent. Use subagents when their output can be summarised back, such as a test report, a review or a cited answer. Build new features in the main session; the debuggers only fix existing code.

## Models

Candidates, all loaded with Hugging Face Transformers and PEFT (LoRA):

- Qwen3 8B
- MedGemma 4B
- DeepSeek R1-Distill 7B

Code should take the model name from config so that switching models needs no code change.

## Hardware

Training targets a single GPU with 16 GB of VRAM.

- Load the 7B and 8B models in 4-bit (bitsandbytes) with LoRA adapters; full-precision loading will not fit alongside PPO gradients.
- Keep rollout and minibatch sizes small and use gradient accumulation instead of large batches.
- Before proposing a change that raises memory use (longer prompts, larger batches, higher LoRA rank, more trainable modules), say what it costs.

## Baselines

- Conventional RL agents (non-LLM policies) on the same environment.
- Zero-shot LLM: the same model and prompt with no PPO fine-tuning.

Every comparison must use the same env configuration, seeds, and evaluation episodes across agents.

## Tooling

- Python virtual environment with `python -m venv` and pip; dependencies in `requirements.txt`. No conda, uv, or poetry.
- Tests with `pytest`. Default tests run on CPU in seconds using a stub policy or a tiny random Hugging Face model. Tests that need a real 4-8B model are marked `@pytest.mark.gpu` and run only where a suitable GPU is available. Run the fast suite with `pytest -m "not gpu"`.
- Metrics logged with TensorBoard. Run outputs (event files, checkpoints, adapter weights) are never committed.
- Code must run on Windows as well as Linux. Use `pathlib` for paths and avoid Linux-only dependencies.

Commands (update this section as the repo takes shape):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install --no-deps --ignore-requires-python simglucose==0.2.3 "DTRGym @ git+https://github.com/GilesLuo/DTRGym.git@1f4fed49769b371c01d90959d2e012154068b3be"
pytest
tensorboard --logdir <run directory>
```


