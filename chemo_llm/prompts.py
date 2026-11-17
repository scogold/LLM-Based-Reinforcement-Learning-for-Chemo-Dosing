"""Turns an AhnChemoEnv observation into the text prompt the LLM policy reads.

The policy picks a dose by reading the model's next-token probabilities over a fixed set
of answer letters (see agent.py), so this module has two jobs: describe the patient and the
dose options clearly, and end the prompt exactly where the model should write its answer.

Public interface:
    ACTION_LABELS    answer letters ["A", ..., "E"], one per dose level. Each must be a
                     single token for every model we use (agent.py checks this).
    DOSE_LEVELS      the dose rate each label stands for, (0.0, 0.25, 0.5, 0.75, 1.0).
                     cfg["doses"] must hold the same values in the same order, because the
                     prompt tells the model these numbers and the env receives cfg["doses"][k].
    PROMPT_VARIANTS  "base", "expert", "reward". All three give the task and the dose options.
                     "expert" adds background knowledge (the drug harms normal cells, drug
                     lingers in the blood, immune cells help, stop when the tumor is small).
                     "reward" adds that and also says what the score rewards and penalises.
    build_prompt(obs, variant, tokenizer) -> str
                     The full chat-templated prompt for one observation.

Observation formats accepted (values in OBS_NAMES order: tumor, immune, drug):
    1-D [tumor, immune, drug]      current state only.
    2-D, one row per step,         current state plus trends (rising / falling / flat),
    oldest first                   tumor relative to the oldest row, and up to MAX_HISTORY
                                   earlier steps.
    NaN or None values are written as "unavailable" (the env can mask values in its noisier
    settings).

Chat template handling:
    The prompt ends right after the assistant turn header, so the next token is the bare
    answer letter. Qwen3's template is called with enable_thinking=False, which inserts an
    empty think block. DeepSeek-R1 templates can end with an open "<think>" tag; it is closed
    here so the model answers instead of starting to reason.

Constants that must match the rest of the project: OBS_NAMES (the env's observation order:
tumor cells, immune cells, drug concentration) and STEP_HOURS (6 hours per env step).

Used by agent.py (build_prompt, ACTION_LABELS), configs/base.yaml (doses), and
tests/test_prompts.py. Standard library only.
"""
from __future__ import annotations

import math

OBS_NAMES = ("tumor", "immune", "drug")   # VERIFY against AhnChemoEnv's observation
STEP_HOURS = 6                            # paper default: 6 hours per step
ACTION_LABELS = ["A", "B", "C", "D", "E"]
DOSE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)  # keep equal to cfg["doses"]
DOSE_NAMES = ("no drug", "low", "medium", "high", "maximum")
PROMPT_VARIANTS = ("base", "expert", "reward")
MAX_HISTORY = 6

_TASK = (
    "You control chemotherapy dosing for a simulated cancer patient. Each step is "
    f"{STEP_HOURS} hours. At every step you pick one drug dose level. Your goal is to shrink "
    "the tumor while keeping normal cells and immune cells healthy and using as little "
    "drug as needed."
)

_EXPERT = (
    "Background knowledge:\n"
    "- The drug kills tumor cells but also harms normal cells, which you cannot observe.\n"
    "- The drug stays in the blood and decays slowly, so a high drug level means a recent "
    "dose is still acting. Do not stack more drug on top of a high drug level.\n"
    "- Immune cells help fight the tumor. When they are low, be more careful.\n"
    "- When the tumor is already very small, reduce or stop the drug."
)

_REWARD = (
    "How you are scored at each step: more normal cells is better, more immune cells is "
    "better, fewer tumor cells is better, and a higher dose costs points."
)


def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _fmt(x) -> str:
    """At least 3 decimals and at least 3 significant digits, without scientific notation,
    so a tiny but regrowing tumor shows as 0.0000123 rather than 0.000. Capped at 8 decimals
    to keep prompts short; anything below 1e-8 prints as 0.00000000."""
    if _is_missing(x):
        return "unavailable"
    x = float(x)
    if x == 0:
        return "0.000"
    decimals = min(8, max(3, 2 - math.floor(math.log10(abs(x)))))
    return f"{x:.{decimals}f}"


def _trend(prev, cur, tol: float = 0.02) -> str:
    if _is_missing(prev) or _is_missing(cur):
        return "unknown"
    prev, cur = float(prev), float(cur)
    if abs(cur - prev) <= tol * max(abs(prev), 1e-6):
        return "flat"
    return "rising" if cur > prev else "falling"


def _rows(obs):
    """Return a list of per-step rows (oldest -> newest), each a dict by OBS_NAMES."""
    seq = list(obs)
    if not seq:
        raise ValueError("empty observation")
    if hasattr(seq[0], "__len__") and not isinstance(seq[0], str):
        rows = [list(r) for r in seq]
    else:
        rows = [seq]
    for r in rows:
        if len(r) != len(OBS_NAMES):
            raise ValueError(f"expected {len(OBS_NAMES)} values per step, got {len(r)}")
    return [dict(zip(OBS_NAMES, r)) for r in rows]


def system_prompt(variant: str = "base") -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"variant must be one of {PROMPT_VARIANTS}")
    parts = [_TASK]
    if variant in ("expert", "reward"):
        parts.append(_EXPERT)
    if variant == "reward":
        parts.append(_REWARD)
    options = "\n".join(
        f"{lab}: {name} (rate {d:.2f})"
        for lab, name, d in zip(ACTION_LABELS, DOSE_NAMES, DOSE_LEVELS)
    )
    parts.append("Dose options:\n" + options)
    parts.append(f"Reply with only one letter ({ACTION_LABELS[0]}-{ACTION_LABELS[-1]}).")
    return "\n\n".join(parts)


def user_prompt(obs) -> str:
    rows = _rows(obs)
    cur = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None
    lines = ["Current observation:"]
    for name, label in (("tumor", "tumor cells"), ("immune", "immune cells"),
                        ("drug", "drug in blood")):
        trend = f" ({_trend(prev[name], cur[name])})" if prev else ""
        lines.append(f"  {label}: {_fmt(cur[name])}{trend}")
    if prev:
        first = rows[0]["tumor"]
        if not _is_missing(first) and not _is_missing(cur["tumor"]) and float(first) != 0:
            lines.append(f"  tumor vs. oldest shown: {float(cur['tumor']) / float(first):.0%}")
        hist = rows[-(MAX_HISTORY + 1):-1]
        lines.append("Earlier steps (oldest first), tumor / immune / drug:")
        for r in hist:
            lines.append(f"  {_fmt(r['tumor'])} / {_fmt(r['immune'])} / {_fmt(r['drug'])}")
    lines.append("Choose the next dose.")
    return "\n".join(lines)


def build_prompt(obs, variant: str, tokenizer) -> str:
    """Chat-templated prompt that ends right where the model writes its action."""
    messages = [
        {"role": "system", "content": system_prompt(variant)},
        {"role": "user", "content": user_prompt(obs)},
    ]
    # enable_thinking=False is read by Qwen3's template; other templates ignore it.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    # Some DeepSeek-R1 templates end the prompt with an open "<think>" tag. Close it so
    # the next token is the answer rather than the start of a reasoning block.
    if text.rstrip().endswith("<think>"):
        text = text.rstrip() + "\n</think>\n\n"
    return text
