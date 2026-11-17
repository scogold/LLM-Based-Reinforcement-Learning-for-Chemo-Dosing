import math

import pytest

from chemo_llm import prompts as P


class FakeTok:
    """Mimics a chat template: ends right after the assistant header."""
    def __init__(self, tail=""):
        self.tail = tail

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        assert tokenize is False and add_generation_prompt is True
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + "<assistant>" + self.tail


def test_labels_and_doses_line_up():
    assert len(P.ACTION_LABELS) == len(P.DOSE_LEVELS) == len(P.DOSE_NAMES)
    assert len(set(P.ACTION_LABELS)) == len(P.ACTION_LABELS)
    assert all(len(lab) == 1 for lab in P.ACTION_LABELS)


def test_system_prompt_lists_every_label_and_dose():
    s = P.system_prompt("base")
    for lab, d in zip(P.ACTION_LABELS, P.DOSE_LEVELS):
        assert f"{lab}:" in s and f"{d:.2f}" in s


def test_variants_differ():
    base, exp, rew = (P.system_prompt(v) for v in P.PROMPT_VARIANTS)
    assert "Background knowledge" not in base
    assert "Background knowledge" in exp and "scored" not in exp
    assert "Background knowledge" in rew and "scored" in rew


def test_bad_variant():
    with pytest.raises(ValueError):
        P.system_prompt("nope")


def test_single_obs_prompt():
    txt = P.user_prompt([0.8, 0.5, 0.2])
    assert "0.800" in txt and "0.500" in txt and "0.200" in txt
    assert "rising" not in txt and "Earlier steps" not in txt


def test_history_prompt_has_trends():
    txt = P.user_prompt([[1.0, 0.5, 0.0], [0.9, 0.5, 0.4], [0.8, 0.52, 0.5]])
    assert "tumor cells: 0.800 (falling)" in txt
    assert "drug in blood: 0.500 (rising)" in txt
    assert "80%" in txt and "Earlier steps" in txt


def test_missing_values_shown_as_unavailable():
    txt = P.user_prompt([float("nan"), None, 0.2])
    assert txt.count("unavailable") == 2 and "nan" not in txt.lower()


def test_numpy_like_scalars_ok():
    class S(float):
        pass
    assert "0.300" in P.user_prompt([S(0.3), S(0.4), S(0.5)])


def test_bad_obs_shape():
    with pytest.raises(ValueError):
        P.user_prompt([0.1, 0.2])
    with pytest.raises(ValueError):
        P.user_prompt([])


def test_history_window_is_capped():
    obs = [[1.0 - i * 0.01, 0.5, 0.1] for i in range(30)]
    lines = P.user_prompt(obs).splitlines()
    rows = [l for l in lines if l.startswith("  ") and " / " in l]
    assert len(rows) == P.MAX_HISTORY


def test_build_prompt_ends_at_assistant_turn():
    out = P.build_prompt([0.8, 0.5, 0.2], "expert", FakeTok())
    assert out.endswith("<assistant>")


def test_build_prompt_closes_open_think_tag():
    out = P.build_prompt([0.8, 0.5, 0.2], "base", FakeTok(tail="<think>\n"))
    assert out.endswith("</think>\n\n")
    assert out.count("<think>") == 1

@pytest.mark.parametrize("value, expected", [
    (0, "0.000"),
    (0.2, "0.200"),
    (0.005, "0.00500"),        # 3 significant digits -> 5 decimals
    (0.0000123, "0.0000123"),
    (1e-12, "0.00000000"),     # capped at 8 decimals
    (float("nan"), "unavailable"),
    (None, "unavailable"),
])
def test_fmt_small_and_missing_values(value, expected):
    assert P._fmt(value) == expected


@pytest.mark.parametrize("value, expected", [(0.1, "0.100"), (0.8, "0.800"), (0.12345, "0.123"), (1.0, "1.000")])
def test_fmt_values_from_point_one_up_use_three_decimals(value, expected):
    assert P._fmt(value) == expected


def test_fmt_never_uses_scientific_notation():
    assert "e" not in P._fmt(3e-7).lower()
