from collections import OrderedDict

import pytest

from chemo_llm.true_state import ahn_true_state


class Base:
    def __init__(self, state):
        self.cur_state = state


class Wrapper:
    """Like a gym wrapper: the real env is under .unwrapped."""
    def __init__(self, state):
        self.unwrapped = Base(state)


def _state():
    return OrderedDict(N=[0.9], T=[0.2], I=[0.005], C=[0.3])


def test_reads_all_four_values_from_cur_state():
    out = ahn_true_state(Wrapper(_state()), [0.5, 0.5, 0.5], {})
    assert out == {"normal": 0.9, "tumor": 0.2, "immune": 0.005, "drug": 0.3}


def test_accepts_plain_floats_and_zero_dim_values():
    class Z:  # 0-d array-like
        def __float__(self):
            return 0.7
    s = OrderedDict(N=0.9, T=Z(), I=0.1, C=0.0)
    assert ahn_true_state(Wrapper(s), [0, 0, 0], {})["tumor"] == 0.7


def test_falls_back_to_info_state():
    class Bare:
        pass
    out = ahn_true_state(Bare(), [0, 0, 0], {"state": _state()})
    assert out["normal"] == 0.9


def test_unknown_drug_key_falls_back_to_observation():
    s = OrderedDict(N=[0.9], T=[0.2], I=[0.1], X=[1.0], Y=[2.0])
    assert ahn_true_state(Wrapper(s), [0.2, 0.1, 0.42], {})["drug"] == 0.42
    assert ahn_true_state(Wrapper(s), [[0, 0, 0.1], [0.2, 0.1, 0.42]], {})["drug"] == 0.42


def test_missing_state_raises_clear_error():
    class Bare:
        pass
    with pytest.raises(RuntimeError):
        ahn_true_state(Bare(), [0, 0, 0], {})
