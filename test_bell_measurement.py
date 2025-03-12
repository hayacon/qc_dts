import pytest
import numpy as np
from bell_measurement import Bell_measurement

def test_init():
    state_1 = ['H', 'V']
    state_2 = ['H', 'V']
    basis_1 = ['H', 'V']
    basis_2 = ['H', 'V']
    bm = Bell_measurement(state_1, state_2)
    assert bm.state_1 == state_1
    assert bm.state_2 == state_2
    assert bm.outcome_1 == []
    assert bm.outcome_2 == []

    with pytest.raises(AssertionError):
        Bell_measurement(['H'], ['H', 'V'],)

