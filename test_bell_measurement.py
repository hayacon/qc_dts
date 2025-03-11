import pytest
import numpy as np
from bell_measurement import Bell_measurement

def test_init():
    state_1 = ['H', 'V']
    state_2 = ['H', 'V']
    basis_1 = ['H', 'V']
    basis_2 = ['H', 'V']
    bm = Bell_measurement(state_1, state_2, basis_1, basis_2)
    assert bm.state_1 == state_1
    assert bm.state_2 == state_2
    assert bm.outcome_1 == []
    assert bm.outcome_2 == []

    with pytest.raises(AssertionError):
        Bell_measurement(['H'], ['H', 'V'], basis_1, basis_2)

def test_beam_spliter():
    bm = Bell_measurement(['H'], ['H'], ['H'], ['H'])
    assert bm.beam_spliter('H') in ['H', 'V']
    assert bm.beam_spliter('V') in ['H', 'V']
    assert bm.beam_spliter('D') == 'H'
    assert bm.beam_spliter('A') == 'V'
    assert bm.beam_spliter('X') == 'Invalid state'

def test_polarization_beam_spliter():
    bm = Bell_measurement(['H'], ['H'], ['H'], ['H'])
    assert bm.polarization_beam_spliter('H', 'Alice') == 'D2h'
    assert bm.polarization_beam_spliter('V', 'Alice') == 'D1v'
    assert bm.polarization_beam_spliter('H', 'Bob') == 'D1h'
    assert bm.polarization_beam_spliter('V', 'Bob') == 'D2v'

def test_announce_result():
    pass