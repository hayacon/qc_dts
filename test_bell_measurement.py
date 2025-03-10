import pytest
import numpy as np

from bell_measurement import Bell_measurement
from user import User


def test_init():
    """
    Test that the BellMeasurement object initializes with the correct bit_len
    and an empty random_bits list.
    """
    user_1 = User(10)
    user_2 = User(10)
    basis_1 = user_1.set_basis()
    basis_2 = user_2.set_basis()
    state_1 = user_1.state_encoder()
    state_2 = user_2.state_encoder()
    b = Bell_measurement(state_1, state_2, basis_1, basis_2)
    assert isinstance(b.state_1, list)
    assert isinstance(b.state_2, list)
    assert len(b.state_1) == len(b.state_2)

def test_beam_spliter():
    """
    Test that the beam_spliter method returns the correct matrix
    """
    user_1 = User(10)
    user_2 = User(10)
    basis_1 = user_1.set_basis()
    basis_2 = user_2.set_basis()
    state_1 = user_1.state_encoder()
    state_2 = user_2.state_encoder()
    b = Bell_measurement(state_1, state_2, basis_1, basis_2)