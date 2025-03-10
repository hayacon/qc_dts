import pytest
from io import StringIO
from unittest.mock import patch
from user import User


def test_init():
    ''' 
    testing the initialization of the User class
    '''
    bit_len = 10
    user = User(bit_len=bit_len)
    assert user.bit_len == bit_len, "bit_len should match the constructor argument"
    assert user.basis == None, "basis should be None initially"
    assert isinstance(user.states, list), "states should be None initially"

def test_set_basis():
    ''' 
    testing the set_basis method of the User class
    '''
    user = User(bit_len=10)
    basis = user.set_basis()
    assert isinstance(basis, list), "basis should be a list"
    assert len(basis) == 10, "basis should have the same length as bit_len"
    assert basis[0] in ['Z', 'X'], "basis should be either 'Z' or 'X'"

def test_state_encoder():
    ''' 
    testing the set_state method of the User class
    '''
    user = User(bit_len=10)
    basis = user.set_basis()
    states = user.state_encoder()
    assert isinstance(states, list), "states should be a list"
    assert len(states) == 10, "states should have the same length as bit_len"
    assert states[0] in ['V', 'H', 'D', 'A'], "states should be either '0' or '1'"

