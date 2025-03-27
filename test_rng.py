import pytest
import numpy as np
from qiskit_aer import Aer
from rng import RNG  # Replace 'your_module' with the actual module name

import pytest
from rng import RNG
import numpy as np
import secrets
import random
from Crypto.Random import get_random_bytes
import os
from scipy.stats import chisquare, entropy
from scipy import stats

@pytest.fixture
def rng_instance():
    return RNG(bit_len=100, options='numpy')

def test_generate_random_bits_numpy(rng_instance):
    rng_instance.options = 'numpy'
    bits = rng_instance.generate_random_bits()
    assert len(bits) == 100
    assert all(bit in [0, 1] for bit in bits)

def test_generate_random_bits_os(rng_instance):
    rng_instance.options = 'os'
    bits = rng_instance.generate_random_bits()
    assert len(bits) == 100
    assert all(bit in [0, 1] for bit in bits)

def test_generate_random_bits_secrets(rng_instance):
    rng_instance.options = 'secrets'
    bits = rng_instance.generate_random_bits()
    assert len(bits) == 100
    assert all(bit in [0, 1] for bit in bits)

def test_generate_random_bits_random(rng_instance):
    rng_instance.options = 'random'
    bits = rng_instance.generate_random_bits()
    assert len(bits) == 100
    assert all(bit in [0, 1] for bit in bits)

def test_generate_random_bits_crypto(rng_instance):
    rng_instance.options = 'crypto'
    bits = rng_instance.generate_random_bits()
    assert len(bits) == 100
    assert all(bit in [0, 1] for bit in bits)

def test_chi_square_test(rng_instance):
    rng_instance.options = 'numpy'
    rng_instance.generate_random_bits()
    result = rng_instance.chi_square_test()
    assert 'chi_square_stat' in result
    assert 'p_value' in result
    assert 'is_random' in result

def test_shannon_entropy(rng_instance):
    rng_instance.options = 'numpy'
    rng_instance.generate_random_bits()
    result = rng_instance.shannon_entropy()
    assert 'shannon_entropy' in result
    assert 'is_random' in result

def test_auto_correlation(rng_instance):
    rng_instance.options = 'numpy'
    rng_instance.generate_random_bits()
    result = rng_instance.auto_correlation()
    assert 'autocorrelation_values' in result
    assert 'is_random' in result

def test_cumulative_sums(rng_instance):
    rng_instance.options = 'numpy'
    rng_instance.generate_random_bits()
    result_forward, result_backward = rng_instance.cumulative_sums()
    assert 'p_value' in result_forward
    assert 'is_random' in result_forward
    assert 'p_value' in result_backward
    assert 'is_random' in result_backward

def test_random_excursions(rng_instance):
    rng_instance.options = 'numpy'
    rng_instance.generate_random_bits()
    result = rng_instance.random_excursions()
    assert 'p_value' in result
    assert 'is_random' in result
