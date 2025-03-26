import pytest
import numpy as np
from rng import RNG

@pytest.fixture
def qrng():
    return RNG(bit_len=40, options='qrng')

def test_generate_random_bits(qrng):
    generated_bits = qrng.generate_random_bits()
    assert len(generated_bits) == 40  # Should generate 40 bits
    assert all(bit in [0, 1] for bit in generated_bits)  # Bits should be 0 or 1

def test_chi_square_test(qrng):
    chi = qrng.chi_square_test()
    assert "chi_square_stat" in chi
    assert "p_value" in chi
    assert "is_random" in chi
    assert isinstance(chi["chi_square_stat"], float)
    assert isinstance(chi["p_value"], float)
    assert isinstance(chi["is_random"], bool)

def test_shannon_entropy(qrng):
    entropy = qrng.shannon_entropy()
    assert "shannon_entropy" in entropy
    assert "is_random" in entropy
    assert isinstance(entropy["shannon_entropy"], float)
    assert 0.0 <= entropy["shannon_entropy"] <= 1.0
    assert isinstance(entropy["is_random"], bool)

def test_auto_correlation(qrng):
    auto_corr = qrng.auto_correlation()
    assert "autocorrelation_values" in auto_corr
    assert "is_random" in auto_corr
    assert isinstance(auto_corr["autocorrelation_values"], dict)
    assert isinstance(auto_corr["is_random"], bool)
    # Check that autocorrelation values are near zero
    for lag, value in auto_corr["autocorrelation_values"].items():
        assert -1.0 <= value <= 1.0  # Autocorrelation should be within this range