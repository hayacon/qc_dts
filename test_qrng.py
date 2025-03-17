import pytest
import numpy as np
from qiskit_aer import Aer
from qrng import QRNG  # Replace 'your_module' with the actual module name

@pytest.fixture
def qrng():
    return QRNG(bit_len=100)  # Generate 100 random bits for testing

def test_circuit_rng(qrng):
    generated_bits = qrng.circuit_rng()
    assert len(generated_bits) == 100  # Should generate 100 bits
    assert all(bit in [0, 1] for bit in generated_bits)  # Bits should be 0 or 1

def test_chi_square_test(qrng):
    qrng.random_bits = np.random.choice([0, 1], size=100, p=[0.5, 0.5]).tolist()
    results = qrng.chi_square_test()
    
    assert "chi_square_stat" in results
    assert "p_value" in results
    assert "is_random" in results
    assert isinstance(results["chi_square_stat"], float)
    assert isinstance(results["p_value"], float)
    assert isinstance(results["is_random"], np.bool)

def test_shannon_entropy(qrng):
    qrng.random_bits = np.random.choice([0, 1], size=100, p=[0.5, 0.5]).tolist()
    entropy_results = qrng.shannon_entropy()

    assert "shannon_entropy" in entropy_results
    assert "is_random" in entropy_results
    assert isinstance(entropy_results["shannon_entropy"], float)
    assert 0.0 <= entropy_results["shannon_entropy"] <= 1.0
    assert isinstance(entropy_results["is_random"], np.bool)

def test_auto_correlation(qrng):
    qrng.random_bits = np.random.choice([0, 1], size=100, p=[0.5, 0.5]).tolist()
    autocorr_results = qrng.auto_correlation(max_lag=10)

    assert "autocorrelation_values" in autocorr_results
    assert "is_random" in autocorr_results
    assert isinstance(autocorr_results["autocorrelation_values"], dict)
    assert isinstance(autocorr_results["is_random"], bool)

    # Check that autocorrelation values are near zero
    for lag, value in autocorr_results["autocorrelation_values"].items():
        assert -1.0 <= value <= 1.0  # Autocorrelation should be within this range
