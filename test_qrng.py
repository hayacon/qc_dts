import pytest
from io import StringIO
from unittest.mock import patch

from qrng import QRNG

def test_init():
    """
    Test that the QRNG object initializes with the correct bit_len
    and an empty random_bits list.
    """
    bit_len = 10
    q = QRNG(bit_len)
    assert q.bit_len == bit_len, "bit_len should match the constructor argument"
    assert isinstance(q.random_bits, list), "random_bits should be a list"
    assert len(q.random_bits) == 0, "random_bits should be empty initially"


def test_circuit_rng_length():
    """
    Test that circuit_rng generates the correct number of bits.
    """
    bit_len = 5
    q = QRNG(bit_len)
    result = q.circuit_rng()
    assert len(result) == bit_len, "Should generate the exact number of bits requested"
    assert len(q.random_bits) == bit_len, "The stored random_bits should match the result length"


def test_circuit_rng_values():
    """
    Test that circuit_rng outputs only valid bits (0 or 1).
    """
    bit_len = 5
    q = QRNG(bit_len)
    bits = q.circuit_rng()
    for bit in bits:
        assert bit in (0, 1), "Each generated value should be 0 or 1"


@patch('sys.stdout', new_callable=StringIO)
def test_benchmark_no_bits(mock_stdout):
    """
    Test that benchmark outputs a warning when no bits are generated.
    """
    q = QRNG(bit_len=0)
    q.benchmark()
    output = mock_stdout.getvalue()
    assert "No random bits generated yet." in output, "Should warn if no bits are available"


@patch('sys.stdout', new_callable=StringIO)
def test_benchmark_randomness(mock_stdout):
    """
    Test that benchmark runs a chi-square test and prints out relevant information.
    We only verify the presence of expected output lines (chi-square and p-value).
    """
    bit_len = 10
    q = QRNG(bit_len)
    q.circuit_rng()
    q.benchmark()
    output = mock_stdout.getvalue()

    # Check that the chi-square statistic and p-value are printed
    assert "Chi-square statistic:" in output, "Should display chi-square statistic"
    assert "P-value:" in output, "Should display p-value"
