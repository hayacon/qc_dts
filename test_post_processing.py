import pytest
from post_processing import AlicePostProcessing, BobPostProcessing  # Replace 'your_module' with actual module name

@pytest.fixture
def alice():
    alice_bases = ['Z', 'X', 'Z', 'X', 'Z']
    alice_bits = [1, 0, 1, 1, 0]
    return AlicePostProcessing(alice_bases, alice_bits)

@pytest.fixture
def bob():
    bob_bases = ['Z', 'X', 'Z', 'X', 'X']
    bob_bits = [0, 0, 1, 0, 1]
    return BobPostProcessing(bob_bases, bob_bits)


# -------- Alice ---------

def test_alice_sifting(alice, bob):
    relay_outcomes = ['|psi^->', '|psi^->', 'no_BSM', '|psi^+>', '|psi^->']
    bob_bases = ['Z', 'X', 'Z', 'X', 'Z']

    sifted_length = alice.sifting(bob_bases, relay_outcomes)
    assert sifted_length == 4
    assert alice.sifted_key == [1, 0, 1, 0]  # Expected sifted key

def test_alice_reveal_subset_bits(alice):
    alice.sifted_key = [1, 0, 0, 1]
    indices = [0, 2]
    revealed = alice.reveal_subset_bits(indices)
    assert revealed == {0: 1, 2: 0}

def test_alice_compute_parity(alice):
    alice.sifted_key = [1, 0, 1, 1]
    assert alice.compute_parity([0, 2, 3]) == (1 + 1 + 1) % 2  # Parity calculation

def test_alice_remove_indices(alice):
    alice.sifted_key = [1, 0, 1, 0]
    alice.remove_indices([1, 3])
    assert alice.sifted_key == [1, 1]  # Removed indices

def test_alice_privacy_amplification(alice):
    alice.sifted_key = [1, 0, 1, 1, 0, 1, 0, 1]
    final_key = alice.privacy_amplification(final_length=8)
    assert len(final_key) == 8  # Ensure correct length

# -------- Bob ---------

def test_bob_sifting(bob):
    alice_bases = ['Z', 'X', 'Z', 'X', 'Z']
    relay_outcomes = ['|psi^->', '|psi^+>', 'no_BSM', '|psi^+>', '|psi^->']

    sifted_length = bob.sifting(alice_bases, relay_outcomes)
    assert sifted_length == 3
    assert bob.sifted_key == [1, 0, 0]  # Expected sifted key after MDI flips

def test_bob_estimate_qber_sample(bob):
    bob.sifted_key = [1, 0, 1, 0]
    alice_revealed = {0: 1, 1: 1, 2: 1}
    qber, indices = bob.estimate_qber_sample(alice_revealed)
    assert qber == 1 / 3  # 1 mismatch out of 3

def test_bob_remove_indices(bob):
    bob.sifted_key = [1, 0, 1, 0]
    bob.remove_indices([1, 3])
    assert bob.sifted_key == [1, 1]  # Removed indices

def test_bob_privacy_amplification(bob):
    bob.sifted_key = [1, 0, 1, 1, 0, 1, 0, 1]
    final_key = bob.privacy_amplification(final_length=8)
    assert len(final_key) == 8  # Ensure correct length

def test_measure_qber_direct(bob):
    bob.sifted_key = [1, 0, 1, 0]
    alice_key = [1, 1, 1, 0]
    qber = bob.measure_qber_direct(alice_key)
    assert qber == 1 / 4  # 1 mismatch out of 4
