import pytest
import numpy as np
from bell_measurement import Bell_measurement


def test_init_length_mismatch():
    """
    If state_1 and state_2 have different lengths, we expect an AssertionError.
    """
    with pytest.raises(AssertionError):
        _ = Bell_measurement(["H","V"], ["H"])


@pytest.mark.parametrize("a_pol,b_pol", [
    ("H","H"), ("H","V"), ("V","V"),
    ("D","D"), ("A","A"), ("D","A"), ("A","D"),
    ("D","H"), ("A","V")
])

def test_prob_sum_to_one(a_pol, b_pol):
    """
    Check that for each pair of polarizations, the sum of the probability
    distribution is close to 1.
    """
    bsm = Bell_measurement([a_pol],[b_pol])
    # We'll just do one pair, so run_measurements or direct
    dist = bsm.simulate_two_photon_BSM(a_pol, b_pol)
    total_prob = sum(dist.values())
    assert np.isclose(total_prob, 1.0, atol=1e-7), \
        f"Probabilities do not sum to 1 for {a_pol} vs {b_pol}"

def test_sample_is_valid_outcome():
    """
    For a known distribution, check that sampling returns only valid keys.
    """
    bsm = Bell_measurement(["H"], ["V"])  # one pair
    dist = bsm.simulate_two_photon_BSM("H","V")
    # sample many times, ensure each sample is a valid outcome
    for _ in range(50):
        outcome = bsm.sample_detection_outcome(dist)
        assert outcome in dist, f"Sampled outcome {outcome} not in distribution keys!"

def test_run_measurements():
    """
    Test that `run_measurements()` populates results with correct length.
    """
    alice = ["H","D","V"]
    bob   = ["V","D","A"]
    bsm = Bell_measurement(alice, bob)
    bsm.run_measurements()
    assert len(bsm.results) == 3, "Should have 3 measurement results"
    # Check each entry has the keys we expect
    for r in bsm.results:
        assert "Alice" in r
        assert "Bob" in r
        assert "prob_dist" in r
        assert "sample_outcome" in r
        assert "interpretation" in r

def test_interpretation():
    """
    Manually check interpretation for known outcomes.
    """
    bsm = Bell_measurement([], [])
    # psi^+ -> D1H+D1V or D2H+D2V
    assert bsm.interpret_bsm_outcome("D1H+D1V") == "|psi^+>"
    assert bsm.interpret_bsm_outcome("D2V+D2H") == "|psi^+>"

    # psi^- -> D1H+D2V or D1V+D2H
    assert bsm.interpret_bsm_outcome("D1H+D2V") == "|psi^->"
    assert bsm.interpret_bsm_outcome("D2H+D1V") == "|psi^->"

    # no_BSM for others
    assert bsm.interpret_bsm_outcome("D1H+D1H") == "no_BSM"
    assert bsm.interpret_bsm_outcome("D2V+D2V") == "no_BSM"
