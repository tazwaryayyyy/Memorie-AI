"""step 8d-1: the planning-aid power simulation is deterministic and uses the real exact test."""
from eval import power_sim
from eval.analysis import exact_sign_flip_p


def test_simulation_is_deterministic_and_the_null_is_exactly_null():
    a = power_sim.simulate(8, None, 1, trials=40)
    assert a == power_sim.simulate(8, None, 1, trials=40)
    assert power_sim.realized_effect(None) == 0.0
    assert 0.0 < power_sim.realized_effect(0.2) < 0.2  # ceiling clipping shrinks the nominal effect


def test_exact_test_is_conservative_under_the_null_and_gains_power_with_effect():
    null = power_sim.simulate(8, None, 1, trials=200)["power_exact"]
    big = power_sim.simulate(16, 0.30, 3, trials=200)["power_exact"]
    assert null <= 0.05 and big > null


def test_simulation_uses_the_same_exact_test_as_the_analysis():
    assert power_sim.exact_sign_flip_p is exact_sign_flip_p
