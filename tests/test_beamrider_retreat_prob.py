
import jax
import jax.numpy as jnp
from jaxatari.games.jax_beamrider import JaxBeamrider

def test_retreat_prob_reduced():
    env = JaxBeamrider()
    
    # Test min probability (at time 0)
    p_min = env._white_ufo_retreat_prob(jnp.array(0))
    print(f"P_min (t=0): {p_min}")
    assert p_min == 0.005, f"Expected min prob 0.005, got {p_min}"
    
    # Test max probability (at very large time)
    # alpha is 0.01. Heat = 1 - exp(-0.01 * t)
    # For heat ~ 1, exp(-0.01 * t) ~ 0 => t very large (e.g. 1000)
    p_max_approx = env._white_ufo_retreat_prob(jnp.array(1000))
    print(f"P_max approx (t=1000): {p_max_approx}")
    # Should be close to 0.1
    assert p_max_approx > 0.09 and p_max_approx < 0.1001, f"Expected max prob approx 0.1, got {p_max_approx}"
    
    # Test intermediate value
    # t=100. heat = 1 - exp(-1) = 1 - 0.3678 = 0.632
    # p = 0.005 + (0.1 - 0.005) * 0.632 = 0.005 + 0.095 * 0.632 = 0.005 + 0.060 = 0.065
    p_mid = env._white_ufo_retreat_prob(jnp.array(100))
    print(f"P_mid (t=100): {p_mid}")
    assert p_mid < 0.07, f"Expected mid prob < 0.07, got {p_mid}"

    print("Test Passed!")

if __name__ == "__main__":
    test_retreat_prob_reduced()
