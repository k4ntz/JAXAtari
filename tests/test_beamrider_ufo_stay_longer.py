
import jax
import jax.numpy as jnp
from jaxatari.games.jax_beamrider import JaxBeamrider, BeamriderConstants, WhiteUFOPattern

def test_ufo_stay_longer():
    env = JaxBeamrider()
    
    # Reset
    obs, state = env.reset(jax.random.key(42))
    
    # We want to check how long a UFO stays on the top lane.
    # Initial state has UFOs at y=43.0, which is exactly TOP_CLIP.
    # So they start on top lane.
    
    ufo_idx = 0
    
    # We will simulate for X steps and count how many times they leave the top lane.
    # With alpha=0.002 (old), they would leave relatively quickly.
    # With alpha=0.0005 (new), they should stay longer.
    
    steps_to_sim = 1000
    
    # We can check the probability function directly to see the theoretical impact
    # entropy_heat_prob(steps, alpha=0.0005) vs alpha=0.002
    
    # Let's verify via simulation though.
    # We'll just run steps and see if they are still there after some time.
    
    # To make the test deterministic and fast, we can just check the probability value for a specific time_on_lane
    # But checking actual behavior is better.
    
    # Let's check `time_on_lane` value. It increments while on top lane.
    # If they leave, it resets.
    
    max_time_on_lane = 0
    
    current_state = state
    key = jax.random.key(0)
    
    # Force UFO to be on top lane initially (it is) and IDLE pattern
    # The default reset puts them at 43.0 with IDLE (0) pattern? 
    # Reset puts pattern_id to 0 (IDLE).
    
    # Run for some steps
    for _ in range(200):
        key, subkey = jax.random.split(key)
        # Random action
        action = 0 # NOOP
        obs, current_state, reward, done, info = env.step(current_state, action)
        
        # Check time on lane for ufo 0
        t = current_state.level.white_ufo_time_on_lane[ufo_idx]
        if t > max_time_on_lane:
            max_time_on_lane = t
            
        # If it resets to 0, it means it left the lane (or we just started measuring but here we start at 0)
        # Actually, reset_level sets time_on_lane to [0, 0, 0].
        # So it should increase.
        
    print(f"Max time on lane observed: {max_time_on_lane}")
    
    # With alpha=0.002, probability to swap rises faster.
    # 200 steps -> 200/10 = 20 units.
    # Old: 1 - exp(-0.002 * 20) = 1 - exp(-0.04) ≈ 0.04 probability max
    # New: 1 - exp(-0.0005 * 20) = 1 - exp(-0.01) ≈ 0.01 probability max
    
    # It's stochastic, so harder to verify with just one run if they "stay longer" compared to before without running both versions.
    # But we can assert that the probability calculation logic is what we expect by inspecting the class method if possible,
    # or just trust the code change.
    
    # Let's verify the method returns lower prob for same input.
    p_old = env.entropy_heat_prob(200, alpha=0.002)
    p_new = env.entropy_heat_prob(200) # uses default which we changed
    
    print(f"Prob at step 200 (Old alpha=0.002): {p_old}")
    print(f"Prob at step 200 (New alpha=0.0005): {p_new}")
    
    assert p_new < p_old, f"New probability {p_new} should be lower than old {p_old}"
    assert p_new < 0.015, f"New probability {p_new} should be very low around step 200"

    print("Test Passed!")

if __name__ == "__main__":
    test_ufo_stay_longer()
