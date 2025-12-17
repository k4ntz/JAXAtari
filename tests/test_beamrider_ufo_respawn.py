
import jax
import jax.numpy as jnp
from jaxatari.games.jax_beamrider import JaxBeamrider, BeamriderConstants, WhiteUFOPattern

def test_ufo_respawn():
    env = JaxBeamrider()
    
    # Reset
    obs, state = env.reset(jax.random.key(0))
    
    # Target UFO index 0
    ufo_idx = 0
    
    # We place the UFO and bullet at (77.0, 100.0) to ensure valid collision
    test_pos = jnp.array([77.0, 100.0])
    
    # Update state: Move UFO 0 to test_pos and set its pattern to something non-IDLE
    current_ufo_pos = state.level.white_ufo_pos
    new_white_ufo_pos = current_ufo_pos.at[0, ufo_idx].set(test_pos[0])
    new_white_ufo_pos = new_white_ufo_pos.at[1, ufo_idx].set(test_pos[1])
    
    # Set pattern to SHOOT (5)
    new_patterns = state.level.white_ufo_pattern_id.at[ufo_idx].set(int(WhiteUFOPattern.SHOOT))
    
    # Place bullet at same position
    new_level = state.level._replace(
        white_ufo_pos=new_white_ufo_pos,
        white_ufo_pattern_id=new_patterns,
        player_shot_pos=test_pos,
        bullet_type=jnp.array(env.consts.LASER_ID),
        player_shot_vel=jnp.array([0.0, 0.0])
    )
    state = state._replace(level=new_level)
    
    print(f"Set UFO pos to: {test_pos}, Pattern to: {int(new_patterns[ufo_idx])} ({WhiteUFOPattern.SHOOT.name})")
    
    # Step the environment
    action = 0 # NOOP
    obs, next_state, reward, done, info = env.step(state, action)
    
    # Check Position
    new_ufo_pos = jnp.array([next_state.level.white_ufo_pos[0][ufo_idx], next_state.level.white_ufo_pos[1][ufo_idx]])
    expected_pos = jnp.array([77.0, 43.0])
    distance = jnp.linalg.norm(new_ufo_pos - expected_pos)
    print(f"New UFO pos: {new_ufo_pos}")
    print(f"Distance from spawn: {distance}")
    assert distance < 0.1, f"UFO did not respawn at (77, 43). Got {new_ufo_pos}"
    
    # Check Pattern
    new_pattern = next_state.level.white_ufo_pattern_id[ufo_idx]
    print(f"New UFO pattern: {int(new_pattern)} ({WhiteUFOPattern(int(new_pattern)).name})")
    
    assert new_pattern == int(WhiteUFOPattern.IDLE), f"UFO pattern not reset to IDLE. Got {new_pattern}"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_ufo_respawn()
