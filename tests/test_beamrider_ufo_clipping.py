
import jax
import jax.numpy as jnp
from jaxatari.games.jax_beamrider import JaxBeamrider, BeamriderConstants

def test_ufo_clipping():
    env = JaxBeamrider()
    
    # Reset
    obs, state = env.reset(jax.random.key(0))
    
    ufo_idx = 0
    
    # Test 1: Below top lane, should NOT be clipped to LEFT_CLIP_PLAYER (27)
    # We'll put it at x=10, y=100
    test_pos_below = jnp.array([10.0, 100.0])
    
    current_ufo_pos = state.level.white_ufo_pos
    new_white_ufo_pos = current_ufo_pos.at[0, ufo_idx].set(test_pos_below[0])
    new_white_ufo_pos = new_white_ufo_pos.at[1, ufo_idx].set(test_pos_below[1])
    
    # Set velocity to 0 to see where it ends up after clipping logic
    new_level = state.level._replace(
        white_ufo_pos=new_white_ufo_pos,
        white_ufo_vel=jnp.zeros_like(state.level.white_ufo_vel)
    )
    state_below = state._replace(level=new_level)
    
    obs, next_state_below, reward, done, info = env.step(state_below, 0)
    
    pos_after_below = jnp.array([next_state_below.level.white_ufo_pos[0][ufo_idx], next_state_below.level.white_ufo_pos[1][ufo_idx]])
    print(f"Pos below after step: {pos_after_below}")
    
    # Should NOT be clipped to 27. It should stay at 10.0
    assert pos_after_below[0] == 10.0, f"UFO was clipped below top lane! Got x={pos_after_below[0]}"
    
    # Test 2: On top lane, SHOULD be clipped to LEFT_CLIP_PLAYER (27)
    # We'll put it at x=10, y=43
    test_pos_top = jnp.array([10.0, 43.0])
    
    new_white_ufo_pos_top = current_ufo_pos.at[0, ufo_idx].set(test_pos_top[0])
    new_white_ufo_pos_top = new_white_ufo_pos_top.at[1, ufo_idx].set(test_pos_top[1])
    
    new_level_top = state.level._replace(
        white_ufo_pos=new_white_ufo_pos_top,
        white_ufo_vel=jnp.zeros_like(state.level.white_ufo_vel)
    )
    state_top = state._replace(level=new_level_top)
    
    obs, next_state_top, reward, done, info = env.step(state_top, 0)
    
    pos_after_top = jnp.array([next_state_top.level.white_ufo_pos[0][ufo_idx], next_state_top.level.white_ufo_pos[1][ufo_idx]])
    print(f"Pos top after step: {pos_after_top}")
    
    # SHOULD be clipped to 27.0
    assert pos_after_top[0] == 27.0, f"UFO was NOT clipped on top lane! Got x={pos_after_top[0]}"

    print("Test Passed!")

if __name__ == "__main__":
    test_ufo_clipping()
