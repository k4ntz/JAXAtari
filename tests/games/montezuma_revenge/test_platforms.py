import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma_revenge import JaxMontezumaRevenge

def test_walking_on_platform():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    
    # Starting position (77, 26) is on a conveyor in room 12
    assert state.is_falling == 0
    assert state.is_jumping == 0
    
    # Walk right (Action 3)
    initial_x = state.player_x
    obs, state, reward, done, info = env.step(state, 3)
    
    assert state.player_x > initial_x
    assert state.is_falling == 0
    
def test_falling_off_platform():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    
    # In room 12, the middle platform/conveyor is from x=57 to x=103?
    # Conveyor starts at 60. Platform ends at 47.
    # Let's move player to x=50. 
    # player_mid_x = 53. check_platform around 53 (50..56) are all 0.
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(26, dtype=jnp.int32)
    )
    
    # Step NOOP (0) to fall
    obs, state, reward, done, info = env.step(state, 0)
    
    assert state.is_falling == 1
    assert state.player_y > 26

def test_jumping_on_platform():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    
    # Jump (Action 1 is FIRE, which triggers jump if on ground)
    obs, state, reward, done, info = env.step(state, 1)
    
    assert state.is_jumping == 1
    assert state.player_y < 26
    
    # Continue jumping until landing
    for _ in range(30):
        obs, state, reward, done, info = env.step(state, 0)
        if state.is_jumping == 0 and state.is_falling == 0:
            break
            
    assert state.player_y == 26
    assert state.is_jumping == 0
    assert state.is_falling == 0
