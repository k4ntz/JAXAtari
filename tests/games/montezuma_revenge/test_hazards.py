import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezumarevenge import JaxMontezumaRevenge

def test_laser_inactive_cycle():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Laser is inactive when laser_cycle >= 92
    # Laser room is room 14
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(14, dtype=jnp.int32), state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(40, dtype=jnp.int32),
        player_y=jnp.array(20, dtype=jnp.int32),
        lasers_x=state.lasers_x.at[0].set(40),
        lasers_active=state.lasers_active.at[0].set(1),
        laser_cycle=jnp.array(95, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # Laser is inactive, player should not die
    assert state.death_timer == 0

