
import jax
import jax.numpy as jnp
from jaxatari.games.jax_beamrider import JaxBeamrider

def test_beamrider_sector_limit():
    """Verify that Beamrider ends after sector 14."""
    env = JaxBeamrider()
    key = jax.random.key(0)
    obs, state = env.reset(key)
    
    # Manually set sector to 14 and lives to 3
    state = state._replace(sector=jnp.array(14, dtype=jnp.int32), lives=jnp.array(3, dtype=jnp.int32))
    
    # Check done before beating sector 14
    done = env._get_done(state)
    assert not done, "Game should not be done at sector 14"
    
    # Manually set sector to 15
    state_after_beating = state._replace(sector=jnp.array(15, dtype=jnp.int32))
    done_after = env._get_done(state_after_beating)
    assert done_after, "Game should be done after beating sector 14 (at sector 15)"

def test_beamrider_lives_limit():
    """Verify that Beamrider ends when lives are 0."""
    env = JaxBeamrider()
    key = jax.random.key(0)
    obs, state = env.reset(key)
    
    # Manually set lives to 0
    state = state._replace(lives=jnp.array(0, dtype=jnp.int32))
    
    done = env._get_done(state)
    assert done, "Game should be done when lives are 0"
