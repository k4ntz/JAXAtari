import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma2 import JaxMontezuma2
from jaxatari.games.montezuma2.rooms import load_room

def test_collect_amulet():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Amulet is in Room 28 at (17, 7)
    state = load_room(jnp.array(28, dtype=jnp.int32), state, env.consts)
    
    # Verify it is an amulet (type 2)
    assert state.items_type[0] == 2
    
    state = state.replace(
        player_x=jnp.array(17, dtype=jnp.int32),
        player_y=jnp.array(7, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # inventory[3] is the amulet
    assert state.inventory[3] == 1
    assert state.amulet_time == env.consts.AMULET_DURATION
    assert reward == 100

def test_amulet_neutralizes_enemies():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    initial_lives = state.lives
    
    # Activate amulet
    state = state.replace(
        amulet_time=jnp.array(100, dtype=jnp.int32),
        inventory=state.inventory.at[3].set(1)
    )
    
    # Place player and enemy (Skull) at the same position
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(100, dtype=jnp.int32),
        enemies_x=state.enemies_x.at[0].set(50),
        enemies_y=state.enemies_y.at[0].set(100),
        enemies_active=state.enemies_active.at[0].set(1),
        enemies_type=state.enemies_type.at[0].set(1) # Skull
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # Player should NOT die
    assert state.death_timer == 0
    assert state.lives == initial_lives
    # Enemy should still be active (amulet doesn't kill them, just neutralizes)
    assert state.enemies_active[0] == 1
    # Amulet time should decrease
    assert state.amulet_time == 99

def test_amulet_expiration():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    initial_lives = state.lives
    
    # Set amulet to expire in 1 frame
    state = state.replace(
        amulet_time=jnp.array(1, dtype=jnp.int32),
        inventory=state.inventory.at[3].set(1)
    )
    
    # Step NOOP
    obs, state, reward, done, info = env.step(state, 0)
    
    # Amulet should be gone
    assert state.amulet_time == 0
    assert state.inventory[3] == 0
    
    # Now enemies should be deadly
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(100, dtype=jnp.int32),
        enemies_x=state.enemies_x.at[0].set(50),
        enemies_y=state.enemies_y.at[0].set(100),
        enemies_active=state.enemies_active.at[0].set(1),
        enemies_type=state.enemies_type.at[0].set(1) # Skull
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # Player should die
    assert state.death_timer > 0
    assert state.lives == initial_lives - 1

if __name__ == "__main__":
    test_collect_amulet()
    test_amulet_neutralizes_enemies()
    test_amulet_expiration()
    print("All amulet tests passed!")
