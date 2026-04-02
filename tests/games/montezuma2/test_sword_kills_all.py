import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma2 import JaxMontezuma2

def test_kill_spider_with_sword():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    initial_lives = state.lives
    
    # Give player a sword
    state = state.replace(
        inventory=state.inventory.at[1].set(1)
    )
    
    # Place player and enemy (Spider = type 3) at the same position
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(100, dtype=jnp.int32),
        enemies_x=state.enemies_x.at[0].set(50),
        enemies_y=state.enemies_y.at[0].set(100),
        enemies_active=state.enemies_active.at[0].set(1),
        enemies_type=state.enemies_type.at[0].set(3) # Spider
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # Player should NOT die
    assert state.death_timer == 0
    assert state.lives == initial_lives
    # Enemy should be GONE
    assert state.enemies_active[0] == 0
    # Sword should be GONE from inventory
    assert state.inventory[1] == 0

def test_kill_snake_with_sword():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    initial_lives = state.lives
    
    # Give player a sword
    state = state.replace(
        inventory=state.inventory.at[1].set(1)
    )
    
    # Place player and enemy (Snake = type 4) at the same position
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(100, dtype=jnp.int32),
        enemies_x=state.enemies_x.at[0].set(50),
        enemies_y=state.enemies_y.at[0].set(100),
        enemies_active=state.enemies_active.at[0].set(1),
        enemies_type=state.enemies_type.at[0].set(4) # Snake
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # Player should NOT die
    assert state.death_timer == 0
    assert state.lives == initial_lives
    # Enemy should be GONE
    assert state.enemies_active[0] == 0
    # Sword should be GONE from inventory
    assert state.inventory[1] == 0

if __name__ == "__main__":
    test_kill_spider_with_sword()
    test_kill_snake_with_sword()
