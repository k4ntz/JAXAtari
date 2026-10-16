import jax
import jax.numpy as jnp
from jaxtari.games.montezuma_revenge.rooms import load_room

def test_kill_spider_with_sword(montezuma_env):
    env = montezuma_env
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

def test_snake_cannot_be_killed_with_sword(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    # Use the room's real snake. The start room's fatal-contact cleanup is
    # specific to its skull and is not a valid setup for this interaction.
    state = load_room(jnp.array(18, dtype=jnp.int32), state, env.consts)
    
    initial_lives = state.lives
    
    # Give player a sword
    state = state.replace(
        inventory=state.inventory.at[1].set(1)
    )
    
    # Place the player at the native snake's contact position.
    state = state.replace(
        player_x=state.enemies_x[0],
        player_y=state.enemies_y[0]
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # The sword cannot protect against snakes and remains unused.
    assert state.death_timer > 0
    assert state.lives == initial_lives - 1
    assert state.enemies_active[0] == 1
    assert state.inventory[1] == 1

if __name__ == "__main__":
    test_kill_spider_with_sword()
    test_snake_cannot_be_killed_with_sword()
