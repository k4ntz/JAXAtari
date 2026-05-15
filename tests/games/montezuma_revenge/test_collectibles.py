import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezumarevenge import JaxMontezumaRevenge

def test_collect_key():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    initial_keys = state.inventory[0]
    
    # Teleport to Room 14, where there is a key at (128, 7)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(14, dtype=jnp.int32), state, env.consts)
    
    # Place player at the key
    state = state.replace(
        player_x=jnp.array(128, dtype=jnp.int32),
        player_y=jnp.array(7, dtype=jnp.int32)
    )
    
    # Step NOOP to collect
    obs, state, reward, done, info = env.step(state, 0)
    
    assert state.inventory[0] == initial_keys + 1
    assert state.items_active[0] == 0
    assert reward == 100

def test_open_door():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 12 has doors at (56, 86) and (100, 86)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    
    # Place player in front of a door
    # Door is at (56, 86), width 4, height 38.
    # Player width 7, height 20.
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32), # next step will hit 51 if RIGHT (3)
        player_y=jnp.array(86, dtype=jnp.int32),
        inventory=state.inventory.at[0].set(3) # ensure we have keys
    )
    
    # Step RIGHT to hit the door
    obs, state, reward, done, info = env.step(state, 3)
    
    # Door should be opened
    assert state.inventory[0] == 2 # one less key
    assert state.doors_active[0] == 0
    assert reward == 300

def test_collect_sword():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Sword is in Room 13 at (12, 7)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(13, dtype=jnp.int32), state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(12, dtype=jnp.int32),
        player_y=jnp.array(7, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0)
    
    assert state.inventory[1] == 1
    assert reward == 1000

def test_kill_enemy_with_sword():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    initial_lives = state.lives
    
    # Give player a sword
    state = state.replace(
        inventory=state.inventory.at[1].set(1)
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
    # Enemy should be GONE
    assert state.enemies_active[0] == 0
    # Sword should be GONE from inventory
    assert state.inventory[1] == 0
    # Score should INCREASE
    assert reward >= 100

def test_door_without_key():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Remove all keys
    state = state.replace(
        inventory=state.inventory.at[0].set(0)
    )
    
    # Place player in front of the door in Room 12 (56, 86)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(86, dtype=jnp.int32)
    )
    
    # Step RIGHT (3) into the door
    obs, state, reward, done, info = env.step(state, 3)
    
    # Door should NOT be opened, and player should be blocked
    assert state.doors_active[0] == 1
    assert state.player_x == 50 # Blocked by wall logic

def test_collect_torch():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Torch is in Room 12 at (77, 7)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(77, dtype=jnp.int32),
        player_y=jnp.array(7, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0)
    
    # inventory[2] is the torch
    assert state.inventory[2] == 1
    assert state.items_active[0] == 0

