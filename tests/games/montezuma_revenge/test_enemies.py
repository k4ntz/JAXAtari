import jax
import jax.numpy as jnp

def test_enemy_bounce_bounds(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Use Room 4, enemy 0
    from jaxtari.games.montezuma_revenge.rooms import load_room
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    state = load_room(state.room_id, state, env.consts)
    
    # Place enemy right at its max bound, moving right
    # Movement is 1 pixel every 2 frames
    max_x = state.enemies_max_x[0]
    state = state.replace(
        enemies_x=state.enemies_x.at[0].set(max_x - 1),
        enemies_direction=state.enemies_direction.at[0].set(1),
        frame_count=jnp.array(0, dtype=jnp.int32) # ensure next step moves it
    )
    
    # Step 1: Moves right, hits boundary, reverses
    obs, state, reward, done, info = env.step(state, 0)
    # Step 2: Next frame, just waiting or moving left depending on frame count parity
    obs, state, reward, done, info = env.step(state, 0)
    
    # Enemy direction should have reversed to -1
    assert state.enemies_direction[0] == -1

def test_jump_over_enemy(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Place player and enemy overlapping in X, but player jumping over
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(60, dtype=jnp.int32), # Feet at 60 + 20 - 1 = 79
        is_jumping=jnp.array(1, dtype=jnp.int32),
        enemies_x=state.enemies_x.at[0].set(50),
        enemies_y=state.enemies_y.at[0].set(100), # Top at 100
        enemies_active=state.enemies_active.at[0].set(1)
    )
    
    obs, state, reward, done, info = env.step(state, 0)
    
    # Player should NOT die because they are vertically clear
    assert state.death_timer == 0

def test_skulls_synchronization(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Use Room 5 (Old 3), which has 2 skulls
    from jaxtari.games.montezuma_revenge.rooms import load_room
    state = state.replace(room_id=jnp.array(5, dtype=jnp.int32))
    state = load_room(state.room_id, state, env.consts)
    
    # Room 5 has 2 active skulls: index 0 and 1
    # Both ed are -1 initially.
    
    # Native pair reverses when the primary would cross x4 (holds x5).
    # Keep enemy 1 away from its own boundary to test shared direction.
    state = state.replace(
        enemies_x=state.enemies_x.at[0].set(5).at[1].set(50),
        enemies_direction=state.enemies_direction.at[0].set(-1).at[1].set(-1),
        enemies_active=state.enemies_active.at[0].set(1).at[1].set(1),
        native_frame_count=jnp.int32(0),
        room5_skull_spawn_pending=jnp.bool_(False),
    )
    
    # Tick: attempted x4 reverses both; neither moves on this tick.
    obs, state, reward, done, info = env.step(state, 0)
    
    # Enemy 0 should have flipped
    assert state.enemies_direction[0] == 1
    
    # Enemy 1 SHOULD also have flipped if synchronized
    assert state.enemies_direction[1] == 1, "Enemy 1 should have flipped direction together with Enemy 0"
