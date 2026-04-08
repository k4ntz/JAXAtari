import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma_revenge import JaxMontezumaRevenge
from jaxatari.games.montezuma_revenge.rooms import load_room

def test_torch_darkness_rendering():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # 1. Enter a dark room (Room 25 / ROOM_3_1) without a torch
    state = load_room(jnp.array(25, dtype=jnp.int32), state, env.consts)
    # Ensure no torch
    state = state.replace(inventory=state.inventory.at[2].set(0))
    
    # Render
    img_dark = env.render(state)
    
    # 2. Add a torch
    state = state.replace(inventory=state.inventory.at[2].set(1))
    img_light = env.render(state)
    
    # Check if light room has more non-black pixels in background area (y=47 to y=196)
    # Use a safe margin for y
    assert jnp.sum(img_light[50:190, ...] != 0) > jnp.sum(img_dark[50:190, ...] != 0)
    
    # 3. Verify interactive elements are STILL visible in dark room
    state = state.replace(inventory=state.inventory.at[2].set(0))
    
    # Move player to (50, 80) and check if visible
    state = state.replace(player_x=jnp.array(50, dtype=jnp.int32), player_y=jnp.array(80, dtype=jnp.int32))
    img_dark_player = env.render(state)
    # Player is at (50, 80+47) = (50, 127)
    # Player width 7, height 20.
    assert jnp.any(img_dark_player[127:147, 50:57] != 0)

    # Check lava (Room 31 has lava, let's switch to Room 31 for this check)
    state = load_room(jnp.array(31, dtype=jnp.int32), state, env.consts)
    state = state.replace(inventory=state.inventory.at[2].set(0))
    img_dark_lava = env.render(state)
    # Lava is at y=123 to 171
    assert jnp.any(img_dark_lava[123:171, :] != 0)
    
    # Room 31 has a ladder at x=72, y=6+47=53 to 44+47=91
    # Ladder color is orange in Room 31 (ORANGE_LADDER_ID)
    assert jnp.any(img_dark_lava[53:91, 72:88] != 0)

def test_dark_room_doors_and_platforms_hidden():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 26 (ROOM_3_2) has doors
    state = load_room(jnp.array(26, dtype=jnp.int32), state, env.consts)
    state = state.replace(inventory=state.inventory.at[2].set(0))
    
    # Ensure doors are active
    state = state.replace(doors_active=jnp.array([1, 1], dtype=jnp.int32))
    img_dark = env.render(state)
    
    # Check door positions. Door 0 is at x=20, y=7+47=54
    # Door width 4, height 38.
    assert jnp.all(img_dark[54:92, 20:24] == 0)
    
    # Give torch
    state = state.replace(inventory=state.inventory.at[2].set(1))
    img_light = env.render(state)
    assert jnp.any(img_light[54:92, 20:24] != 0)

    # Room 27 (ROOM_3_3) has platform (dropout floor)
    state = load_room(jnp.array(27, dtype=jnp.int32), state, env.consts)
    state = state.replace(inventory=state.inventory.at[2].set(0))
    state = state.replace(platforms_active=jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
                          platform_cycle=jnp.array(0, dtype=jnp.int32))
    
    img_dark_plat = env.render(state)
    # Platform is at x=32, y=47+47=94
    # Width 96, height 4.
    # Platform SHOULD be rendered even in the dark room without a torch
    assert jnp.any(img_dark_plat[94:98, 32:128] != 0)

def test_dark_room_gem_hidden():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 32 (ROOM_3_8) has 3 gems
    state = load_room(jnp.array(32, dtype=jnp.int32), state, env.consts)
    state = state.replace(inventory=state.inventory.at[2].set(0)) # No torch
    
    # Ensure items are active and type 1
    state = state.replace(items_active=jnp.array([1, 1, 1], dtype=jnp.int32),
                          items_type=jnp.array([1, 1, 1], dtype=jnp.int32))
    
    img_dark_gem = env.render(state)
    
    # First gem is at x=99, y=7+47=54
    # Gem width 6, height 8.
    # Gem SHOULD NOT be rendered in the dark room without a torch
    assert jnp.all(img_dark_gem[54:62, 99:105] == 0)
    
    # Give torch
    state = state.replace(inventory=state.inventory.at[2].set(1))
    img_light_gem = env.render(state)
    assert jnp.any(img_light_gem[54:62, 99:105] != 0)

