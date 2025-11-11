import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_kangaroo import KangarooState
from jaxatari.games.kangaroo_levels import LevelConstants, Kangaroo_Level_1, Kangaroo_Level_2, Kangaroo_Level_3

# --- 1. Internal Mods (Group 1) ---

class NoMonkeyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable monkeys.
    This patches the environment's '_monkey_controller' method.
    """
    
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _monkey_controller(self, state: KangarooState, punching: chex.Array):
        """
        No-op override for _monkey_controller.
        """
        score_addition = jnp.zeros((), dtype=jnp.int32)
        
        return (
            state.level.monkey_states,       
            state.level.monkey_positions,    
            state.level.monkey_throw_timers, 
            score_addition,                  
            state.level.coco_positions,      
            state.level.coco_states,         
            jnp.array(False),                
        )

class NoFallingCoconutMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable the single falling coconut.
    This patches the environment's '_falling_coconut_controller' method.
    """
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _falling_coconut_controller(self, 
                                    state: KangarooState, 
                                    punching: chex.Array
                                    ):
        """
        No-op override for _falling_coconut_controller.
        """
        return (
            state.level.falling_coco_position, 
            state.level.falling_coco_dropping, 
            state.level.falling_coco_counter,  
            state.level.falling_coco_skip_update,
            jnp.zeros((), dtype=jnp.int32),     
        )

# --- 2. Post-Step Mod (Group 2) ---

class PinChildMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to pin the child kangaroo in place.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: KangarooState, new_state: KangarooState):
        """
        Called *after* the main step. Overwrites the child's
        position with its static starting position.
        """
        # Get the level constants to find the child's start position
        level_constants = self._env._get_level_constants(new_state.current_level)
        
        # Pin the child's position and velocity
        pinned_level_state = new_state.level._replace(
            child_position=level_constants.child_position, #
            child_velocity=jnp.array(0) # Also stop its velocity
        )
        
        return new_state._replace(level=pinned_level_state)


class RenderDebugInfo(JaxAtariInternalModPlugin):
    """
    Patches the render hook to draw the player's X,Y coords
    over the UI.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster, state: KangarooState):
        """
        This function patches the hook in KangarooRenderer.
        'self_env' is the JaxKangaroo instance, so we use
        'self_env.renderer.jr' to access the utils with correct config.
        """
        jr = self._env.renderer.jr
        masks = self._env.renderer.SHAPE_MASKS["score_digits"]
        
        # Draw Player X Position to frame
        x_digits = jr.int_to_digits(state.player.x, max_digits=3)
        raster = jr.render_label(raster, 10, 10, x_digits, masks, 8, 3)
        
        return raster


class ReplaceChildWithMonkeyMod(JaxAtariInternalModPlugin):
    """
    Replaces the child sprite with a monkey sprite.
    """

    asset_overrides = {
        "child": "ape"
    }

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster, state: KangarooState):
        """
        This function patches the hook in KangarooRenderer.
        'self_env' is the JaxKangaroo instance, so we use
        'self_env.renderer.jr' to access the utils with correct config.
        """
        
        # do nothing
        
        return raster

# Multiple Plugins to change bell to a fire (bundled into *modpack* in the kangaroo_mods.py file)

# --- MOD A: Replace Bell Sprite + Patch Animation ---

class ReplaceBellWithFlameMod(JaxAtariInternalModPlugin):
    """
    Replaces the 'bell' asset group with a new 'flame' asset group
    and patches the _draw_bell render hook to make it animate constantly.
    
    Expects 'flame_0.npy' and 'flame_1.npy' to exist.
    """
    
    # 1. Swap the assets (using Method 2: manually overriding a key in the asset_overrides dict)
    asset_overrides = {
        "bell": {
            'name': 'bell', 
            'type': 'group',
            'files': ['flame_0.npy', 'flame_1.npy']
        }
    }

    # 2. Patch the new animation hook
    @partial(jax.jit, static_argnums=(0,))
    def _draw_bell(self, raster: jnp.ndarray, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_bell method.
        This logic ignores the bell_animation timer and uses the
        step_counter for a constant flicker.
        """
        jr = self._env.renderer.jr
        
        # Animate using the global step counter for a constant flicker
        is_flicker_frame = (state.level.step_counter % 16) < 8
        flame_idx = jax.lax.select(is_flicker_frame, 1, 0) # 1 for frame 1, 0 for frame 0
        
        # We use "bell" as the key because our override mapped to it
        flame_mask = self._env.renderer.SHAPE_MASKS["bell"][flame_idx]
        flame_offset = self._env.renderer.FLIP_OFFSETS["bell"]
        
        # Keep original logic for *when* to draw
        should_draw_flame = (state.level.bell_position[0] != -1) & ~jnp.any(state.level.fruit_stages == 3)
        
        raster = jax.lax.cond(should_draw_flame,
            lambda r: jr.render_at(
                r, 
                state.level.bell_position[0].astype(int), 
                state.level.bell_position[1].astype(int), 
                flame_mask, 
                flip_horizontal=jnp.array(False), # No flipping
                flip_offset=flame_offset
            ),
            lambda r: r, 
            raster
        )
        return raster

# --- MOD B: Make the "Flame" (Bell) Lethal ---

class LethalFlameMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod that kills the player if they touch the bell.
    
    This runs *after* the main step and overrides the final
    state if a collision is detected.
    Using this instead of overriding the _bell_step method because kangaroo has a central function that handles collisions
    which might overwrite internal changes that happen before it executes.
    Other possibility would have been to either override the _bell_step method and the _lives_controller method or to insert a dedicated hook.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: KangarooState, new_state: KangarooState):
        """
        Called by the wrapper after the main step is complete.
        """
        # 1. Check for collision between player and bell
        is_colliding = self._env._entities_collide(
            new_state.player.x,
            new_state.player.y,
            self._env.consts.PLAYER_WIDTH,
            new_state.player.height,
            new_state.level.bell_position[0],
            new_state.level.bell_position[1],
            self._env.consts.BELL_WIDTH,
            self._env.consts.BELL_HEIGHT,
        )
        
        # 2. If colliding and not already crashing, set crash state
        #    and remove one life.
        return jax.lax.cond(
            is_colliding & ~new_state.player.is_crashing,
            
            # --- If True (Kill the player AND remove life) ---
            lambda state: self._trigger_crash_and_lose_life(state),
            
            # --- If False (Do nothing) ---
            lambda state: state,
            
            # Pass in the state
            new_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _trigger_crash_and_lose_life(self, state: KangarooState) -> KangarooState:
        """
        Returns a new state with the player set to 'is_crashing'
        and the life count decremented.
        """
        # 1. Set the player to crashing
        crashed_player_state = state.player._replace(
            is_crashing=jnp.array(True)
        )
        
        # 2. Decrement the life count
        new_lives = state.lives - 1
        
        # 3. Return the new state with both changes
        return state._replace(
            player=crashed_player_state,
            lives=new_lives
        )


class SpawnAtSecondLevelMod(JaxAtariInternalModPlugin):
    """Mod to spawn the player at the second level position."""
    # overwrite constants
    constants_overrides = {
        "PLAYER_START_Y": 52,
    }


# --- Ladder Modification Mods ---

def _remove_ladders(level_constants):
    """Remove all ladders by setting positions to invalid values."""
    num_ladders = level_constants.ladder_positions.shape[0]
    invalid_ladders = jnp.full((num_ladders, 2), -1, dtype=jnp.int32)
    invalid_sizes = jnp.zeros((num_ladders, 2), dtype=jnp.int32)
    
    return LevelConstants(
        ladder_positions=invalid_ladders,
        ladder_sizes=invalid_sizes,
        platform_positions=level_constants.platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=level_constants.bell_position,
        child_position=level_constants.child_position,
    )

class NoLaddersMod(JaxAtariInternalModPlugin):
    """
    Internal mod to remove all ladders from all levels.
    Uses constants_overrides to directly modify LEVEL_1, LEVEL_2, LEVEL_3.
    """
    # Create modified level constants with no ladders
    _level1_no_ladders = _remove_ladders(Kangaroo_Level_1)
    _level2_no_ladders = _remove_ladders(Kangaroo_Level_2)
    _level3_no_ladders = _remove_ladders(Kangaroo_Level_3)
    
    # Override constants directly
    constants_overrides = {
        "LEVEL_1": _level1_no_ladders,
        "LEVEL_2": _level2_no_ladders,
        "LEVEL_3": _level3_no_ladders,
    }


def _center_ladders(level_constants):
    """Center all ladders horizontally on the screen while keeping their y positions."""
    # Screen width is 160, ladder width is 8
    # Center x position: (160 - 8) / 2 = 76
    center_x = 76
    
    # Keep invalid positions (-1) as invalid
    is_valid = level_constants.ladder_positions[:, 0] >= 0
    
    # Create new positions: center x, keep original y
    original_y = level_constants.ladder_positions[:, 1]
    centered_positions = jnp.where(
        is_valid[:, jnp.newaxis],
        jnp.stack([jnp.full_like(original_y, center_x), original_y], axis=1),
        level_constants.ladder_positions  # Keep invalid positions as -1
    )
    
    return LevelConstants(
        ladder_positions=centered_positions,
        ladder_sizes=level_constants.ladder_sizes,
        platform_positions=level_constants.platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=level_constants.bell_position,
        child_position=level_constants.child_position,
    )

class CenterLaddersMod(JaxAtariInternalModPlugin):
    """
    Internal mod to center all ladder positions horizontally on the screen.
    All ladders will be perfectly aligned at x=76 (center of 160px screen).
    Uses constants_overrides to directly modify LEVEL_1, LEVEL_2, LEVEL_3.
    """
    
    # Create modified level constants with centered ladders
    _level1_centered = _center_ladders(Kangaroo_Level_1)
    _level2_centered = _center_ladders(Kangaroo_Level_2)
    _level3_centered = _center_ladders(Kangaroo_Level_3)
    
    # Override constants directly
    constants_overrides = {
        "LEVEL_1": _level1_centered,
        "LEVEL_2": _level2_centered,
        "LEVEL_3": _level3_centered,
    }