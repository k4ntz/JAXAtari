import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_kangaroo import KangarooState, PlayerState, LevelState, KangarooConstants
from jaxatari.games.kangaroo_levels import LevelConstants, Kangaroo_Level_1, Kangaroo_Level_2, Kangaroo_Level_3

# --- 1. Internal Mods (Group 1) ---
class NoBellMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable the Bell.
    Patches '_bell_step'.
    """
    
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _bell_step(self, state: KangarooState):
        """
        No-op override for _bell_step.
        Returns 0 for the timer and False for the respawn flag, 
        effectively disabling the bell mechanics.
        """
        return jnp.zeros_like(state.level.bell_timer), jnp.array(False)


    @partial(jax.jit, static_argnums=(0,))
    def _draw_bell(self, raster: jnp.ndarray, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_bell method.
        Draws a static sprite (no animation) shifted 4 pixels up.
        """

        return raster


class NoFruitMod(JaxAtariInternalModPlugin):
    """
    Internal mod to remove Fruits.
    Patches '_fruits_step'.
    """   

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _fruits_step(self, state: KangarooState):
        """
        Override for _fruits_step to remove fruits.
        """
        # We must still call _bell_step because the environment logic 
        # normally chains these together.
        bell_timer, _ = self._env._bell_step(state)

        return (
            jnp.zeros((), dtype=jnp.int32),                             # Score addition
            jnp.zeros_like(state.level.fruit_actives, dtype=jnp.bool_), # Set actives to False (Hides them visually)
            state.level.fruit_stages,                                   # Keep stages (irrelevant since inactive)
            bell_timer                                                  # Pass through the bell timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_single_fruit(self, i, raster, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_fruits method.
        Does not draw any fruits.
        """
        return raster


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


class NoThrownCoconutMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable thrown coconuts.
    This patches the environment's '_update_coco_state' method to prevent 
    coconuts from transitioning to active states (1 or 2).
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_coco_state(
        self,
        old_m_state: chex.Array,
        new_m_state: chex.Array,
        old_m_timer: chex.Array,
        new_m_timer: chex.Array,
        c_state: chex.Array,
        c_pos_x: chex.Array,
    ) -> chex.Array:
        """
        Override to prevent coconut state updates.
        Returns 0 (non-existent) regardless of monkey state.
        """
        return jnp.array(0, dtype=jnp.int32)


class AlwaysHighCoconutMod(JaxAtariInternalModPlugin):
    """
    Internal mod to force coconuts to always spawn at the 'head' (high) position.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_coco_positions(
        self,
        new_c_state: chex.Array,
        old_c_state: chex.Array,
        stepc: chex.Array,
        old_c_pos: chex.Array,
        new_m_pos: chex.Array,
        spawn_position: chex.Array,
    ) -> chex.Array:
        
        return jnp.where(
            new_c_state == 2,
            # --- Flight Logic (Unchanged) ---
            jnp.where(
                stepc % 2 == 0,
                jnp.array([old_c_pos[0] - 2, old_c_pos[1]]),
                old_c_pos,
            ),
            # --- Spawn Logic (Modified) ---
            jnp.where(
                (new_c_state == 1) & (old_c_state == 0),
                jnp.array(
                    [
                        new_m_pos[0] - 6,
                        new_m_pos[1] - 5 
                    ]
                ),
                old_c_pos,
            ),
        )

class FirstLevelOnlyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to force the game to always stay on level 1.
    This patches the environment's '_level_transition_controller' method.
    """
    conflicts_with = ["second_level_only", "third_level_only"]

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _next_level(self, state: KangarooState):
        RESET_AFTER_TICKS = 256

        counter = state.levelup_timer
        counter_start = state.level_finished & (counter == 0)
        counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
        reset_timer_done = counter == RESET_AFTER_TICKS
        counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

        reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
        levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

        current_level = jnp.where(levelup, 1, state.current_level)

        return current_level, counter, reset_coords, levelup


class SecondLevelOnlyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to force the game to always stay on level 2.
    This patches the environment's '_level_transition_controller' method.
    """
    conflicts_with = ["first_level_only", "third_level_only", "center_ladders", "invert_ladders", "flame_trap"]

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _next_level(self, state: KangarooState):
        RESET_AFTER_TICKS = 256

        counter = state.levelup_timer
        counter_start = state.level_finished & (counter == 0)
        counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
        reset_timer_done = counter == RESET_AFTER_TICKS
        counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

        reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
        levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

        current_level = jnp.where(levelup, 2, state.current_level)

        return current_level, counter, reset_coords, levelup

    @partial(jax.jit, static_argnums=(0,))
    def reset_level(self, next_level=1) -> KangarooState:
        next_level = 2
        level_constants = Kangaroo_Level_2
        main_consts = KangarooConstants()
        new_state = KangarooState(
            player=PlayerState(
                x=jnp.array(main_consts.PLAYER_START_X),
                y=jnp.array(main_consts.PLAYER_START_Y),
                vel_x=jnp.array(0),
                is_crouching=jnp.array(False),
                is_jumping=jnp.array(False),
                is_climbing=jnp.array(False),
                jump_counter=jnp.array(0),
                orientation=jnp.array(1),
                jump_base_y=jnp.array(main_consts.PLAYER_START_Y),
                landing_base_y=jnp.array(main_consts.PLAYER_START_Y),
                height=jnp.array(main_consts.PLAYER_HEIGHT),
                jump_orientation=jnp.array(0),
                climb_base_y=jnp.array(main_consts.PLAYER_START_Y),
                climb_counter=jnp.array(0),
                punch_left=jnp.array(False),
                punch_right=jnp.array(False),
                cooldown_counter=jnp.array(0),
                chrash_timer=jnp.array(0),
                is_crashing=jnp.array(False),
                last_stood_on_platform_y=jnp.array(1000),
                walk_animation=jnp.array(0),
                punch_counter=jnp.array(0),
                needs_release=jnp.array(False),
            ),
            level=LevelState(
                bell_position=level_constants.bell_position,
                bell_timer=jnp.array(0),
                fruit_positions=level_constants.fruit_positions,
                fruit_actives=jnp.ones(3, dtype=jnp.bool_),
                fruit_stages=jnp.zeros(3, dtype=jnp.int32),
                ladder_positions=level_constants.ladder_positions,
                ladder_sizes=level_constants.ladder_sizes,
                platform_positions=level_constants.platform_positions,
                platform_sizes=level_constants.platform_sizes,
                child_position=level_constants.child_position,
                child_timer=jnp.array(0),
                child_velocity=jnp.array(1),
                timer=jnp.array(2000),  # to be modified
                falling_coco_position=jnp.array([13, -1]),
                falling_coco_dropping=jnp.array(False),
                falling_coco_counter=jnp.array(0),
                falling_coco_skip_update=jnp.array(False),
                step_counter=jnp.array(0),
                monkey_states=jnp.zeros(4, dtype=jnp.int32),
                monkey_positions=jnp.array([[152, 5], [152, 5], [152, 5], [152, 5]]),
                monkey_throw_timers=jnp.zeros(4, dtype=jnp.int32),
                spawn_protection=jnp.array(True),
                coco_positions=jnp.array(
                    [[-10, -10], [-10, -10], [-10, -10], [-10, -10]]
                ),
                coco_states=jnp.zeros(4, dtype=jnp.int32),
                spawn_position=jnp.array(False),
                bell_animation=jnp.array(0),
            ),
            score=jnp.array(0),
            current_level=next_level,
            level_finished=jnp.array(False),
            levelup_timer=jnp.array(0),
            reset_coords=jnp.array(False),
            levelup=jnp.array(False),
            lives=jnp.array(3),
        )
        return new_state

class ThirdLevelOnlyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to force the game to always stay on level 3.
    This patches the environment's '_level_transition_controller' method.
    """
    conflicts_with = ["first_level_only", "second_level_only", "center_ladders", "invert_ladders", "flame_trap"]

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _next_level(self, state: KangarooState):
        RESET_AFTER_TICKS = 256

        counter = state.levelup_timer
        counter_start = state.level_finished & (counter == 0)
        counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
        reset_timer_done = counter == RESET_AFTER_TICKS
        counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

        reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
        levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

        current_level = jnp.where(levelup, 3, state.current_level)

        return current_level, counter, reset_coords, levelup
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_level(self, next_level=1) -> KangarooState:
        next_level = 3
        level_constants = Kangaroo_Level_3
        main_consts = KangarooConstants()
        new_state = KangarooState(
            player=PlayerState(
                x=jnp.array(main_consts.PLAYER_START_X),
                y=jnp.array(main_consts.PLAYER_START_Y),
                vel_x=jnp.array(0),
                is_crouching=jnp.array(False),
                is_jumping=jnp.array(False),
                is_climbing=jnp.array(False),
                jump_counter=jnp.array(0),
                orientation=jnp.array(1),
                jump_base_y=jnp.array(main_consts.PLAYER_START_Y),
                landing_base_y=jnp.array(main_consts.PLAYER_START_Y),
                height=jnp.array(main_consts.PLAYER_HEIGHT),
                jump_orientation=jnp.array(0),
                climb_base_y=jnp.array(main_consts.PLAYER_START_Y),
                climb_counter=jnp.array(0),
                punch_left=jnp.array(False),
                punch_right=jnp.array(False),
                cooldown_counter=jnp.array(0),
                chrash_timer=jnp.array(0),
                is_crashing=jnp.array(False),
                last_stood_on_platform_y=jnp.array(1000),
                walk_animation=jnp.array(0),
                punch_counter=jnp.array(0),
                needs_release=jnp.array(False),
            ),
            level=LevelState(
                bell_position=level_constants.bell_position,
                bell_timer=jnp.array(0),
                fruit_positions=level_constants.fruit_positions,
                fruit_actives=jnp.ones(3, dtype=jnp.bool_),
                fruit_stages=jnp.zeros(3, dtype=jnp.int32),
                ladder_positions=level_constants.ladder_positions,
                ladder_sizes=level_constants.ladder_sizes,
                platform_positions=level_constants.platform_positions,
                platform_sizes=level_constants.platform_sizes,
                child_position=level_constants.child_position,
                child_timer=jnp.array(0),
                child_velocity=jnp.array(1),
                timer=jnp.array(2000),  # to be modified
                falling_coco_position=jnp.array([13, -1]),
                falling_coco_dropping=jnp.array(False),
                falling_coco_counter=jnp.array(0),
                falling_coco_skip_update=jnp.array(False),
                step_counter=jnp.array(0),
                monkey_states=jnp.zeros(4, dtype=jnp.int32),
                monkey_positions=jnp.array([[152, 5], [152, 5], [152, 5], [152, 5]]),
                monkey_throw_timers=jnp.zeros(4, dtype=jnp.int32),
                spawn_protection=jnp.array(True),
                coco_positions=jnp.array(
                    [[-10, -10], [-10, -10], [-10, -10], [-10, -10]]
                ),
                coco_states=jnp.zeros(4, dtype=jnp.int32),
                spawn_position=jnp.array(False),
                bell_animation=jnp.array(0),
            ),
            score=jnp.array(0),
            current_level=next_level,
            level_finished=jnp.array(False),
            levelup_timer=jnp.array(0),
            reset_coords=jnp.array(False),
            levelup=jnp.array(False),
            lives=jnp.array(3),
        )
        return new_state

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

class ReplaceCoconutWithFireball(JaxAtariInternalModPlugin):
    asset_overrides = {
        "coconut": {
            'name': 'coconut',
            'type': 'single',
            'file': 'fireball.npy'
        }
    }
    constants_overrides = {
        "THROWN_COCONUT_WIDTH": 16,
        "THROWN_COCONUT_HEIGHT": 12,
    }

class ReplaceCoconutWithWasp(JaxAtariInternalModPlugin):
    asset_overrides = {
        "coconut": {
            'name': 'coconut',
            'type': 'single',
            'file': 'wasp.npy'
        }
    }
    constants_overrides = {
        "THROWN_COCONUT_WIDTH": 16,
        "THROWN_COCONUT_HEIGHT": 12,
    }

class ReplaceCoconutWithHoneyBee(JaxAtariInternalModPlugin):
    asset_overrides = {
        "coconut": {
            'name': 'coconut',
            'type': 'single',
            'file': 'honey_bee.npy'
        }
    }
    constants_overrides = {
        "THROWN_COCONUT_WIDTH": 16,
        "THROWN_COCONUT_HEIGHT": 12,
    }

class ReplaceMonkeyWithTankMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['tank_15x8.npy']
        }
    }

class ReplaceMonkeyWithChickenMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['chicken.npy']
        }
    }

class ReplaceMonkeyWithDangerSignMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['danger_sign.npy']
        }
    }

class ReplaceMonkeyWithDragonMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['dragon.npy']
        }
    }

class ReplaceMonkeyWithPolarbearMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['polarbear.npy']
        }
    }

class ReplaceMonkeyWithSnakeMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "ape": {
            'name': 'ape',
            'type': 'group',
            'files': ['snake.npy']
        }
    }

# --- MOD A: Replace Bell Sprite + Patch Animation ---
class ReplaceBellWithCactusMod(JaxAtariInternalModPlugin):
    """
    Replaces the 'bell' asset group with a new 'cactus' asset group.
    
    Expects 'cactus.npy' to exist.
    """
    
    # 1. Swap the assets (using Method 2: manually overriding a key in the asset_overrides dict)
    asset_overrides = {
        "bell": {
            'name': 'bell', 
            'type': 'group',
            'files': ['cactus_tall.npy']
        }
    }
    @partial(jax.jit, static_argnums=(0,))
    def _draw_bell(self, raster: jnp.ndarray, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_bell method.
        Draws a static sprite (no animation) shifted 4 pixels up.
        """
        jr = self._env.renderer.jr
        
        # CHANGED: Removed flicker logic. Hardcoded to index 0 for a static image.
        flame_idx = 0 
        
        # We use "bell" as the key because our override mapped to it
        flame_mask = self._env.renderer.SHAPE_MASKS["bell"][flame_idx]
        flame_offset = self._env.renderer.FLIP_OFFSETS["bell"]
        
        # Keep original logic for *when* to draw
        should_draw_flame = (state.level.bell_position[0] != -1) & ~jnp.any(state.level.fruit_stages == 3)
        
        # CHANGED: Adjusted Y position logic inside render_at
        raster = jax.lax.cond(should_draw_flame,
            lambda r: jr.render_at(
                r, 
                state.level.bell_position[0].astype(int), 
                # CHANGED: Subtract 4 from Y to move it up
                state.level.bell_position[1].astype(int) - 8, 
                flame_mask, 
                flip_horizontal=jnp.array(False),
                flip_offset=flame_offset
            ),
            lambda r: r, 
            raster
        )
        return raster



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

class ReplaceBellWithDangerSignMod(JaxAtariInternalModPlugin):
    """
    Replaces the 'bell' asset group with a new 'danger_sign' asset group.
    
    Expects 'danger_sign.npy' to exist.
    """
    
    # 1. Swap the assets (using Method 2: manually overriding a key in the asset_overrides dict)
    asset_overrides = {
        "bell": {
            'name': 'bell', 
            'type': 'group',
            'files': ['danger_sign.npy']
        }
    }
    @partial(jax.jit, static_argnums=(0,))
    def _draw_bell(self, raster: jnp.ndarray, state: KangarooState):
        """
        Overrides the KangarooRenderer._draw_bell method.
        Draws a static sprite (no animation) shifted 4 pixels up.
        """
        jr = self._env.renderer.jr
        
        # CHANGED: Removed flicker logic. Hardcoded to index 0 for a static image.
        flame_idx = 0 
        
        # We use "bell" as the key because our override mapped to it
        flame_mask = self._env.renderer.SHAPE_MASKS["bell"][flame_idx]
        flame_offset = self._env.renderer.FLIP_OFFSETS["bell"]
        
        # Keep original logic for *when* to draw
        should_draw_flame = (state.level.bell_position[0] != -1) & ~jnp.any(state.level.fruit_stages == 3)
        
        # CHANGED: Adjusted Y position logic inside render_at
        raster = jax.lax.cond(should_draw_flame,
            lambda r: jr.render_at(
                r, 
                state.level.bell_position[0].astype(int), 
                state.level.bell_position[1].astype(int), 
                flame_mask, 
                flip_horizontal=jnp.array(False),
                flip_offset=flame_offset
            ),
            lambda r: r, 
            raster
        )
        return raster

class ReplaceLadderWithChainMod(JaxAtariInternalModPlugin):
    """
    Replaces the ladder sprites with grey chain sprites.
    Chains are drawn as a 4-pixel wide alternating pattern (Inner vs Outer pixels).
    """
    NEW_CHAIN_COLOR = (128, 128, 128) # Grey

    # Create the procedural asset (1x1 pixel RGBA sprite)
    custom_color_rgba = jnp.array([[[
        NEW_CHAIN_COLOR[0],  # R
        NEW_CHAIN_COLOR[1],  # G
        NEW_CHAIN_COLOR[2],  # B
        255  # Alpha (fully opaque)
    ]]], dtype=jnp.uint8)

    # Add via asset_overrides
    asset_overrides = {
        'custom_chain_color': {
            'name': 'custom_chain_color',
            'type': 'procedural',
            'data': custom_color_rgba
        }
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_ladders(self, raster: jnp.ndarray, state):
        """
        Draws chains: A 4px wide pattern alternating between inner connectors and outer loops.
        """
        # 1. Access Data from the Environment State
        positions = state.level.ladder_positions
        sizes = state.level.ladder_sizes
        
        # Access the renderer's calculated color ID for the chain
        chain_color = self._env.renderer.COLOR_TO_ID.get(self.NEW_CHAIN_COLOR, 0)
            
        # 2. Get dimensions
        h, w = raster.shape
        
        # 3. Create the meshgrid for vectorization
        yy, xx = jnp.mgrid[:h, :w]

        # 4. Define Visual Constants
        chain_visual_width = 4
        half_width = 2
        segment_height = 2  # Height of one link segment

        def _create_single_chain_mask(pos, size):
            """Generates a boolean mask for a single chain."""
            # Only draw if the ladder exists (x != -1)
            is_active = pos[0] != -1
            
            # Geometry
            x, y = pos
            hitbox_width, height = size

            y -= 8
            height += 12
            
            # Calculate Center
            center_x = x + (hitbox_width // 2)
            
            # Start drawing x (shift left by 2 to center the 4px chain)
            draw_x_start = center_x - half_width
            
            # --- Bounding Box Logic ---
            dx = xx - draw_x_start
            dy = yy - y
            
            # Check if pixel is within the 4px wide x height bounding box
            in_box = (dx >= 0) & (dx < chain_visual_width) & \
                     (dy >= 0) & (dy < height)
            
            # --- Chain Pattern Logic ---
            segment_idx = dy // segment_height
            
            # Logic:
            # Inner pixels (dx=1, dx=2) represent the vertical connector
            # Outer pixels (dx=0, dx=3) represent the sides of the link loop
            is_inner_pixel = (dx == 1) | (dx == 2)
            is_outer_pixel = (dx == 0) | (dx == 3)
            
            # Alternating Pattern:
            # Even segments (0, 2...) -> Draw Inner (Connector)
            # Odd segments  (1, 3...) -> Draw Outer (Loop Sides)
            segment_is_even = (segment_idx % 2) == 0
            
            is_chain_pixel = (segment_is_even & is_inner_pixel) | \
                             (~segment_is_even & is_outer_pixel)
            
            # Combine
            return in_box & is_chain_pixel & is_active

        # 5. Vectorize
        all_masks = jax.vmap(_create_single_chain_mask)(positions, sizes)
        
        # 6. Collapse
        combined_mask = jnp.any(all_masks, axis=0)
        
        # 7. Apply to Raster
        return jnp.where(combined_mask, jnp.asarray(chain_color, dtype=raster.dtype), raster)


class ReplaceLadderWithRopeMod(JaxAtariInternalModPlugin):
    """
    Replaces the ladder sprites with rope sprites.
    Ropes are drawn as a 2-pixel wide zig-zag pattern centered on the original ladder position.
    """
    NEW_LADDER_COLOR = (149, 75, 49)

    # Create the procedural asset (1x1 pixel RGBA sprite)
    custom_color_rgba = jnp.array([[[
        NEW_LADDER_COLOR[0],  # R
        NEW_LADDER_COLOR[1],  # G
        NEW_LADDER_COLOR[2],  # B
        255  # Alpha (fully opaque)
    ]]], dtype=jnp.uint8)

    # Add via asset_overrides (can add new assets, not just override existing ones!)
    asset_overrides = {
        'custom_ladder_color': {
            'name': 'custom_ladder_color',
            'type': 'procedural',
            'data': custom_color_rgba
        }, 
        "kangaroo": {
            'name': 'kangaroo', 
            'type': 'group',
            'files': ['kangaroo.npy', 'kangaroo_dead.npy', 'kangaroo_rope_climb.npy', 'kangaroo_ducking.npy', 'kangaroo_jump.npy', 'kangaroo_boxing.npy', 'kangaroo_walk.npy', 'kangaroo_jump_high.npy']
        }
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_ladders(self, raster: jnp.ndarray, state: KangarooState):
        """
        Draws ropes: a 2-pixel wide zig-zag pattern.
        """
        # 1. Access Data from the Environment State
        positions = state.level.ladder_positions
        sizes = state.level.ladder_sizes
        
        # Access the renderer's calculated color ID for the ladder/rope
        rope_color = self._env.renderer.COLOR_TO_ID.get(self.NEW_LADDER_COLOR, 0)
            
        # 2. Get dimensions
        h, w = raster.shape
        
        # 3. Create the meshgrid for vectorization
        # yy corresponds to row indices, xx to column indices
        yy, xx = jnp.mgrid[:h, :w]

        # 4. Define Visual Constants
        rope_visual_width = 2 
        segment_height = 4  # How many pixels tall one 'twist' of the rope is

        def _create_single_rope_mask(pos, size):
            """Generates a boolean mask for a single rope."""
            # Only draw if the ladder exists (x != -1)
            is_active = pos[0] != -1
            
            # Geometry
            x, y = pos
            hitbox_width, height = size

            y -= 8
            height += 8
            
            # Calculate Center: The rope hangs in the middle of the ladder hitbox
            center_x = x + (hitbox_width // 2)
            
            # Start drawing x (shift left by 1 to center the 2px rope)
            draw_x_start = center_x - 1
            
            # --- Bounding Box Logic ---
            # Determine relative coordinates to the top-left of the rope
            dx = xx - draw_x_start
            dy = yy - y
            
            # Check if pixel is within the 2px wide x height bounding box
            in_box = (dx >= 0) & (dx < rope_visual_width) & \
                     (dy >= 0) & (dy < height)
            
            # --- Zig-Zag Pattern Logic ---
            # We determine the 'segment' index based on Y position.
            # Example: Rows 0-1 are segment 0, Rows 2-3 are segment 1.
            segment_idx = dy // segment_height
            
            # Check if the pixel is on the left side (dx=0) or right side (dx=1)
            is_left_pixel = (dx == 0)
            
            # Pattern: 
            # Even segments (0, 2, 4...) -> Draw Left Pixel
            # Odd segments (1, 3, 5...)  -> Draw Right Pixel
            segment_is_even = (segment_idx % 2) == 0
            
            # Draw if (Even Segment AND Left Pixel) OR (Odd Segment AND Right Pixel)
            # This is equivalent to checking if the boolean values are equal
            is_rope_pixel = (segment_is_even == is_left_pixel)
            
            # Combine: Must be in bounding box, match the pattern, and be active
            return in_box & is_rope_pixel & is_active

        # 5. Vectorize: Apply logic to all ladders simultaneously
        # resulting shape: (Num_Ladders, Height, Width)
        all_masks = jax.vmap(_create_single_rope_mask)(positions, sizes)
        
        # 6. Collapse: Combine all ladder masks into one single layer
        combined_mask = jnp.any(all_masks, axis=0)
        
        # 7. Apply to Raster: Where mask is True, paint the rope color
        return jnp.where(combined_mask, jnp.asarray(rope_color, dtype=raster.dtype), raster)


        # Vectorize over all ropes in the array
        all_masks = jax.vmap(_create_single_rope_mask)(pos_scaled, size_scaled)
        
        # Combine all rope masks into one layer
        combined_mask = jnp.logical_or.reduce(all_masks, axis=0)
        
        # Apply to raster
        return jnp.where(combined_mask, jnp.asarray(self.LADDER_COLOR_ID, raster.dtype), raster)


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


class SpawnOnSecondFloorMod(JaxAtariInternalModPlugin):
    """Mod to spawn the player on the second level position."""
    # overwrite constants
    constants_overrides = {
        "PLAYER_START_Y": 52,
    }


# --- Ladder Modification Mods ---
def _center_ladders(level_constants):
    """Center all ladders horizontally on the screen while keeping their y positions."""
    # Screen width is 160, ladder width is 8
    SCREEN_WIDTH = 160
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
    
    # Center also the platforms accordingly
    platform_original_y = level_constants.platform_positions[:, 1]
    platform_centered_x = (SCREEN_WIDTH - level_constants.platform_sizes[:, 0]) // 2
    centered_platform_positions = jnp.where(
        (level_constants.platform_sizes[:, 0] < 128)[:, None], # Only center small platforms (x > 16)
        jnp.stack([jnp.full_like(platform_original_y, platform_centered_x), platform_original_y], axis=1),
        level_constants.platform_positions
    )

    return LevelConstants(
        ladder_positions=centered_positions,
        ladder_sizes=level_constants.ladder_sizes,
        platform_positions=centered_platform_positions,
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

def _invert_ladders(level_constants):
    """Invert ladder positions horizontally on the screen."""
    # Screen width is 160, ladder width is 8
    screen_width = 160
    ladder_width = 8
    
    # Invert x positions: new_x = screen_width - ladder_width - original_x
    inverted_x = screen_width - ladder_width - level_constants.ladder_positions[:, 0]
    
    inverted_positions = jnp.stack([inverted_x, level_constants.ladder_positions[:, 1]], axis=1)

    inverted_platform_x = screen_width - level_constants.platform_positions[:, 0] - level_constants.platform_sizes[:, 0]

    inverted_platform_positions = jnp.stack([inverted_platform_x, level_constants.platform_positions[:, 1]], axis=1)

    inverted_bell_x = screen_width - level_constants.bell_position[0] - 6  # Bell width is 6

    inverted_bell_position = jnp.array([inverted_bell_x, level_constants.bell_position[1]])
    
    return LevelConstants(
        ladder_positions=inverted_positions,
        ladder_sizes=level_constants.ladder_sizes,
        platform_positions=inverted_platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=inverted_bell_position,
        child_position=level_constants.child_position,
    )

class InvertLaddersMod(JaxAtariInternalModPlugin):
    """
    Internal mod to invert all ladder positions horizontally on the screen.
    Uses constants_overrides to directly modify LEVEL_1, LEVEL_2, LEVEL_3.
    """
    
    # Create modified level constants with inverted ladders
    _level1_inverted = _invert_ladders(Kangaroo_Level_1)
    _level2_inverted = _invert_ladders(Kangaroo_Level_2)
    _level3_inverted = _invert_ladders(Kangaroo_Level_3)
    
    # Override constants directly
    constants_overrides = {
        "LEVEL_1": _level1_inverted,
        "LEVEL_2": _level2_inverted,
        "LEVEL_3": _level3_inverted,
    }


# Create modified level constants with four ladders
def _add_fourth_ladder(level_constants, level_number):
    # Get existing ladder positions and sizes
    ladder_positions = level_constants.ladder_positions
    ladder_sizes = level_constants.ladder_sizes
    
    # Identify the third ladder (index 2)
    fourth_ladder_pos = jnp.array([132, 84])  # Default position for the fourth ladder
    fourth_ladder_size = jnp.array([8, 36])  # Standard ladder size
    

    # Append the fourth ladder
    new_ladder_positions = jnp.vstack([ladder_positions, fourth_ladder_pos])
    new_ladder_sizes = jnp.vstack([ladder_sizes, fourth_ladder_size])
    return LevelConstants(
        ladder_positions=new_ladder_positions,
        ladder_sizes=new_ladder_sizes,
        platform_positions=level_constants.platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=level_constants.bell_position,
        child_position=level_constants.child_position,
    )

class FourLaddersMod(JaxAtariInternalModPlugin):
    """
    Internal mod to add a fourth ladder to each level.
    The fourth ladder is placed symmetrically to the third ladder.
    """
    _level1_with_four = _add_fourth_ladder(Kangaroo_Level_1, 0)
    _level2_with_four = _add_fourth_ladder(Kangaroo_Level_2, 1)
    _level3_with_four = _add_fourth_ladder(Kangaroo_Level_3, 2)
    
    constants_overrides = {
        "LEVEL_1": _level1_with_four,
        "LEVEL_2": _level2_with_four,
        "LEVEL_3": _level3_with_four,
    }


def flame_trap(level_constants):
    """Moves the flame to the first floor position."""
    return LevelConstants(
        ladder_positions=level_constants.ladder_positions,
        ladder_sizes=level_constants.ladder_sizes,
        platform_positions=level_constants.platform_positions,
        platform_sizes=level_constants.platform_sizes,
        fruit_positions=level_constants.fruit_positions,
        bell_position=jnp.array([100, 113]),  # First floor position
        child_position=level_constants.child_position,
    )

class FlameTrapMod(JaxAtariInternalModPlugin):
    """
    Internal mod to place the flame (bell) on the way to the fruit at each level.
    """

    _level1_centered = _center_ladders(Kangaroo_Level_1)
    _level2_centered = _center_ladders(Kangaroo_Level_2)
    _level3_centered = _center_ladders(Kangaroo_Level_3)
    constants_overrides = {
        "LEVEL_1": flame_trap(_level1_centered),
        "LEVEL_2": flame_trap(_level2_centered),
        "LEVEL_3": flame_trap(_level3_centered),
    }