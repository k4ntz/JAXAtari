import os
import time
import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_tutankham import TutankhamState
from jaxatari.games.jax_tutankham import can_walk_to, is_onscreen
from jaxatari.environment import JAXAtariAction as Action



# SIMPLE MODS
# 1. NightModeMod + NightModeStepMod
# 2. UpsideDownMod
# 3. ShrinkPlayerMod
# 4. MovingItemsMod
# 5. KnockbackMod

# HARD MODS
# 1. MimicMod + MimicStepMod
# 2. GhostMod
# 3. WhipMod


class NightModeMod(JaxAtariInternalModPlugin):
    """
    Draws the night_cover while the night_timer is in the active phase, which means it is currently night time. Only works with NightModeStepMod.
    """
    _TIME_CYCLE = 1000   # Total frames in a day/night loop
    _NIGHT_FRAMES = 300  # Duration of the night phase
    
    asset_overrides = {
        "night_cover": {
            "name": "night_cover",
            "type": "single",
            "file": "night_cover.npy",
        }
    }

    def _render_hook_night_mode(self, raster, state, camera_offset):
        """
        Rendering hook to render the night_cover.
        """
        jr = self._env.renderer.jr
        night_cover_mask = self._env.renderer.SHAPE_MASKS["night_cover"]

        # Calculate day time and if it is night time
        day_time = self._TIME_CYCLE - self._NIGHT_FRAMES
        night = state.night_timer > day_time # True for the first _NIGHT_FRAMES frames of the cycle

        def _draw_night(r):
            """
            Draw the night_cover between the ui and inside/over the game area.
            """
            return jr.render_at_clipped(
                r,
                0,
                36,
                night_cover_mask,
                flip_offset=self._env.consts.ZERO_FLIP,
            )

        return jax.lax.cond(night, _draw_night, lambda r: r, raster)


class NightModeStepMod(JaxAtariPostStepModPlugin):
    """
    Step mod for the day/night cycle. Only works with NightModeMod.
    """
    _TIME_CYCLE = 1000   # Total frames in a day/night loop
    _NIGHT_FRAMES = 300  # Duration of the night phase

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        """
        Update step for the day/night cycle.
        """
        #Count down and reset wehen the nighttimer hits 0.
        night_time_left = new_state.night_timer - 1
        day_time = self._TIME_CYCLE - self._NIGHT_FRAMES
        new_timer = jnp.where(night_time_left <= 0, jnp.int32(_TIME_CYCLE), night_time_left)

        # Resets the night timer to day time on player death or map transition.
        should_reset_to_day = (new_state.lives < prev_state.lives) | (new_state.level != prev_state.level)
        new_timer = jnp.where(should_reset_to_day, day_time, new_timer)

        return new_state.replace(night_timer=new_timer)


class MimicMod(JaxAtariInternalModPlugin):
    """
    Draws the mimic and the kill animation for it, when the mimic is active. Only works with MimicStepMod.
    """
    asset_overrides = {
        "mimic": {
            "name": "mimic",
            "type": "group",
            "files": ["mimic_01.npy", "mimic_00.npy"],
        },
        "kill_mimic": {
            "name": "kill_mimic",
            "type": "single",
            "file": "kill_mimic.npy",
        }
    }
    def _render_hook_mimic(self, raster, state, camera_offset):
        """
        Rendering hook to render the mimic.
        """
        jr = self._env.renderer.jr
        mimic_mask = self._env.renderer.SHAPE_MASKS["mimic"]
        kill_mask = self._env.renderer.SHAPE_MASKS["kill_mimic"]
        frame_idx = (state.step_counter // self._env.consts.ANIMATION_SPEED) % 2
        mimic_x = state.mimic_state[0]
        mimic_y = state.mimic_state[1]
        mimic_active = state.mimic_state[2]
        mimic_direction = state.mimic_state[3]
        mimic_death_timer = state.mimic_state[5]

        # Render normal mimic sprites.
        raster = jax.lax.cond(
            (mimic_active == 1) & (mimic_death_timer == -1),
            lambda r: jr.render_at_clipped(
                r,
                mimic_x,
                mimic_y - camera_offset + frame_idx, # +frame_idx to have the mimic sprite "jump"
                mimic_mask[frame_idx],
                flip_offset=self._env.consts.ZERO_FLIP,
                flip_horizontal=(mimic_direction == -2)
            ),
            lambda r: r,
            raster
        )
        
        # Render kill sprite when mimic is dying
        raster = jax.lax.cond(
            mimic_death_timer > 0,
            lambda r: jr.render_at_clipped(
                r,
                mimic_x,
                mimic_y - camera_offset,
                kill_mask,
                flip_offset=self._env.consts.ZERO_FLIP,
                flip_horizontal=False
            ),
            lambda r: r,
            raster
        )
        return raster


class MimicStepMod(JaxAtariPostStepModPlugin):
    """
    Step mod for the mimic. Only works with MimicMod. 
    Player has to run into the triggerbox of the selected item to trigger the mimic.
    """
    # Triggerbox around the mimic/selected item.
    TRIGGER_RANGE_X: int = 30
    TRIGGER_RANGE_Y: int = 30
    INVALID_TARGET_IDX: int = -1 # Index of the key
    MIMIC_SIZE: jnp.ndarray = jnp.array([8, 8], dtype=jnp.int32)
    MIMIC_SPEED: float = 0.8

    @partial(jax.jit, static_argnums=(0,))
    def _pick_target_idx(self, item_states, key):
        """
        Pick a random item to be the mimic. Return the index of the selecte item.
        """
        # check for active items. Usually any items because its only used at the levelstart. Necessary for future mod collisions.
        active_mask = item_states[:, 3] == self._env.consts.ACTIVE
        candidate_mask = active_mask & (jnp.arange(item_states.shape[0]) != 0)
        has_active_items = jnp.any(candidate_mask)
        # Normalize weights
        weights = candidate_mask.astype(jnp.float32)
        weights = weights / jnp.maximum(1.0, jnp.sum(weights))
        idx = jax.random.choice(key, item_states.shape[0], shape=(), p=weights)
        return jnp.where(has_active_items, idx.astype(jnp.int32), jnp.int32(self.INVALID_TARGET_IDX))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        """
        Pick a target item after reset. Gamestart or player death.
        """
        rng, subkey = jax.random.split(state.rng_key)
        target_idx = self._pick_target_idx(state.item_states, subkey)
        mimic_state = jnp.array([0, 0, 0, 0, target_idx, -1], dtype=jnp.int32)  # [x, y, active, direction, target_idx, death_timer]
        return obs, state.replace(mimic_state=mimic_state, rng_key=rng, mimic_subpixel=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: TutankhamState, new_state: TutankhamState) -> TutankhamState:
        """
        Update step for the mimic.
        Handles the movement of the mimic, the activation with the triggerbox and the level switch initialisation.
        """
        rng, subkey = jax.random.split(new_state.rng_key)

        # gather mimic state variables
        mimic_state = new_state.mimic_state
        mimic_x = mimic_state[0]
        mimic_y = mimic_state[1]
        mimic_active = mimic_state[2]
        mimic_direction = mimic_state[3]
        mimic_target_idx = mimic_state[4]
        mimic_death_timer = mimic_state[5]

        # bools for mimic reset and the repicking-logic
        level_changed = new_state.level != prev_state.level
        life_lost = new_state.lives < prev_state.lives

        # New random target item for the mimic on level transition.
        repicked_target_idx = self._pick_target_idx(new_state.item_states, subkey)
        target_idx = jnp.where(level_changed, repicked_target_idx, mimic_target_idx)

        # On death or level change: hard-reset the mimic.
        mimic_active = jnp.where(life_lost | level_changed, 0, mimic_active)
        mimic_death_timer = jnp.where(life_lost | level_changed, jnp.int32(-1), mimic_death_timer)
        mimic_x = jnp.where(level_changed, 0, mimic_x)
        mimic_y = jnp.where(level_changed, 0, mimic_y)
        mimic_direction = jnp.where(level_changed, 0, mimic_direction)

        # Trigger death animation
        mimic_active, mimic_death_timer, mimic_x, mimic_y = self._env.process_death_timer(mimic_active, mimic_death_timer, mimic_x, mimic_y)

        # clone the target item values
        target_item = new_state.item_states[target_idx]
        target_x = target_item[0]
        target_y = target_item[1]
        target_is_active = (target_item[3] == self._env.consts.ACTIVE)

        # Check if the player is in range of the target item
        in_range = ((jnp.abs(new_state.player_x - target_x) <= self.TRIGGER_RANGE_X) & (jnp.abs(new_state.player_y - target_y) <= self.TRIGGER_RANGE_Y))
        
        # Trigger when the mimic is fully dormant.
        trigger_now = (mimic_active == 0) & (mimic_death_timer == -1) & target_is_active & in_range

        # Set the target item to inactive when the mimic is triggered.
        new_item_states = jax.lax.cond(
            trigger_now,
            lambda items: items.at[target_idx, 3].set(self._env.consts.INACTIVE),
            lambda items: items,
            new_state.item_states,
        )

        # Update mimic state
        mimic_state = jnp.array(
            [
                jnp.where(trigger_now, target_x, mimic_x),
                jnp.where(trigger_now, target_y, mimic_y),
                jnp.where(trigger_now, 1, mimic_active),
                mimic_direction,
                target_idx,
                jnp.where(trigger_now, jnp.int32(-1), mimic_death_timer),  # death_timer
            ],
            dtype=jnp.int32,
        )

        # --- Mimic Movement ---
        rng, subkey_move = jax.random.split(rng)
        new_mimic_state, new_mimic_subpixel = self._mimic_movement(
            new_state, mimic_state, new_state.mimic_subpixel, subkey_move,
            new_state.camera_offset, new_state.level, new_state.player_x, new_state.player_y
        )

        new_state = new_state.replace(
            item_states=new_item_states, 
            mimic_state=new_mimic_state,
            mimic_subpixel=new_mimic_subpixel,
            rng_key=rng
        )
        # Resolve mimic collision with player and bullets
        new_state = self._resolve_mimic_collisions(new_state)
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def _mimic_movement(self, state: TutankhamState, mimic, mimic_subpixel, rng_key, camera_offset, level, player_x, player_y):
        """
        Handles the movement of the mimic. (Like the creatures in the base game)
        """
        mimic_x, mimic_y, mimic_active, mimic_direction, mimic_target_idx, mimic_death_timer = mimic
        
        lookup_x = self._env.consts.DIR_LOOKUP_X
        lookup_y = self._env.consts.DIR_LOOKUP_Y

        is_alive = (mimic_active == 1) & (mimic_death_timer == -1)

        # Natural patrol: randomly change direction
        rng_key, subkey_01, subkey_02 = jax.random.split(rng_key, 3)
        random_dir = jax.random.choice(subkey_01, jnp.array([-1, -2, 1, 2]))
        should_change = jax.random.bernoulli(subkey_02, p=0.08)
        new_direction = jnp.where(should_change, random_dir, mimic_direction)
        new_direction = jnp.where(mimic_direction == 0, random_dir, new_direction)

        # Chase player: if nearby
        dx = player_x - mimic_x
        dy = player_y - mimic_y
        player_near = (jnp.abs(dx) < self._env.consts.CREATURE_DETECTION_RANGE_X) & (jnp.abs(dy) < self._env.consts.CREATURE_DETECTION_RANGE_Y)
        horizontal_direction = jnp.where(dx >= 0, jnp.int32(-1), jnp.int32(-2))
        vertical_direction   = jnp.where(dy >= 0, jnp.int32(1),  jnp.int32(2))
        prefer_h = jnp.abs(dx) >= jnp.abs(dy)
        primary_dir = jnp.where(prefer_h, horizontal_direction, vertical_direction)
        secondary_dir = jnp.where(prefer_h, vertical_direction, horizontal_direction)
        next_x = mimic_x + lookup_x[primary_dir]
        next_y = mimic_y + lookup_y[primary_dir]
        _, _, primary_walkable = can_walk_to(self.MIMIC_SIZE, next_x, next_y, mimic_x, mimic_y, self._env.consts.VALID_POS_MAPS[level%4])
        toward_player = jnp.where(primary_walkable, primary_dir, secondary_dir)
        new_direction = jnp.where(player_near, toward_player, new_direction)
        
        # only update direction for alive mimics
        new_direction = jnp.where(is_alive, new_direction, mimic_direction)

        # --- movement ---
        speed = self.MIMIC_SPEED
        actual_speed, new_sub_acc = self._env.subpixel_accumulator(speed, mimic_subpixel)
        new_x = mimic_x + actual_speed * lookup_x[new_direction] * is_alive
        new_y = mimic_y + actual_speed * lookup_y[new_direction] * is_alive
        mimic_x, mimic_y, _ = can_walk_to(self.MIMIC_SIZE, new_x, new_y, mimic_x, mimic_y, self._env.consts.VALID_POS_MAPS[level%4])

        # Deactivate mimic if offscreen
        mimic_on_screen = is_onscreen(mimic_y, self.MIMIC_SIZE[1], camera_offset)
        active = jnp.where(mimic_on_screen, mimic_active, self._env.consts.INACTIVE)

        # Process death timer
        active, new_death_timer, mimic_x, mimic_y = self._env.process_death_timer(active, mimic_death_timer, mimic_x, mimic_y)

        return jnp.array([mimic_x, mimic_y, active, new_direction, mimic_target_idx, new_death_timer], dtype=jnp.int32), new_sub_acc

    @partial(jax.jit, static_argnums=(0,))
    def _resolve_mimic_collisions(self, state: TutankhamState) -> TutankhamState:
        """Unified AABB collision handler for the mimic.
        Triggers death animation (death_timer=15) if either the player or a bullet hits.
        On player hit: respawns player (lives-1).
        On bullet hit: deactivates the bullet."""
        mimmic_state = state.mimic_state
        mimic_x = mimmic_state[0]
        mimic_y = mimmic_state[1]
        mimic_alive = (mimmic_state[2] == 1) & (mimmic_state[5] == -1)

        bullet_state = state.bullet_state
        bull_active = bullet_state[3]
        
        # 1. Check Hits
        player_hit = self._env.check_entity_collision(
            state.player_x, state.player_y, self._env.consts.PLAYER_SIZE,
            mimic_x, mimic_y, self.MIMIC_SIZE
        ) & mimic_alive
        
        bullet_hit = self._env.check_entity_collision(
            bullet_state[0], bullet_state[1], self._env.consts.BULLET_SIZE,
            mimic_x, mimic_y, self.MIMIC_SIZE
        ) & mimic_alive & (bull_active == 1)

        # 2. Compute Respawn (for player hit)
        (resp_x, resp_y, resp_bull, resp_creat, resp_lives, resp_spwn, resp_dir) = self._env.respawn_player(
            state.player_y, state.lives, state.level)

        # 3. Apply Consequences
        # Player hit takes precedence for state reset, bullet hit just zeros the bullet
        new_player_x                = jnp.where(player_hit, resp_x, state.player_x)
        new_player_y                = jnp.where(player_hit, resp_y, state.player_y)
        new_lives                   = jnp.where(player_hit, resp_lives, state.lives)
        new_creature_states         = jnp.where(player_hit, resp_creat, state.creature_states)
        new_creature_subpixels      = jnp.where(player_hit, jnp.zeros_like(state.creature_subpixels), state.creature_subpixels)
        new_last_creature_spawn     = jnp.where(player_hit, resp_spwn, state.last_creature_spawn)
        new_last_movement_action = jnp.where(player_hit, resp_dir, state.last_movement_action)

        # Bullet state: reset if player hit (unconditional respawn rules) OR if bullet specifically hit the mimic
        new_bullet_state = jnp.where(player_hit, resp_bull,
                                    jnp.where(bullet_hit, jnp.zeros(5, dtype=jnp.int32), bullet_state))

        # Mimic state: trigger death if either hit
        new_mimic_state = jnp.where(player_hit, mimmic_state.at[5].set(0), mimmic_state)
        new_mimic_state = jnp.where(bullet_hit, mimmic_state.at[5].set(15), mimmic_state)

        # 4. Update Score (Mimic destroyed by bullet awards points from its mimicked item)
        target_idx = mimmic_state[4]
        mimicked_item_type = state.item_states[target_idx, 2]
        mimic_points = self._env.consts.ITEM_POINTS[mimicked_item_type]
        new_score = state.tutankham_score + jnp.where(bullet_hit, mimic_points, 0)

        return state.replace(
            player_x=new_player_x,
            player_y=new_player_y,
            lives=new_lives,
            bullet_state=new_bullet_state,
            creature_states=new_creature_states,
            creature_subpixels=new_creature_subpixels,
            last_creature_spawn=new_last_creature_spawn,
            last_movement_action=new_last_movement_action,
            mimic_state=new_mimic_state,
            tutankham_score=new_score
        )


class ShrinkPlayerMod(JaxAtariInternalModPlugin):
    """
    Shrinks the player's hitbox and sprite.
    """
    constants_overrides = {
        "PLAYER_SIZE": jnp.array([3, 5], dtype=jnp.int32),
        "PLAYER_SPEED": 2, # Zoom zoom :3
        "TELEPORTER_HEIGHT": 8,
    }
    asset_overrides = {
        "player": {
            "name": "player",
            "type": "group",
            "files": ["shrink_player_idle.npy", "shrink_player_key_idle.npy"],
        },
        "player_move": {
            "name": "player_move",
            "type": "group",
            "files": ["shrink_player_move_00.npy", "shrink_player_move_01.npy","shrink_player_key_move_00.npy", "shrink_player_key_move_01.npy"],
        }
    }


class WhipMod(JaxAtariInternalModPlugin):
    """
    Adds the Whip, a medium-range weapon that kills enemies in a certain area around the player, hard reduction of the ammotimer and usage cooldown.
    """
    attribute_overrides = {
        "ACTION_SET": jnp.array([
            Action.NOOP, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
            Action.RIGHTFIRE, Action.LEFTFIRE, Action.UPFIRE, Action.DOWNFIRE
        ])
    }

    asset_overrides = {
        "ui_whip": {
            "name": "ui_whip",
            "type": "single",
            "file": "ui_whip.npy",
        },
        "whip_effect": {
            "name": "whip_effect",
            "type": "group",
            "files": ["whip_horizontal_left.npy", "whip_bl.npy", "whip_vertical_bottom.npy", "whip_br.npy", "whip_horizontal_right.npy", "whip_tr.npy", "whip_vertical_top.npy", "whip_tl.npy"],
        }
    }

    def _render_hook_whip(self, raster, state, camera_offset):
        """
        Renders the whip icon in the ui and the whip effect.
        """
        jr = self._env.renderer.jr
        ui_whip_mask = self._env.renderer.SHAPE_MASKS["ui_whip"]
        whip_mask = self._env.renderer.SHAPE_MASKS["whip_effect"]
        
        # Visible when whip is ready to use
        is_ready = (state.whip_timer == 0) & (state.ammunition_timer > 0)

        # Render whip icon in the ui
        raster = jnp.where(is_ready, jr.render_at_clipped(
                raster,
                136, 
                185, 
                ui_whip_mask,
                flip_offset=self._env.consts.ZERO_FLIP,
            ), raster)

        # Whip offsets for each frame sprites for the whip for same distance around the player.
        whip_offset_x = jnp.array([-9, -8, 0, 5, 5, 5, 1, -6])
        whip_offset_y = jnp.array([  2, 8, 8, 8, 2, -6, -9, -8])

        # Whip animation lasts for 16 frames.
        is_whipping = state.whip_timer > 184
        whip_frame = jnp.abs(state.whip_timer - 200) % 8

        # Render whip effect
        raster = jax.lax.cond(
            is_whipping,
                lambda r: jr.render_at_clipped(
                    r,
                    state.player_x + whip_offset_x[whip_frame], 
                    state.player_y + whip_offset_y[whip_frame] - camera_offset, 
                    whip_mask[whip_frame],
                    flip_offset=self._env.consts.ZERO_FLIP,
                ),
                lambda r: r,
                raster
            )
            
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def whip_step(self, whip_timer, creature_states, player_x, player_y, ammunition_timer, action, tutankham_score):
        """
        Updates the whip state and kills creatures in range, increase score for each creature killed by its value.
        """        
        # Advance whip_timer if active
        new_whip_timer = jnp.where(whip_timer > 0, whip_timer - 1, 0)
        
        # Start whipping if DOWNFIRE pressed.
        start_whip = (action == Action.DOWNFIRE) & (whip_timer == 0) & (ammunition_timer > 0)
        
        # set new whip timer
        new_whip_timer = jnp.where(start_whip, 200, new_whip_timer)
        
        # Ammo penalty: Reduce ammo by 25%
        new_ammo = jnp.where(start_whip, (ammunition_timer * 3) // 4, ammunition_timer)

        def kill_creature(creature):
            """
            Checks if a creature is in range of the whip and kills it if it is.
            """
            dist_sq = (creature[0] - player_x)**2 + (creature[1] - player_y)**2
            in_range = dist_sq <= 45**2 # 45 pixel radius
            is_active = (creature[3] == 1) & (creature[5] == -1)
            should_kill = start_whip & in_range & is_active
            new_death = jnp.where(should_kill, 15, creature[5])
            kill_score = jnp.where(should_kill, self._env.consts.CREATURE_POINTS[creature[2]], 0)
            return creature.at[5].set(new_death), kill_score
        
        # Apply kill_creature to all creatures
        new_creatures, kill_scores = jax.vmap(kill_creature)(creature_states)

        # Calculate new score after killing creatures
        new_tutankham_score = tutankham_score + jnp.sum(kill_scores)
        
        return new_whip_timer, new_creatures, new_ammo, new_tutankham_score


class KnockbackMod(JaxAtariInternalModPlugin):
    """
    Knocks back creatures when hit by bullets instead of killing it.
    """

    @partial(jax.jit, static_argnums=(0,))
    def resolve_bullet_collisions(self, creature_states, bullet_state, level):
        """
        Resolves bullet collisions with creatures.
        """
        active_mask = (creature_states[:, 3] == self._env.consts.ACTIVE) & (creature_states[:, 5] == -1)

        def bullet_hits_creature(creature):
            """
            Checks if a creature is in range of the bullet.
            """
            creature_type = creature[2]
            return self._env.check_entity_collision(
                bullet_state[0], bullet_state[1], self._env.consts.BULLET_SIZE,
                creature[0], creature[1], self._env.consts.CREATURE_SIZES[creature_type],
            )

        # check bullet-creature collisions and determine which creature was hit
        bullet_hits = jax.vmap(bullet_hits_creature)(creature_states)
        bullet_hits = bullet_hits & active_mask & (bullet_state[3] == 1)
        any_bullet_hit = jnp.any(bullet_hits)
        first_bullet_hit = (jnp.cumsum(bullet_hits) == 1) & bullet_hits
        hit_creature_idx = jnp.argmax(first_bullet_hit)

        # check for new_x wall collision and set new x coordinate
        knockbacked_x, _, knockbackable = can_walk_to(self._env.consts.CREATURE_SIZES[creature_states[hit_creature_idx, 2]], creature_states[hit_creature_idx, 0] + 15 * bullet_state[2], creature_states[hit_creature_idx, 1], creature_states[hit_creature_idx, 0], creature_states[hit_creature_idx, 1], self._env.consts.VALID_POS_MAPS[level%4])
        new_creature_x = jnp.where(any_bullet_hit & knockbackable, knockbacked_x, creature_states[hit_creature_idx, 0])
        new_creature_states = creature_states.at[hit_creature_idx, 0].set(new_creature_x)
        new_bullet_state = jnp.where(any_bullet_hit, jnp.zeros(5, dtype=bullet_state.dtype), bullet_state)

        return new_creature_states, new_bullet_state


class GhostMod(JaxAtariInternalModPlugin):
    """
    Adds ghosts to the game as a new spawnable creature that can move through walls and tracks the player down permanently.
    """    
    asset_overrides = {
        "creature_00": {"name": "creature_00", "type": "group", "files": ["creature_snake_00.npy", "creature_scorpion_00.npy", "creature_bat_00.npy", "creature_turtle_00.npy", "creature_jackel_00.npy", "creature_condor_00.npy", "creature_lion_00.npy", "creature_moth_00.npy", "creature_virus_00.npy", "creature_monkey_00.npy", "creature_mysteryweapon_00.npy", "ghost_00.npy"]},
        "creature_01": {"name": "creature_01", "type": "group", "files": ["creature_snake_01.npy", "creature_scorpion_01.npy", "creature_bat_01.npy", "creature_turtle_01.npy", "creature_jackel_01.npy", "creature_condor_01.npy", "creature_lion_01.npy", "creature_moth_01.npy", "creature_virus_01.npy", "creature_monkey_01.npy", "creature_mysteryweapon_01.npy", "ghost_01.npy"]},
        "kill_sprites": {"name": "kill_sprites", "type": "group", "files": ["kill_snake_00.npy", "kill_scorpion_00.npy", "kill_bat_00.npy", "kill_turtle_00.npy", "kill_jackel_00.npy", "kill_condor_00.npy", "kill_lion_00.npy", "kill_moth_00.npy", "kill_virus_00.npy", "kill_monkey_00.npy", "kill_mysteryweapon_00.npy", "kill_ghost.npy"]},
    }

    constants_overrides = {

    "CREATURE_SIZES": jnp.array([
        [8, 8],   # SNAKE
        [8, 8],   # SCORPION
        [8, 8],   # BAT
        [8, 8],   # TURTLE
        [8, 8],   # JACKEL  
        [8, 7],   # CONDOR
        [8, 8],   # LION
        [8, 8],   # MOTH
        [8, 6],   # VIRUS
        [8, 8],   # MONKEY
        [8, 8],   # MYSTERY_WEAPON
        [8, 8]    # GHOST (MOD)
    ], dtype=jnp.int32),

    "CREATURE_SPEED": jnp.array(
        [0.65, 0.75, 0.95, 0.5, 0.75, 0.95, 0.85, 0.95, 0.75, 0.85, 0.95, 0.4], # Ghost has slow speed of 0.4
        dtype=jnp.float32),

    "CREATURE_POINTS": jnp.array(
        [1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 0], # Ghost gives zero points
        dtype=jnp.int32),

    "MAP_CREATURES": jnp.array([ # Ghost can spawn on any map
        [0, 1, 2, 11], 
        [3, 4, 5, 11],
        [0, 6, 7, 11],
        [8, 9, 10, 11]  
    ], dtype=jnp.int32)
    }

    @partial(jax.jit, static_argnums=(0,))
    def move_creature(self, creature, creature_subpixel, rng_key, camera_offset, level, player_x, player_y):
        creature_x, creature_y, creature_type, active, direction, death_timer = creature

        lookup_x = self._env.consts.DIR_LOOKUP_X
        lookup_y = self._env.consts.DIR_LOOKUP_Y

        # creature state is active and creature is not in it's death animation
        is_alive = (active == self._env.consts.ACTIVE) & (death_timer == -1)

        # --- pathing ---
        # Natural patrol: randomly change direction
        rng_key, subkey_01, subkey_02 = jax.random.split(rng_key, 3)
        random_dir = jax.random.choice(subkey_01, jnp.array([-1, -2, 1, 2]))
        should_change = jax.random.bernoulli(subkey_02, p=0.08)
        new_direction = jnp.where(should_change, random_dir, direction)
        new_direction = jnp.where(direction == 0, random_dir, new_direction)

        is_ghost = (creature_type == 11) # int 11 for ghost creature

        # Chase player if nearby (ghost always chases, others only within detection range)
        dx = player_x - creature_x
        dy = player_y - creature_y
        player_near = is_ghost | (
            (jnp.abs(dx) < self._env.consts.CREATURE_DETECTION_RANGE_X) &
            (jnp.abs(dy) < self._env.consts.CREATURE_DETECTION_RANGE_Y)
        )
        horizontal_direction = jnp.where(dx >= 0, jnp.int32(-1), jnp.int32(-2))
        vertical_direction   = jnp.where(dy >= 0, jnp.int32(1),  jnp.int32(2))
        prefer_h      = jnp.abs(dx) >= jnp.abs(dy)
        primary_dir   = jnp.where(prefer_h, horizontal_direction, vertical_direction)
        secondary_dir = jnp.where(prefer_h, vertical_direction, horizontal_direction)
        next_x = creature_x + lookup_x[primary_dir]
        next_y = creature_y + lookup_y[primary_dir]
        _, _, primary_walkable = can_walk_to(self._env.consts.CREATURE_SIZES[creature_type], next_x, next_y, creature_x, creature_y, self._env.consts.VALID_POS_MAPS[level%4])
        # ghost ignores wall collisions: always takes primary direction
        toward_player = jnp.where(is_ghost | primary_walkable, primary_dir, secondary_dir)
        new_direction = jnp.where(player_near, toward_player, new_direction)

        # only update direction for alive creatures
        new_direction = jnp.where(is_alive, new_direction, direction)

        # --- movement ---
        speed = self._env.consts.CREATURE_SPEED[creature_type.astype(jnp.int32)]
        actual_speed, new_subpixel = self._env.subpixel_accumulator(speed, creature_subpixel)
        new_x = creature_x + actual_speed * lookup_x[new_direction] * is_alive
        new_y = creature_y + actual_speed * lookup_y[new_direction] * is_alive
        # ghost ignores wall collisions: always moves to new position
        ghost_x = new_x.astype(jnp.int32)
        ghost_y = new_y.astype(jnp.int32)
        walked_x, walked_y, _ = can_walk_to(self._env.consts.CREATURE_SIZES[creature_type], new_x, new_y, creature_x, creature_y, self._env.consts.VALID_POS_MAPS[level%4])
        creature_x = jnp.where(is_ghost, ghost_x, walked_x)
        creature_y = jnp.where(is_ghost, ghost_y, walked_y)

        # Deactivate creature if offscreen
        creature_on_screen = is_onscreen(creature_y, self._env.consts.CREATURE_SIZES[creature_type][1], camera_offset)
        active = jnp.where(creature_on_screen, active, self._env.consts.INACTIVE)

        active, new_death_timer, creature_x, creature_y = self._env.process_death_timer(active, death_timer, creature_x, creature_y)

        return jnp.array([creature_x, creature_y, creature_type, active, new_direction, new_death_timer], dtype=jnp.int32), new_subpixel


class UpsideDownMod(JaxAtariInternalModPlugin):
    """
    Flips the map upside down, and switches the player's start poins goal positions and flips the checkpoints.
    """
    constants_overrides = {
        "MAP_CHECKPOINTS": jnp.array([
            # MAP 1 
            [
                [588, 800, 18, 684],
                [405, 587, 80, 586],
                [201, 404, 12, 403],
                [0,   200, 78, 199],
            ],
            # MAP 2 
            [
                [573, 800, 19, 634],
                [425, 572, 24, 572],
                [261, 426, 78, 426],
                [0,   260, 78, 259],
            ],
            # MAP 3 
            [
                [553, 800,  107, 715],
                [401, 552,  98,  550],
                [269, 400,  78,  396],
                [0,   268,  39,  248],
            ],
            # MAP 4 
            [
                [531, 800, 77,  719],
                [391, 532, 119, 531],
                [204, 392, 18,  391],
                [0 ,  203, 30,  203],
            ],
        ], dtype=jnp.int32),

        "MAP_GOAL_POSITIONS": jnp.array([
            [[134, 61]],  # MAP 1
            [[136, 60]],  # MAP 2
            [[16,  93]],  # MAP 3
            [[82,  95]]   # MAP 4
        ], dtype=jnp.int32)
    }


class MovingItemsMod(JaxAtariInternalModPlugin):
    """
    Makes the items roam around the map.
    """
    
    # Add item direction to the initial item states (to calculate movement) [x, y, item_type, active, direction]
    constants_overrides = {
        "MAP_ITEMS": jnp.array([
            # MAP 1
            [
                [51, 87, 0, 1, 0],   # KEY_MAP1=0       
                [99, 183, 5, 1, 0],  # CROWN_02_MAP1=5
                [68, 262, 2, 1, 0],  # RING_MAP1=2
                [7, 311, 3, 1, 0],   # RUBY_MAP1=3
                [93, 382, 4, 1, 0],  # CHALICE_MAP1=4
                [18, 494, 1, 1, 0],  # CROWN_01_MAP1=1
                [0, 0, 0, 0, 0],     # Padding
            ],
            # MAP 2
            [
                [21, 272, 6, 1, 0],  # KEY_MAP2=6
                [44, 155, 8, 1, 0],  # CROWN_MAP2=8
                [128, 98, 7, 1, 0],  # RING_MAP2=7
                [37, 406, 9, 1, 0],  # EMERALD_MAP2=9
                [91, 482, 10, 1, 0], # GOBLET_MAP2=10
                [23, 547, 11, 1, 0], # BUST_MAP2=11
                [0, 0, 0, 0, 0],     # Padding
            ],
            # MAP 3
            [
                [22, 411, 12, 1, 0], # KEY_MAP3=12
                [15, 173, 14, 1, 0], # RING_MAP3=14
                [128, 98, 13, 1, 0], # TRIDENT_MAP3=13
                [17, 278, 15, 1, 0], # HERB_MAP3=15
                [108, 323, 16, 1, 0],# DIAMOND_MAP3=16
                [27, 656, 17, 1, 0], # CANDELABRA_MAP3=17
                [0, 0, 0, 0, 0],     # Padding
            ],
            # MAP 4
            [
                [144, 110, 18, 1, 0], # KEY_MAP4=18
                [125, 221, 19, 1, 0], # RING_MAP4=19
                [117, 269, 20, 1, 0], # AMULET_MAP4=20
                [19, 326, 21, 1, 0],  # FAN_MAP4=21
                [55, 510, 23, 1, 0],  # ZIRCON_MAP4=23
                [110, 401, 22, 1, 0], # CRYSTAL_MAP4=22
                [66, 607, 24, 1, 0],  # DAGGER_MAP4=24
            ],
        ], dtype=jnp.int32),
    }

    @partial(jax.jit, static_argnums=(0,))
    def item_step(self, item_states, level, rng_key):
        """
        Moves active items using random-walk pathing
        """

        def move_item(item, rng_key):
            """
            Moves a single active item using random-walk pathing
            """
            item_x, item_y, item_type, active, direction = item
            
            # Natural patrol: randomly change direction
            change_probability = 0.08
            possible_directions = jnp.array([-1, -2, 1, 2]) # right, left, down, up

            rng_key, subkey_01, subkey_02 = jax.random.split(rng_key, 3)
            random_dir = jax.random.choice(subkey_01, possible_directions)
            should_change = jax.random.bernoulli(subkey_02, p=change_probability)

            new_direction = jnp.where(should_change, random_dir, direction)
            new_direction = jnp.where(direction == 0, random_dir, new_direction)

            # Mapping array where the index matches the direction value
            lookup_x = jnp.array([0, 0,  0, -1, 1])
            lookup_y = jnp.array([0, 1, -1, 0,  0])
            x_direction = lookup_x[new_direction]
            y_direction = lookup_y[new_direction]

            # Move item in new direction
            new_x = item_x +  x_direction * active
            new_y = item_y +  y_direction * active           
            item_x, item_y, is_walkable = can_walk_to(self._env.consts.ITEM_SIZES[item_type], new_x, new_y, item_x, item_y, self._env.consts.VALID_POS_MAPS[level%4])

            return jnp.array([item_x, item_y, item_type, active, new_direction], dtype=jnp.int32), rng_key

        # Apply movement to all items
        keys = jax.random.split(rng_key, len(item_states) + 1)
        rng_key = keys[0]
        new_item_states, _ = jax.vmap(move_item)(item_states, keys[1:])

        return new_item_states, rng_key

    