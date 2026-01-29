import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from flax import struct
from jaxatari.modification import AutoDerivedConstants


def _get_popup_durations() -> jnp.ndarray:
    return jnp.array([254, 134, 66, 32, 32], dtype=jnp.int32)


def _create_static_procedural_sprites(screen_height: int, screen_width: int) -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Create a black background
    bg_data = jnp.zeros((screen_height, screen_width, 4), dtype=jnp.uint8)
    bg_data = bg_data.at[0, 0, 3].set(255)  # Add one black, opaque pixel
    return {
        'background': bg_data
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Asterix.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    Note: Background is created procedurally and will be added in renderer.
    """
    asterix_item_names = ['CAULDRON', 'HELMET', 'SHIELD', 'LAMP']
    obelix_item_names = ['APPLE', 'FISH', 'WILD_BOAR_LEG', 'MUG']
    points_names = ['POINTS50', 'POINTS100', 'POINTS200', 'POINTS300', 'POINTS400', 'POINTS500']
    
    config_list = [
        {'name': 'STAGE', 'type': 'single', 'file': 'STAGE.npy'},
        {'name': 'TOP', 'type': 'single', 'file': 'TOP.npy'},
        {'name': 'BOTTOM', 'type': 'single', 'file': 'BOTTOM.npy'},
        
        {'name': 'ASTERIX_SPRITES', 'type': 'group', 'files': [
            'ASTERIX_LEFT.npy', 'ASTERIX_RIGHT.npy',
            'ASTERIX_LEFT_HIT.npy', 'ASTERIX_RIGHT_HIT.npy',
        ]},
        {'name': 'OBELIX_SPRITES', 'type': 'group', 'files': [
            'OBELIX_LEFT.npy', 'OBELIX_RIGHT.npy',
            'OBELIX_LEFT_HIT.npy', 'OBELIX_RIGHT_HIT.npy',
        ]},
        
        {'name': 'LYRE_LEFT', 'type': 'single', 'file': 'LYRE_LEFT.npy'},
        {'name': 'LYRE_RIGHT', 'type': 'single', 'file': 'LYRE_RIGHT.npy'},
        
        {'name': 'digit', 'type': 'digits', 'pattern': 'DIGIT_{}.npy'},
        
        {'name': 'points', 'type': 'group', 'files': [f'{n}.npy' for n in points_names]},
        
        {'name': 'OBELIX_WAVE_SCREEN', 'type': 'single', 'file': 'OBELIX_WAVE_SCREEN.npy'},
    ]
    
    for name in asterix_item_names + obelix_item_names:
        config_list.append({'name': name, 'type': 'single', 'file': f'{name}.npy'})
    
    return tuple(config_list)


def _compute_bottom_border(stage_spacing: int = 16, top_border: int = 23) -> int:
    """Compute bottom_border based on stage_spacing and top_border."""
    return 8 * stage_spacing + top_border


def _compute_stage_positions(stage_spacing: int = 16, top_border: int = 23) -> List[int]:
    """Compute stage_positions list based on stage_spacing and top_border."""
    return [
        top_border,  # TOP
        1 * stage_spacing + top_border,  # Stage 1
        2 * stage_spacing + top_border,  # Stage 2
        3 * stage_spacing + top_border,  # Stage 3
        4 * stage_spacing + top_border,  # Stage 4
        5 * stage_spacing + top_border,  # Stage 5
        6 * stage_spacing + top_border,  # Stage 6
        7 * stage_spacing + top_border,  # Stage 7
        8 * stage_spacing + top_border,  # BOTTOM
    ]


def _get_asterix_item_points() -> jnp.ndarray:
    """Factory function for ASTERIX_ITEM_POINTS array."""
    return jnp.array([50, 100, 200, 300, 0], dtype=jnp.int32)


def _get_obelix_item_points() -> jnp.ndarray:
    """Factory function for OBELIX_ITEM_POINTS array."""
    return jnp.array([400, 500, 500, 500, 500], dtype=jnp.int32)

class AsterixConstants(AutoDerivedConstants):
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)
    player_width: int = struct.field(pytree_node=False, default=8)
    player_height: int = struct.field(pytree_node=False, default=8)
    speed_multiplier: float = struct.field(pytree_node=False, default=2.0)
    num_stages: int = struct.field(pytree_node=False, default=8)
    stage_spacing: int = struct.field(pytree_node=False, default=16)
    top_border: int = struct.field(pytree_node=False, default=23)
    
    # Derived constants (dynamic calculation based on static fields)
    bottom_border: Optional[int] = struct.field(pytree_node=False, default=None)
    stage_positions: Optional[List[int]] = struct.field(pytree_node=False, default=None)
    cooldown_frames: int = struct.field(pytree_node=False, default=4)
    hit_frames: int = struct.field(pytree_node=False, default=60)
    respawn_frames: int = struct.field(pytree_node=False, default=120)
    character_transition_frames:int = struct.field(pytree_node=False, default=120)
    
    POPUP_DURATIONS: jnp.ndarray = struct.field(pytree_node=False, default_factory=_get_popup_durations)
    
    num_lives: int = struct.field(pytree_node=False, default=3)
    max_digits_score: int = struct.field(pytree_node=False, default=6)
    
    player_base_speed: float = struct.field(pytree_node=False, default=1.0)
    player_character_speed_factor : float = struct.field(pytree_node=False, default=0.5)
    
    spawn_min_delay: int = struct.field(pytree_node=False, default=20)
    spawn_max_delay: int = struct.field(pytree_node=False, default=50)
    spawn_mode_duration: int = struct.field(pytree_node=False, default=60)
    
    # Initial spawn offsets for entities (off-screen distance)
    initial_spawn_offset_left: int = struct.field(pytree_node=False, default=-4)
    initial_spawn_offset_right: int = struct.field(pytree_node=False, default=-13)

    ASTERIX_ITEM_POINTS: jnp.ndarray = struct.field(pytree_node=False, default_factory=_get_asterix_item_points)
    OBELIX_ITEM_POINTS: jnp.ndarray = struct.field(pytree_node=False, default_factory=_get_obelix_item_points)
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)
    
    def compute_derived(self):
        return {
            'bottom_border': 8 * self.stage_spacing + self.top_border,
            'stage_positions': [
                self.top_border,
                1 * self.stage_spacing + self.top_border,
                2 * self.stage_spacing + self.top_border,
                3 * self.stage_spacing + self.top_border,
                4 * self.stage_spacing + self.top_border,
                5 * self.stage_spacing + self.top_border,
                6 * self.stage_spacing + self.top_border,
                7 * self.stage_spacing + self.top_border,
                8 * self.stage_spacing + self.top_border,
            ],
        }

class CollectibleEnt(struct.PyTreeNode):
    x: jnp.ndarray
    vx: jnp.ndarray
    alive: jnp.ndarray
    type_index: jnp.ndarray

class Enemy(struct.PyTreeNode):
    x: jnp.ndarray
    vx: jnp.ndarray
    alive: jnp.ndarray

class ScorePopup(struct.PyTreeNode):
    x: jnp.ndarray
    value: jnp.ndarray
    timer: jnp.ndarray
    active: jnp.ndarray


class AsterixState(struct.PyTreeNode):
    """Represents the current state of the game"""
    player_x: chex.Array
    player_y: chex.Array
    score: chex.Array
    lives: chex.Array
    game_over: chex.Array
    stage_cooldown: chex.Array
    bonus_life_stage: chex.Array
    player_direction: chex.Array
    enemies: Enemy
    lane_timers: jnp.ndarray
    spawn_mode: jnp.ndarray
    spawn_mode_timer: jnp.ndarray
    first_spawn_done: jnp.ndarray
    force_enemy_next: jnp.ndarray 
    rng: jax.random.PRNGKey
    character_id: chex.Array
    collect_type_index: chex.Array
    collect_type_count: chex.Array
    collectibles: CollectibleEnt
    hit_timer: chex.Array
    respawn_timer: chex.Array
    score_popups: ScorePopup
    character_transition_timer: chex.Array
    level: chex.Array


class EntityPosition(struct.PyTreeNode):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class AsterixObservation(struct.PyTreeNode):
    player: EntityPosition
    enemies: jnp.ndarray
    collectibles: jnp.ndarray


class AsterixInfo(struct.PyTreeNode):
    pass 

class JaxAsterix(JaxEnvironment[AsterixState, AsterixObservation, AsterixInfo, AsterixConstants]):
    # ALE minimal action set: [NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT]
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
        Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT
    ], dtype=jnp.int32)

    def __init__(self, consts: AsterixConstants = None):
        if consts is None:
            consts = AsterixConstants()
        
        super().__init__(consts)
        self.renderer = AsterixRenderer(consts=self.consts)

        stage_borders = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        lane_y_centers = (stage_borders[:-1] + stage_borders[1:]) // 2
        
        entity_height = 8
        self.lane_y_coords = lane_y_centers - (entity_height // 2)

        _, self.state = self.reset()  # Initial state


    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[AsterixObservation, AsterixState]:
        """Initialize a new game state"""
        stage_borders = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        player_x = self.consts.screen_width // 2
        player_y = (stage_borders[3] + stage_borders[4]) // 2 - (self.consts.player_height // 2)

        if key is None:
            key = jax.random.PRNGKey(0)
        
        max_entities = self.consts.num_stages
        
        timer_rng, state_rng = jax.random.split(key, 2)

        lane_indices = jnp.arange(max_entities)
        is_even = (lane_indices % 2) == 0

        lane_timers = jax.random.randint(
            timer_rng,
            (max_entities,),
            self.consts.spawn_min_delay,
            self.consts.spawn_max_delay + 1,
        )
        enemies = Enemy(
            x=jnp.full((max_entities,), -5.0),
            vx=jnp.zeros((max_entities,)),
            alive=jnp.zeros((max_entities,), dtype=bool)
        )
        # Use the configurable offset constants
        left_offset = self.consts.initial_spawn_offset_left
        right_offset = self.consts.initial_spawn_offset_right
        x_start = jnp.where(is_even, -float(left_offset), float(self.consts.screen_width + right_offset))
        base_speed = self.consts.player_base_speed
        vx_start = jnp.where(is_even, 1.0, -1.0) * base_speed
        collectibles = CollectibleEnt(
            x=x_start,
            vx=vx_start,
            alive=jnp.ones((max_entities,), dtype=bool),
            type_index=jnp.zeros((max_entities,), dtype=jnp.int32)
        )

        score_popups = ScorePopup(
            x=jnp.full((max_entities,), -5.0),
            value=jnp.zeros((max_entities,), dtype=jnp.int32),
            timer=jnp.zeros((max_entities,), dtype=jnp.int32),
            active=jnp.zeros((max_entities,), dtype=bool)
        )

        state = AsterixState(
            player_x =jnp.array(player_x, dtype=jnp.int32),
            player_y=jnp.array(player_y, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.num_lives, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            stage_cooldown = jnp.array(self.consts.cooldown_frames, dtype=jnp.int32),
            bonus_life_stage=jnp.array(0, dtype=jnp.int32),
            player_direction=jnp.array(1, dtype=jnp.int32),
            enemies=enemies,
            lane_timers=lane_timers,
            spawn_mode=jnp.array(1, dtype=jnp.int32),
            spawn_mode_timer=jnp.array(60, dtype=jnp.int32),
            first_spawn_done=jnp.array(True, dtype=jnp.bool_),
            force_enemy_next=jnp.ones((max_entities,), dtype=jnp.bool_), 
            rng=state_rng,
            character_id=jnp.array(0, dtype=jnp.int32),
            collect_type_index=jnp.array(0, dtype=jnp.int32),
            collect_type_count=jnp.array(0, dtype=jnp.int32),
            collectibles=collectibles,
            hit_timer=jnp.array(0, dtype=jnp.int32),
            respawn_timer = jnp.array(0, dtype=jnp.int32),
            score_popups = score_popups,
            character_transition_timer = jnp.array(0, dtype=jnp.int32),
            level=jnp.array(1, dtype=jnp.int32),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AsterixState, action: int) -> tuple[
        AsterixObservation, AsterixState, float, bool, AsterixInfo]:
        
        player_height = self.consts.player_height
        cooldown_frames = self.consts.cooldown_frames
        can_switch_stage = state.stage_cooldown <= 0

        stage_borders = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        num_stage = stage_borders.shape[0]

        stage_diffs = jnp.abs(stage_borders - state.player_y)
        current_stage = jnp.argmin(stage_diffs)

        dx_table = jnp.array([0, 0, 1, -1, 0, 1, -1, 1, -1], dtype=jnp.int32)
        dy_table = jnp.array([0, -1, 0, 0, 1, -1, -1, 1, 1], dtype=jnp.int32)
        dx = dx_table[action]
        dy = dy_table[action]
        
        float_dx = dx.astype(jnp.float32) * self.consts.speed_multiplier
        int_dx = jnp.where(
            dx == 0,
            jnp.int32(0),
            jnp.int32(jnp.sign(float_dx) * jnp.maximum(1.0, jnp.floor(jnp.abs(float_dx) + 0.5)))
        )

        paused = (state.respawn_timer > 0) | (state.character_transition_timer > 0)

        stage_move = jnp.where(dy < 0, -1, jnp.where(dy > 0, 1, 0))
        tentative_stage = current_stage + jnp.where(~paused & can_switch_stage, stage_move, 0)
        new_stage_idx = jnp.clip(tentative_stage, 0, num_stage - 2)
        computed_y = ((stage_borders[new_stage_idx] + stage_borders[new_stage_idx + 1]) // 2) - (player_height // 2)

        _stage_mask_w_scaled = self.renderer.SHAPE_MASKS['STAGE'].shape[1]
        _width_scale = self.renderer.config.width_scaling
        _stage_w_orig = jnp.maximum(1, jnp.round(_stage_mask_w_scaled / _width_scale)).astype(jnp.int32)
        stage_left_x = (self.consts.screen_width - _stage_w_orig) // 2
        stage_right_x = stage_left_x + _stage_w_orig

        computed_player_x = jnp.clip(
            state.player_x + jnp.where(paused, jnp.int32(0), int_dx),
            stage_left_x,
            stage_right_x - self.consts.player_width,
        ).astype(jnp.int32)

        computed_direction = jnp.where(
            dx < 0, 1,
            jnp.where(dx > 0, 2, state.player_direction)
        )

        changed_stage = (stage_move != 0) & can_switch_stage & (~paused)
        computed_cooldown = jnp.where(
            changed_stage,
            cooldown_frames,
            jnp.maximum(state.stage_cooldown - jnp.where(paused, 0, 1), 0)
        )

        new_player_x = jnp.where(paused, state.player_x, computed_player_x)
        new_y = jnp.where(paused, state.player_y, computed_y)
        new_player_direction = jnp.where(paused, state.player_direction, computed_direction)
        new_cooldown = jnp.where(paused, state.stage_cooldown, computed_cooldown)

        lane_y_coords = self.lane_y_coords
        item_w = 8
        item_h = 8
        enemy_width = 8
        enemy_w = 8
        enemy_h = 8
        screen_width = self.consts.screen_width
        num_platforms = self.consts.num_stages

        rng_mode, rng_lanes_pos, rng_next = jax.random.split(state.rng, 3)

        mode_timer = state.spawn_mode_timer - 1
        switch_mode = mode_timer <= 0
        new_mode = jnp.where(switch_mode, 1 - state.spawn_mode, state.spawn_mode)
        new_mode_timer = jnp.where(switch_mode, self.consts.spawn_mode_duration, mode_timer)

        is_blocked = (state.enemies.alive | state.collectibles.alive | state.score_popups.active)
        new_lane_timers = jnp.where(paused | is_blocked, state.lane_timers, state.lane_timers - 1)
        should_spawn_lane = (~paused) & (~is_blocked) & (new_lane_timers <= 0)
        
        rng_pos_keys = jax.random.split(rng_lanes_pos, num_platforms)
        entity_speed = jax.lax.select(state.character_id == 1, 2.0, 1.0)

        def update_lanes(i, carry):
            ens, cols, timers, first_done, force_enemy = carry
            
            pred = should_spawn_lane[i]
            
            def spawn_entity(args):
                e, c, t, first_done, force_flag = args
                key_pos = rng_pos_keys[i]
                key_type, key_side = jax.random.split(key_pos)
                
                # GATEKEEPER LOGIC:
                # If force_flag is True, we MUST spawn an enemy.
                # If False, we respect the global spawn_mode (clumping).
                must_be_enemy = force_flag[i]
                
                prob_enemy = jax.lax.select(new_mode == 0, 0.9, 0.1)
                is_enemy = must_be_enemy | jax.lax.select(first_done, jax.random.bernoulli(key_type, prob_enemy), False)
                
                # UPDATE FORCE FLAG:
                # If we just spawned an Enemy, next can be an Item (False).
                # If we just spawned an Item, next MUST be an Enemy (True).
                new_force_val = jnp.where(is_enemy, False, True)
                
                x_start = jax.lax.select(jax.random.bernoulli(key_side), screen_width + 8, -8)
                vx_val = entity_speed * jax.lax.select(x_start > 0, -1.0, 1.0)
                
                new_e = e.replace(
                    x=e.x.at[i].set(jnp.where(is_enemy, x_start, e.x[i])),
                    vx=e.vx.at[i].set(jnp.where(is_enemy, vx_val, e.vx[i])),
                    alive=e.alive.at[i].set(jnp.where(is_enemy, True, e.alive[i]))
                )
                new_c = c.replace(
                    x=c.x.at[i].set(jnp.where(~is_enemy, x_start, c.x[i])),
                    vx=c.vx.at[i].set(jnp.where(~is_enemy, vx_val, c.vx[i])),
                    alive=c.alive.at[i].set(jnp.where(~is_enemy, True, c.alive[i])),
                    type_index=c.type_index.at[i].set(state.collect_type_index)
                )
                
                new_delay = jax.random.randint(key_side, (), self.consts.spawn_min_delay, self.consts.spawn_max_delay)
                new_t = t.at[i].set(new_delay)
                new_first_done = first_done | pred
                
                # Update the specific lane in the force_flag array
                new_force_arr = force_flag.at[i].set(new_force_val)
                
                return new_e, new_c, new_t, new_first_done, new_force_arr

            return jax.lax.cond(pred, spawn_entity, lambda x: x, (ens, cols, timers, first_done, force_enemy))

        enemies, collectibles, new_lane_timers, first_spawn_done, new_force_enemy_next = jax.lax.fori_loop(
            0, num_platforms, update_lanes, 
            (state.enemies, state.collectibles, new_lane_timers, state.first_spawn_done, state.force_enemy_next)
        )

        new_enemy_x = jnp.where(paused, enemies.x, enemies.x + enemies.vx)
        alive_enemies = jnp.where(
            paused,
            enemies.alive,
            (new_enemy_x >= -enemy_width) & (new_enemy_x <= screen_width + enemy_width) & enemies.alive
        )
        enemies = enemies.replace(x=new_enemy_x, alive=alive_enemies)

        new_item_x = jnp.where(paused, collectibles.x, collectibles.x + collectibles.vx)
        alive_items = jnp.where(
            paused,
            collectibles.alive,
            (new_item_x >= -item_w) & (new_item_x <= self.consts.screen_width + item_w) & collectibles.alive
        )
        collectibles = collectibles.replace(x=new_item_x, alive=alive_items)

        def check_collision(px, py, pw, ph, ex, ey, ew, eh):
            return ((px < ex + ew) & (px + pw > ex) & (py < ey + eh) & (py + ph > ey))

        collisions_enemy = jnp.where(
            paused,
            False,
            check_collision(new_player_x, new_y, self.consts.player_width, self.consts.player_height,
                            enemies.x, lane_y_coords, enemy_w, enemy_h) & enemies.alive
        )
        any_collision_enemy = jnp.where(paused, False, jnp.any(collisions_enemy))
        enemies = enemies.replace(alive=jnp.where(collisions_enemy, False, enemies.alive))

        collisions_item = jnp.where(
            paused,
            False,
            check_collision(new_player_x, new_y,
                            self.consts.player_width, self.consts.player_height,
                            collectibles.x, lane_y_coords, item_w, item_h) & collectibles.alive
        )
        any_collisions_item = jnp.sum(collisions_item) > 0
        hit_items_count = jnp.where(paused, jnp.int32(0), jnp.sum(collisions_item).astype(jnp.int32))
        collectibles = collectibles.replace(alive=jnp.where(collisions_item & (~paused), False, collectibles.alive))

        char_id = state.character_id
        start_type_idx = state.collect_type_index
        start_type_count = state.collect_type_count
        types_count = jnp.where(char_id == 0, jnp.int32(4), jnp.int32(5))
        points_array = jnp.where(char_id == 0, self.consts.ASTERIX_ITEM_POINTS, self.consts.OBELIX_ITEM_POINTS)

        k = hit_items_count
        new_type_count_total = start_type_count + k
        type_progressions = new_type_count_total // jnp.int32(50)
        end_type_count = new_type_count_total % jnp.int32(50)
        end_type_idx = (start_type_idx + type_progressions) % types_count

        cap_current = jnp.int32(50) - start_type_count
        items_current = jnp.minimum(k, cap_current)
        points_current = items_current * points_array[start_type_idx]

        rem_after_current = jnp.maximum(0, k - items_current)
        full_segments = rem_after_current // jnp.int32(50)
        rem_tail = rem_after_current % jnp.int32(50)

        start_seg_type = (start_type_idx + 1) % types_count
        cycle_sum = jnp.int32(50) * jnp.sum(points_array)
        num_cycles = full_segments // types_count
        leftover_segments = full_segments % types_count
        points_full_cycles = num_cycles * cycle_sum
        seq_indices = (start_seg_type + jnp.arange(10, dtype=jnp.int32)) % types_count
        mask_left = (jnp.arange(10, dtype=jnp.int32) < leftover_segments)
        points_leftover_segments = jnp.int32(50) * jnp.sum(jnp.where(mask_left, points_array[seq_indices], 0))

        rem_type = (start_seg_type + full_segments) % types_count
        points_tail = rem_tail * points_array[rem_type]

        total_points = points_current + points_full_cycles + points_leftover_segments + points_tail
        computed_score = state.score + total_points
        new_score = jnp.where(paused, state.score, computed_score)

        def spawn_score_popup(score_popups, collisions_item, collectibles, points_array):
            def body(i, popup):
                should_spawn = collisions_item[i]
                is_free = ~popup.active[i]
                should_spawn_here = should_spawn & is_free
                
                col_type = collectibles.type_index[i]
                lut_idx = jnp.minimum(col_type, 4)
                duration = self.consts.POPUP_DURATIONS[lut_idx]
                
                value = points_array[col_type]
                popup = popup.replace(
                    x=popup.x.at[i].set(jnp.where(should_spawn_here, collectibles.x[i], popup.x[i])),
                    value=popup.value.at[i].set(jnp.where(should_spawn_here, value, popup.value[i])),
                    timer=popup.timer.at[i].set(jnp.where(should_spawn_here, duration, popup.timer[i])),
                    active=popup.active.at[i].set(jnp.where(should_spawn_here, True, popup.active[i]))
                )
                return popup

            return jax.lax.fori_loop(0, self.consts.num_stages, body, score_popups)

        score_popups = jax.lax.cond(
            any_collisions_item,
            lambda: spawn_score_popup(state.score_popups, collisions_item, collectibles, points_array),
            lambda: state.score_popups
        )

        switch_to_obelix = (state.character_id == 0) & (new_score >= jnp.int32(32500)) & (
                    state.score < jnp.int32(32500))
        new_character_id = jnp.where(switch_to_obelix, jnp.int32(1), state.character_id)
        new_collect_type_index = jnp.where(paused, state.collect_type_index, jnp.where(switch_to_obelix, jnp.int32(0), end_type_idx))
        new_collect_type_count = jnp.where(paused, state.collect_type_count, jnp.where(switch_to_obelix, jnp.int32(0), end_type_count))

        transition_frames = jnp.int32(self.consts.character_transition_frames)
        new_transition_timer = jnp.where(
            switch_to_obelix,
            transition_frames,
            jnp.maximum(state.character_transition_timer - 1, 0)
        )

        def _clear_entities(_):
            cleared_enemies = enemies.replace(
                x=jnp.full_like(enemies.x, -5.0),
                vx=jnp.zeros_like(enemies.vx),
                alive=jnp.zeros_like(enemies.alive),
            )
            cleared_collectibles = collectibles.replace(
                x=jnp.full_like(collectibles.x, -5.0),
                vx=jnp.zeros_like(collectibles.vx),
                alive=jnp.zeros_like(collectibles.alive),
            )
            return cleared_enemies, cleared_collectibles

        enemies, collectibles = jax.lax.cond(
            switch_to_obelix,
            _clear_entities,
            lambda _: (enemies, collectibles),
            operand=None
        )

        bonus_thresholds = jnp.array([10_000, 30_000, 50_000, 80_000, 110_000], dtype=jnp.int32)
        bonus_interval = 40_000

        def calc_bonus_stage(score):
            below = jnp.sum(score >= bonus_thresholds)
            above = jnp.maximum(score - 110_000, 0) // bonus_interval
            return below + above

        new_bonus_stage = jnp.where(paused, state.bonus_life_stage, calc_bonus_stage(new_score))
        bonus_lives_gained = jnp.maximum(new_bonus_stage - state.bonus_life_stage, 0)
        new_lives = jnp.where(any_collision_enemy, state.lives - 1, state.lives + bonus_lives_gained)

        rt_prev = state.respawn_timer
        rt_after_decr = jnp.maximum(rt_prev - 1, 0)
        new_respawn_timer = jnp.where(any_collision_enemy & (~paused),
                                      jnp.int32(self.consts.respawn_frames),
                                      rt_after_decr)
        new_hit_timer = jnp.where(
            any_collision_enemy & (~paused),
            jnp.array(self.consts.hit_frames, dtype=jnp.int32),
            jnp.maximum(state.hit_timer - 1, 0),
        )

        just_finished_respawn = (rt_prev > 0) & (new_respawn_timer == 0)

        target_stage_idx = 3
        respawn_y = ((stage_borders[target_stage_idx] + stage_borders[target_stage_idx + 1]) // 2) - (
                    player_height // 2)
        respawn_x = self.consts.screen_width // 2

        enemies = jax.lax.cond(
            just_finished_respawn,
            lambda _: enemies.replace(
                x=jnp.full_like(enemies.x, -5.0),
                vx=jnp.zeros_like(enemies.vx),
                alive=jnp.zeros_like(enemies.alive)
            ),
            lambda _: enemies,
            operand=None
        )
        collectibles = jax.lax.cond(
            just_finished_respawn,
            lambda _: collectibles.replace(
                x=jnp.full_like(collectibles.x, -5.0),
                vx=jnp.zeros_like(collectibles.vx),
                alive=jnp.zeros_like(collectibles.alive)
            ),
            lambda _: collectibles,
            operand=None
        )

        new_player_x = jnp.where(just_finished_respawn, jnp.int32(respawn_x), new_player_x)
        new_y = jnp.where(just_finished_respawn, jnp.int32(respawn_y), new_y)
        new_cooldown = jnp.where(just_finished_respawn, jnp.int32(self.consts.cooldown_frames), new_cooldown)
        
        def reset_timers_fn(t):
            return jax.random.randint(rng_next, (num_platforms,), self.consts.spawn_min_delay, self.consts.spawn_max_delay)
        new_lane_timers = jax.lax.cond(just_finished_respawn, reset_timers_fn, lambda t: t, new_lane_timers)
        
        # Also reset gatekeeper on death to prevent unfair starts
        new_force_enemy_next = jax.lax.cond(just_finished_respawn, lambda f: jnp.ones_like(f, dtype=jnp.bool_), lambda f: f, new_force_enemy_next)

        def _update_popups(_):
            new_timer = jnp.maximum(score_popups.timer - 1, 0)
            new_active = score_popups.active & (new_timer > 0)
            return score_popups.replace(timer=new_timer, active=new_active)

        def _clear_popups(_):
            return score_popups.replace(
                x=jnp.full_like(score_popups.x, -5.0),
                value=jnp.zeros_like(score_popups.value),
                timer=jnp.zeros_like(score_popups.timer),
                active=jnp.zeros_like(score_popups.active),
            )

        score_popups = jax.lax.cond(just_finished_respawn, _clear_popups, _update_popups, operand=None)

        game_over = jnp.where(new_lives <= 0, jnp.array(True), state.game_over)

        new_level = jnp.maximum(1, (new_score // 10000) + 1).astype(jnp.int32)
        
        new_state = AsterixState(
            player_x=new_player_x,
            player_y=new_y,
            lives=new_lives,
            score=new_score,
            game_over=game_over,
            stage_cooldown=new_cooldown,
            bonus_life_stage=new_bonus_stage,
            player_direction=new_player_direction,
            enemies=enemies,
            lane_timers=new_lane_timers,
            spawn_mode=new_mode,
            spawn_mode_timer=new_mode_timer,
            first_spawn_done=first_spawn_done,
            force_enemy_next=new_force_enemy_next,
            rng=rng_next,
            character_id=new_character_id,
            collect_type_index=new_collect_type_index,
            collect_type_count=new_collect_type_count,
            collectibles=collectibles,
            hit_timer=new_hit_timer,
            respawn_timer=new_respawn_timer,
            score_popups=score_popups,
            character_transition_timer=new_transition_timer,
            level=new_level,
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AsterixState):
        player = EntityPosition(
            x=state.player_x.astype(jnp.int32),
            y=state.player_y.astype(jnp.int32),
            width=jnp.array(self.consts.player_width, dtype=jnp.int32),
            height=jnp.array(self.consts.player_height, dtype=jnp.int32),
        )

        enemy = jnp.stack([
            state.enemies.x.astype(jnp.int32),
            self.lane_y_coords.astype(jnp.int32),
            jnp.full_like(state.enemies.x, 8, dtype=jnp.int32),
            jnp.full_like(state.enemies.x, 8, dtype=jnp.int32),
        ], axis=-1)  # Shape: (num_stages, 4)

        collectible = jnp.stack([
            state.collectibles.x.astype(jnp.int32),
            self.lane_y_coords.astype(jnp.int32),
            jnp.full_like(state.collectibles.x, 8, dtype=jnp.int32),
            jnp.full_like(state.collectibles.x, 8, dtype=jnp.int32),
        ], axis=-1)  # Shape: (num_stages, 4)
        
        return AsterixObservation(player=player, enemies=enemy, collectibles=collectible)


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AsterixState) -> AsterixInfo:
        return AsterixInfo()

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsterixState, state: AsterixState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AsterixState) -> bool:
        return state.game_over

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Asterix.
        Actions are:
        0: NOOP
        1: UP
        2: RIGHT
        3: LEFT
        4: DOWN
        5: UPRIGHT
        6: UPLEFT
        7: DOWNRIGHT
        8: DOWNLEFT
        """
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        # Returns the observation space for Asterix.
        # The observation contains:
        # - player: EntityPosition (x, y, width, height)
        # - score: int (0-99)
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "enemies": spaces.Box(low=0, high=160, shape=(8, 4), dtype=jnp.int32),
            "collectibles": spaces.Box(low=0, high=160, shape=(8, 4), dtype=jnp.int32),
        })


    def image_space(self) -> spaces.Box:
        """Returns the image space for Asterix.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    def render(self, state: AsterixState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: AsterixObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.width.flatten(),
            obs.player.height.flatten(),
            obs.enemies.flatten(),
            obs.collectibles.flatten(),
        ])


class AsterixRenderer(JAXGameRenderer):
    def __init__(self, consts: AsterixConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or AsterixConstants()
        super().__init__(self.consts)

        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.screen_height, self.consts.screen_width),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Asset base path
        self._sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites/asterix")

        # Load all assets via declarative manifest
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # Add procedural background
        static_procedural = _create_static_procedural_sprites(self.consts.screen_height, self.consts.screen_width)
        final_asset_config.insert(0, {'name': 'background', 'type': 'background', 'data': static_procedural['background']})
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, self._sprite_path)

        # Ensure player sprite stacks have identical shapes across characters
        def _pad_stack_to(hwc_stack, target_h, target_w):
            pad_h = target_h - hwc_stack.shape[1]
            pad_w = target_w - hwc_stack.shape[2]
            return jnp.pad(
                hwc_stack,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=self.jr.TRANSPARENT_ID,
            )

        ax_stack = self.SHAPE_MASKS['ASTERIX_SPRITES']  # (N, H, W)
        ob_stack = self.SHAPE_MASKS['OBELIX_SPRITES']   # (N, H, W)
        max_h = int(max(ax_stack.shape[1], ob_stack.shape[1]))
        max_w = int(max(ax_stack.shape[2], ob_stack.shape[2]))
        if (ax_stack.shape[1] != max_h) or (ax_stack.shape[2] != max_w):
            ax_stack = _pad_stack_to(ax_stack, max_h, max_w)
        if (ob_stack.shape[1] != max_h) or (ob_stack.shape[2] != max_w):
            ob_stack = _pad_stack_to(ob_stack, max_h, max_w)
        self.SHAPE_MASKS['ASTERIX_SPRITES'] = ax_stack
        self.SHAPE_MASKS['OBELIX_SPRITES'] = ob_stack

        # Precompute static frames (ID raster and wave RGB)
        self.PRE_RENDERED_BACKGROUND, self.PRE_RENDERED_WAVE = self._precompute_static_frames()

        # Cache collectible stacks (pad singles to common shape before stacking)
        def _pad_masks_to_common_shape(mask_list):
            # mask_list: list of 2D ID masks (H, W)
            heights = [m.shape[0] for m in mask_list]
            widths = [m.shape[1] for m in mask_list]
            max_h = max(heights)
            max_w = max(widths)
            padded = []
            for m in mask_list:
                pad_h = max_h - m.shape[0]
                pad_w = max_w - m.shape[1]
                # Pad bottom and right with TRANSPARENT_ID to match renderer convention
                padded.append(jnp.pad(m, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=self.jr.TRANSPARENT_ID))
            return jnp.stack(padded)

        self.AX_COLLECTIBLES = _pad_masks_to_common_shape([
            self.SHAPE_MASKS['CAULDRON'],
            self.SHAPE_MASKS['HELMET'],
            self.SHAPE_MASKS['SHIELD'],
            self.SHAPE_MASKS['LAMP'],
        ])
        self.OB_COLLECTIBLES = _pad_masks_to_common_shape([
            self.SHAPE_MASKS['APPLE'],
            self.SHAPE_MASKS['FISH'],
            self.SHAPE_MASKS['WILD_BOAR_LEG'],
            self.SHAPE_MASKS['MUG'],
            self.SHAPE_MASKS['CAULDRON'],
        ])

        stage_borders = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        lane_y_centers = (stage_borders[:-1] + stage_borders[1:]) // 2
        
        enemy_height = 8
        collectible_height = 8
        popup_height = 8
        
        self.enemy_y_coords = lane_y_centers - (enemy_height // 2)
        self.collectible_y_coords = lane_y_centers - (collectible_height // 2)
        self.popup_y_coords = lane_y_centers - (popup_height // 2)


    def _precompute_static_frames(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Start from background ID raster
        raster = self.BACKGROUND

        stage_sprite = self.SHAPE_MASKS['STAGE']
        width_scale = self.config.width_scaling
        height_scale = self.config.height_scaling
        # Convert scaled sprite dimensions back to original game units for positioning
        stage_sprite_w_orig = int(jnp.maximum(1, jnp.round(stage_sprite.shape[1] / width_scale)))
        stage_sprite_h_orig = int(jnp.maximum(1, jnp.round(stage_sprite.shape[0] / height_scale)))
        stage_x = (self.consts.screen_width - stage_sprite_w_orig) // 2

        def stamp_stage(i, r):
            stage_positions = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
            stage_y = stage_positions[i]
            is_drawable = (i > 0) & (i < (stage_positions.shape[0] - 1))
            return jax.lax.cond(
                is_drawable,
                lambda rr: self.jr.render_at(rr, stage_x, stage_y, stage_sprite),
                lambda rr: rr,
                r,
            )

        stage_positions = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        raster = jax.lax.fori_loop(0, stage_positions.shape[0], stamp_stage, raster)

        top_sprite = self.SHAPE_MASKS['TOP']
        stage_height_orig = stage_sprite_h_orig
        top_sprite_w_orig = int(jnp.maximum(1, jnp.round(top_sprite.shape[1] / width_scale)))
        top_sprite_h_orig = int(jnp.maximum(1, jnp.round(top_sprite.shape[0] / height_scale)))
        top_x = (self.consts.screen_width - top_sprite_w_orig) // 2
        top_y = self.consts.top_border - self.consts.stage_spacing + stage_height_orig
        raster = self.jr.render_at(raster, top_x, top_y, top_sprite)

        bottom_sprite = self.SHAPE_MASKS['BOTTOM']
        bottom_sprite_w_orig = int(jnp.maximum(1, jnp.round(bottom_sprite.shape[1] / width_scale)))
        bottom_x = (self.consts.screen_width - bottom_sprite_w_orig) // 2
        stage_positions = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        bottom_y = stage_positions[-1]
        raster = self.jr.render_at(raster, bottom_x, bottom_y, bottom_sprite)

        pre_rendered_background = raster

        wave_raster_base = jnp.full_like(self.BACKGROUND, self.BACKGROUND[0, 0])
        wave_sprite = self.SHAPE_MASKS['OBELIX_WAVE_SCREEN']
        x = (self.consts.screen_width - wave_sprite.shape[1]) // 2
        y = (self.consts.screen_height - wave_sprite.shape[0]) // 2
        pre_rendered_wave = self.jr.render_at(wave_raster_base, x, y, wave_sprite)
        pre_rendered_wave_rgb = self.jr.render_from_palette(pre_rendered_wave, self.PALETTE)
        return pre_rendered_background, pre_rendered_wave_rgb

    @partial(jax.jit, static_argnums=(0,))
    def _render_lyres(self, state, raster):
        lyre_left_sprite = self.SHAPE_MASKS['LYRE_LEFT']
        lyre_right_sprite = self.SHAPE_MASKS['LYRE_RIGHT']
        enemy_y_coords = self.enemy_y_coords

        def render_single_lyre(i, raster_inner):
            is_alive = state.enemies.alive[i]
            x = state.enemies.x[i].astype(jnp.int32)
            y = enemy_y_coords[i].astype(jnp.int32)
            vx = state.enemies.vx[i]
            lyre_sprite = jax.lax.select(vx < 0, lyre_left_sprite, lyre_right_sprite)
            return jax.lax.cond(
                is_alive,
                lambda r: self.jr.render_at_clipped(r, x, y, lyre_sprite),
                lambda r: r,
                raster_inner,
            )

        return jax.lax.fori_loop(0, state.enemies.x.shape[0], render_single_lyre, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_collectibles(self, state, raster):
        collectible_y_coords = self.collectible_y_coords
        
        def render_one(i, r_in):
            is_alive = state.collectibles.alive[i]
            x = state.collectibles.x[i].astype(jnp.int32)
            y = collectible_y_coords[i].astype(jnp.int32)
            idx_item = state.collectibles.type_index[i]

            def render_for_char(r):
                def render_ax(r_ax):
                    idx_ax = jnp.minimum(idx_item, 3)
                    sprite = self.AX_COLLECTIBLES[idx_ax]
                    return self.jr.render_at_clipped(r_ax, x, y, sprite)

                def render_ob(r_ob):
                    idx_ob = jnp.minimum(idx_item, 4)
                    sprite = self.OB_COLLECTIBLES[idx_ob]
                    return self.jr.render_at_clipped(r_ob, x, y, sprite)

                return jax.lax.switch(state.character_id, [render_ax, render_ob], r)

            return jax.lax.cond(is_alive, render_for_char, lambda r: r, r_in)

        return jax.lax.fori_loop(0, state.collectibles.x.shape[0], render_one, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_score_popups(self, state, raster):
        point_sprites = self.SHAPE_MASKS['points']
        popup_y_coords = self.popup_y_coords

        def body(i, r_in):
            def render_active(r):
                value = state.score_popups.value[i]
                idx = jnp.where(value == 50, 0,
                      jnp.where(value == 100, 1,
                      jnp.where(value == 200, 2,
                      jnp.where(value == 300, 3,
                      jnp.where(value == 400, 4,
                      jnp.where(value == 500, 5, 0))))))
                sprite = point_sprites[idx]
                x = state.score_popups.x[i].astype(jnp.int32)
                y = popup_y_coords[i].astype(jnp.int32)
                return self.jr.render_at_clipped(r, x, y, sprite)

            return jax.lax.cond(state.score_popups.active[i], render_active, lambda r: r, r_in)

        return jax.lax.fori_loop(0, state.score_popups.x.shape[0], body, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_player(self, state, raster):
        player_sprites_stack = jax.lax.switch(
            state.character_id,
            [
                lambda: self.SHAPE_MASKS['ASTERIX_SPRITES'],
                lambda: self.SHAPE_MASKS['OBELIX_SPRITES'],
            ]
        )

        normal_idx = state.player_direction - 1
        hit_idx = (state.player_direction - 1) + 2
        sprite_idx = jax.lax.select(state.hit_timer > 0, hit_idx, normal_idx)
        player_sprite = player_sprites_stack[sprite_idx]

        flip_offset = jax.lax.switch(
            state.character_id,
            [
                lambda: self.FLIP_OFFSETS['ASTERIX_SPRITES'],
                lambda: self.FLIP_OFFSETS['OBELIX_SPRITES'],
            ]
        )

        return self.jr.render_at(
            raster,
            state.player_x,
            state.player_y,
            player_sprite,
            flip_offset=flip_offset,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_score_and_lives(self, state, raster):
        digit_sprites = self.SHAPE_MASKS['digit']
        max_digits = self.consts.max_digits_score
        score = state.score.astype(jnp.int32)

        def count_digits(n):
            return jnp.where(n < 10, 1,
                     jnp.where(n < 100, 2,
                       jnp.where(n < 1000, 3,
                         jnp.where(n < 10_000, 4,
                           jnp.where(n < 100_000, 5, 6)))))

        num_digits = count_digits(score)

        def fill_body(i, carry):
            n, digits = carry
            digits = digits.at[max_digits - 1 - i].set(n % 10)
            return (n // 10, digits)

        _, digits_full = jax.lax.fori_loop(0, max_digits, fill_body, (score, jnp.zeros((max_digits,), dtype=jnp.int32)))
        start_idx = max_digits - num_digits

        score_spacing = 8  # spacing in original game units
        score_x = (self.consts.screen_width - (num_digits * score_spacing)) // 2
        stage_positions = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        bottom_y = stage_positions[-1]
        width_scale = self.config.width_scaling
        height_scale = self.config.height_scaling
        bottom_sprite_height_scaled = self.SHAPE_MASKS['BOTTOM'].shape[0]
        player_sprite_height_scaled = self.SHAPE_MASKS['ASTERIX_SPRITES'][0].shape[0]
        bottom_sprite_height_orig = jnp.maximum(1, jnp.round(bottom_sprite_height_scaled / height_scale)).astype(jnp.int32)
        player_sprite_height_orig = jnp.maximum(1, jnp.round(player_sprite_height_scaled / height_scale)).astype(jnp.int32)
        score_y = bottom_y + bottom_sprite_height_orig + player_sprite_height_orig + 6
        raster = self.jr.render_label_selective(
            raster, score_x, score_y,
            digits_full, digit_sprites,
            start_idx, num_digits,
            spacing=score_spacing,
            max_digits_to_render=max_digits,
        )

        num_lives_to_draw = jnp.maximum(state.lives - 1, 0).astype(jnp.int32)
        player_sprites_stack = jax.lax.switch(
            state.character_id,
            [lambda: self.SHAPE_MASKS['ASTERIX_SPRITES'], lambda: self.SHAPE_MASKS['OBELIX_SPRITES']]
        )
        life_sprite = player_sprites_stack[state.player_direction - 1]

        life_width = jnp.int32(self.consts.player_width)
        lives_spacing = jnp.int32(8)
        total_lives_width = jnp.where(
            num_lives_to_draw > 0,
            num_lives_to_draw * life_width + (num_lives_to_draw - 1) * lives_spacing,
            0,
        )
        lives_start_x = (self.consts.screen_width - total_lives_width) // 2
        lives_y = bottom_y + bottom_sprite_height_orig + 3
        # Use Python ints for static args of the jitted function to avoid tracer hashing
        render_spacing = int(self.consts.player_width + 8)
        render_max_value = int(max(2, int(self.consts.num_lives)))
        raster = self.jr.render_indicator(
            raster, lives_start_x, lives_y,
            num_lives_to_draw,
            life_sprite,
            spacing=render_spacing,
            max_value=render_max_value,
        )
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.PRE_RENDERED_BACKGROUND
        raster = self._render_lyres(state, raster)
        raster = self._render_collectibles(state, raster)
        raster = self._render_score_popups(state, raster)
        raster = self._render_player(state, raster)
        raster = self._render_score_and_lives(state, raster)
        final_raster = self.jr.render_from_palette(raster, self.PALETTE)

        wave_raster_rgb = self.PRE_RENDERED_WAVE
        return jax.lax.cond(
            state.character_transition_timer > 0,
            lambda: wave_raster_rgb,
            lambda: final_raster,
        )