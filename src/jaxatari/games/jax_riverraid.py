import os
#from pty import spawn
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from jax import lax
import jax.lax
import jax
from flax import struct
from sympy import false

import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants

# -------- Game constants --------
def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for River Raid.
    """
    asset_list = [
        # Player (Center, Left, Right)
        {'name': 'player', 'type': 'group', 'files': [
            'plane_center.npy', 
            'plane_left.npy', 
            'plane_right.npy'
        ]},
        # Player Explosion
        {'name': 'player_explosion', 'type': 'single', 'file': 'explosion_2.npy'}, # Uses a normalized enemy explosion
        
        # Bullet
        {'name': 'bullet', 'type': 'single', 'file': 'bullet.npy'},
        
        # Enemies
        {'name': 'enemy_boat', 'type': 'single', 'file': 'boat.npy'},
        {'name': 'enemy_airplane', 'type': 'single', 'file': 'enemy_plane.npy'},
        {'name': 'enemy_helicopter', 'type': 'group', 'files': [
            'helicopter_1.npy', 
            'helicopter_2.npy'
        ]},
        
        # Enemy Explosion
        {'name': 'enemy_explosion', 'type': 'group', 'files': [
            'explosion_1.npy',
            'explosion_2.npy',
        ]},
        
        # Fuel
        {'name': 'fuel', 'type': 'single', 'file': 'fuel.npy'},
        
        # Scenery
        {'name': 'house_tree', 'type': 'single', 'file': 'house_tree.npy'},
        {'name': 'full_dam', 'type': 'single', 'file': 'full_dam.npy'},
        {'name': 'outer_dam', 'type': 'single', 'file': 'outer_dam.npy'},
        {'name': 'street', 'type': 'single', 'file': 'street.npy'},
        {'name': 'dam_explosion', 'type': 'group', 'files': [
            'dam_explosion_1.npy',
            'dam_explosion_2.npy'
        ]},
        # UI
        {'name': 'fuel_display', 'type': 'single', 'file': 'fuel_display.npy'},
        {'name': 'fuel_indicator', 'type': 'single', 'file': 'fuel_indicator.npy'},
        {'name': 'activision', 'type': 'single', 'file': 'activision.npy'},
        {'name': 'score_digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        
        # --- Procedural Colors ---
        # These are 1x1 pixels to get colors into the palette
        {'name': 'green_banks', 'type': 'procedural', 'data': jnp.array([[[26, 132, 26, 255]]], dtype=jnp.uint8)},
        {'name': 'blue_river', 'type': 'procedural', 'data': jnp.array([[[42, 42, 189, 255]]], dtype=jnp.uint8)},
        {'name': 'red_river', 'type': 'procedural', 'data': jnp.array([[[189, 42, 42, 255]]], dtype=jnp.uint8)},
        {'name': 'ui_color', 'type': 'procedural', 'data': jnp.array([[[128, 128, 128, 255]]], dtype=jnp.uint8)},
        {'name': 'black', 'type': 'procedural', 'data': jnp.array([[[0, 0, 0, 255]]], dtype=jnp.uint8)},
        
        # Use a dummy background (just black) since the real background is procedural
        {'name': 'background', 'type': 'background', 'data': jnp.array([[[0, 0, 0, 255]]], dtype=jnp.uint8)},
    ]
    return tuple(asset_list)

class RiverraidConstants(AutoDerivedConstants):
    """Complete constants for Riverraid game"""
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)
    DEFAULT_RIVER_WIDTH: int = struct.field(pytree_node=False, default=80)
    MIN_RIVER_WIDTH: int = struct.field(pytree_node=False, default=40)
    MAX_RIVER_WIDTH: int = struct.field(pytree_node=False, default=160)
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=10)
    MAX_HOUSE_TREES: int = struct.field(pytree_node=False, default=10)
    MINIMUM_SPAWN_COOLDOWN: int = struct.field(pytree_node=False, default=20)
    MAX_FUEL: int = struct.field(pytree_node=False, default=30)
    UI_HEIGHT: int = struct.field(pytree_node=False, default=35)
    SEGMENT_LENGTH: int = struct.field(pytree_node=False, default=400)
    DAM_OFFSET: int = struct.field(pytree_node=False, default=25)
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=7)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=14)
    DEATH_COOLDOWN: int = struct.field(pytree_node=False, default=50)  # longer in real game
    BUFFER: int = struct.field(pytree_node=False, default=50)
    ISLAND_SPAWN_PROB: float = struct.field(pytree_node=False, default=0.3)
    STRAIGHT_PROP: float = struct.field(pytree_node=False, default=0.6)
    EXPANSE_PROP: float = struct.field(pytree_node=False, default=0.2)
    SHRINK_PROP: float = struct.field(pytree_node=False, default=0.2)
    MIN_ALTERNATION_LENGTH: int = struct.field(pytree_node=False, default=3)
    MAX_ALTERNATION_LENGTH: int = struct.field(pytree_node=False, default=10)
    ENTITY_SPAWN_PROP: float = struct.field(pytree_node=False, default=0.2)  # includes fuel, only checked if constraints met (cant be higher than 0.33)
    FUEL_SPAWN_PROP: float = struct.field(pytree_node=False, default=0.15)  # percentage of spawned entity being fuel
    PLAYER_ACCELERATION: float = struct.field(pytree_node=False, default=0.06)
    PLAYER_MAX_SPEED: float = struct.field(pytree_node=False, default=4.0)
    ENEMY_START_MOVING_PROP: float = struct.field(pytree_node=False, default=0.01)  # (cant be higher than 0.33)
    # 0.33 limit due to prop * player_speed and player speed being <= 3 -> prop cant be higher than 1
    SPEED_CHANGE_COOLDOWN: int = struct.field(pytree_node=False, default=20)
    MAX_SCORE_DIGITS: int = struct.field(pytree_node=False, default=8)
    # Asset config
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default=_get_default_asset_config())

@struct.dataclass
class RiverraidState:
    turn_step: chex.Array
    turn_step_linear: chex.Array
    master_key: chex.Array
    river_left: chex.Array
    river_right: chex.Array
    river_inner_left: chex.Array
    river_inner_right: chex.Array
    river_state: chex.Array # 0 keep straight, 1 expanse, 2 shrinking
    river_island_present: chex.Array #0 no, 1 yes, 2 spawning, 3 removing, 4 removing helper
    alternation_cooldown: chex.Array
    river_alternation_length: chex.Array
    river_ongoing_alternation: chex.Array
    island_transition_state: chex.Array

    segment_state : chex.Array
    segment_transition_state: chex.Array
    segment_straigt_counter: chex.Array
    dam_position: chex.Array
    housetree_x: chex.Array
    housetree_y: chex.Array
    housetree_state: chex.Array  # 0 for inactive, 1 for active
    housetree_side: chex.Array  # 1 for left, 2 for right
    housetree_direction: chex.Array  # 1 for left-facing, 2 for right-facing
    housetree_cooldown: chex.Array

    player_x: chex.Array
    player_y: chex.Array
    player_velocity: chex.Array
    player_direction: chex.Array  # 0 left, 1 straight, 2 right
    player_state: chex.Array # 0 alive, 1 dead/respawning
    player_fuel: chex.Array
    player_score: chex.Array
    player_lives: chex.Array
    player_speed: chex.Array
    player_speedchange_cooldown: chex.Array

    player_bullet_x: chex.Array
    player_bullet_y: chex.Array

    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_type: chex.Array
    enemy_state: chex.Array  # 0 empty/dead, 1 alive 2 dying first animation 3 dying second 4 dying final
    enemy_direction: chex.Array  # 0 left static, 1 right static, 2 left moving, 3 right moving
    fuel_x: chex.Array
    fuel_y: chex.Array
    fuel_state: chex.Array  # 0 empty, 1 present
    spawn_cooldown: chex.Array
    enemy_animation_cooldowns: chex.Array
    fuel_animation_cooldowns: chex.Array
    death_cooldown: chex.Array
    dam_explosion_cooldown: chex.Array


@struct.dataclass
class RiverraidInfo:
    time: jnp.ndarray


@struct.dataclass
class RiverraidObservation:
    player: ObjectObservation
    enemies: ObjectObservation
    fuel_tanks: ObjectObservation
    player_projectile: ObjectObservation 
    house_trees: ObjectObservation
    
    river_left_bank: chex.Array
    river_right_bank: chex.Array
    island_left_bank: chex.Array
    island_right_bank: chex.Array
    dam_indicator: chex.Array
    
    player_fuel: chex.Array
    player_lives: chex.Array
    player_score: chex.Array
    player_speed: chex.Array


# logic sperated into 3 branches: island, no_island, island_transition
# except for the transition phases, the states are managed at the top of the method
# islands are randomly spawned in no_island branch only when expanse is in progress (spawn trigger in expanse method)
# island_transition_state that manages the required straight segment at start and end
# islands can always shrink, if shrank below a threshold it is removed entirely
# the logic also tests wheter to despawn the island at any point (even when large)

@jax.jit
def get_action_from_keyboard(state: RiverraidState) -> Action:
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    shooting = keys[pygame.K_SPACE]

    # Diagonale Bewegungen
    up_right = up and right and not left and not down
    up_left = up and left and not right and not down
    down_right = down and right and not left and not up
    down_left = down and left and not right and not up

    # Einzelne Richtungen
    up_only = up and not left and not right and not down
    down_only = down and not left and not right and not up
    left_only = left and not right and not up and not down
    right_only = right and not left and not up and not down

    if shooting:
        if up_right:
            return Action.UPRIGHTFIRE
        elif up_left:
            return Action.UPLEFTFIRE
        elif down_right:
            return Action.DOWNRIGHTFIRE
        elif down_left:
            return Action.DOWNLEFTFIRE
        elif up_only:
            return Action.UPFIRE
        elif down_only:
            return Action.DOWNFIRE
        elif left_only:
            return Action.LEFTFIRE
        elif right_only:
            return Action.RIGHTFIRE
        else:
            return Action.FIRE
    else:
        if up_right:
            return Action.UPRIGHT
        elif up_left:
            return Action.UPLEFT
        elif down_right:
            return Action.DOWNRIGHT
        elif down_left:
            return Action.DOWNLEFT
        elif up_only:
            return Action.UP
        elif down_only:
            return Action.DOWN
        elif left_only:
            return Action.LEFT
        elif right_only:
            return Action.RIGHT
        else:
            return Action.NOOP


class JaxRiverraid(JaxEnvironment):
    # Minimal ALE action set for River Raid
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )
    
    def __init__(self, consts: RiverraidConstants = None):
        self.consts = consts or RiverraidConstants()
        super().__init__(self.consts)
        self.renderer = RiverraidRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def generate_altering_river(self, state: RiverraidState) -> RiverraidState:
        key = state.master_key
        key, alter_check_key, alter_select_key, alter_length_key = jax.random.split(key, 4)


        # Scroll the screen down
        scrolled_left = jnp.roll(state.river_left, 1)
        scrolled_right = jnp.roll(state.river_right, 1)
        scrolled_inner_left = jnp.roll(state.river_inner_left, 1)
        scrolled_inner_right = jnp.roll(state.river_inner_right, 1)

        # jnp.roll puts the last element at the front
        # so we need to clear the history of the island
        scrolled_inner_left = scrolled_inner_left.at[0].set(-1)
        scrolled_inner_right = scrolled_inner_right.at[0].set(-1)
        state = state.replace(river_left=scrolled_left,
                               river_right=scrolled_right,
                               river_inner_left=scrolled_inner_left,
                               river_inner_right=scrolled_inner_right)

        # New state set here ONLY if NO alternation is ongoing
        # determine IF the river is to be altered, else use old state
        new_river_state = jax.lax.cond(
            jnp.logical_and(state.river_alternation_length <= 0, state.alternation_cooldown <= 0),
            lambda state: (jax.random.choice(
                alter_select_key, jnp.array([0, 1, 2]), p=jnp.array([self.consts.STRAIGHT_PROP, self.consts.EXPANSE_PROP, self.consts.SHRINK_PROP])
            )),
            lambda state: state.river_state,
            operand=state
        )

        # alternation length is set here AND whenever an island spawns or despawns (due to the need for hard coding the segment)
        new_alternation_length = jax.lax.cond(state.river_alternation_length <= 0,
                                              lambda state: jax.random.randint(alter_length_key, (), self.consts.MIN_ALTERNATION_LENGTH, self.consts.MAX_ALTERNATION_LENGTH),
                                              lambda state: state.river_alternation_length - 1,
                                              operand=state
                                              )
        # cooldown is handled and new_river state potentially reset to straight
        new_alternation_length, new_river_state, new_alternation_cooldown = jax.lax.cond(
            (state.alternation_cooldown > 0) & (state.river_alternation_length <= 0),
            lambda state: (jnp.array(0), jnp.array(0), state.alternation_cooldown - 1),
            lambda state: (new_alternation_length, new_river_state, new_alternation_length + 10),
            operand=state
        )

        state = state.replace( river_alternation_length=new_alternation_length,
                                river_state=new_river_state,
                                alternation_cooldown=new_alternation_cooldown)


        # used by both island and non island branches
        def straight(state: RiverraidState) -> RiverraidState:
            new_river_left = state.river_left.at[0].set(state.river_left[1])
            new_river_right = state.river_right.at[0].set(state.river_right[1])
            new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1])
            new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1])
            return state.replace(river_left=new_river_left,
                                  river_right=new_river_right,
                                  river_inner_left=new_river_inner_left,
                                  river_inner_right=new_river_inner_right)

        def no_island_branch(state: RiverraidState) -> RiverraidState:
            def expanse(state: RiverraidState) -> RiverraidState:
                new_river_left = state.river_left.at[0].set(state.river_left[1] - 3)
                new_river_right = state.river_right.at[0].set(state.river_right[1] + 3)

                key, subkey = jax.random.split(state.master_key)
                should_set_island = jnp.logical_and(jax.random.bernoulli(subkey, self.consts.ISLAND_SPAWN_PROB),
                                                    new_river_right[0] - new_river_left[0] > 50)
                new_island_present = jax.lax.select(should_set_island, jnp.array(2), state.river_island_present)

                return state.replace(river_left=new_river_left,
                                      river_right=new_river_right,
                                      river_island_present=new_island_present)

            def shrink(state: RiverraidState) -> RiverraidState:
                new_river_left = state.river_left.at[0].set(state.river_left[1] + 3)
                new_river_right = state.river_right.at[0].set(state.river_right[1] - 3)
                return state.replace(river_left=new_river_left,
                                      river_right=new_river_right)

            return lax.switch(
                state.river_state,
                [straight, expanse, shrink],
                state
            )


        def island_branch(state: RiverraidState):
            def expanse(state: RiverraidState) -> RiverraidState:
                new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1] - 3)
                new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1] + 3)
                new_left = state.river_left.at[0].set(state.river_left[1])
                new_right = state.river_right.at[0].set(state.river_right[1])
                return state.replace(river_inner_left=new_river_inner_left,
                                      river_inner_right=new_river_inner_right,
                                      river_left=new_left,
                                      river_right=new_right)

            def shrink(state: RiverraidState) -> RiverraidState:
                proposed_inner_left = state.river_inner_left[1] + 3
                proposed_inner_right = state.river_inner_right[1] - 3
                new_river_inner_left, new_river_inner_right, new_island_present = jax.lax.cond(
                    proposed_inner_left + 0 <= proposed_inner_right,
                    lambda _: (
                        state.river_inner_left.at[0].set(proposed_inner_left),
                        state.river_inner_right.at[0].set(proposed_inner_right),
                        jnp.array(4)
                    ),
                    lambda _: (
                        state.river_inner_left.at[0].set(state.river_inner_left[1]),
                        state.river_inner_right.at[0].set(state.river_inner_right[1]),
                        state.river_island_present
                    ),
                    operand=None
                )

                new_left = state.river_left.at[0].set(state.river_left[1])
                new_right = state.river_right.at[0].set(state.river_right[1])

                return state.replace(river_inner_left=new_river_inner_left,
                                      river_inner_right=new_river_inner_right,
                                      river_left=new_left,
                                      river_right=new_right,
                                      river_island_present=new_island_present
                )
            return lax.switch(
                state.river_state,
                [straight, expanse, shrink],
                state
            )


        def island_transition(state: RiverraidState) -> RiverraidState:
            def spawn_island(state: RiverraidState) -> RiverraidState:
                def straight(state: RiverraidState) -> RiverraidState:
                    new_river_left = state.river_left.at[0].set(state.river_left[1])
                    new_river_right = state.river_right.at[0].set(state.river_right[1])
                    new_alternation_length, new_island_transition_state = (
                        jax.lax.switch(
                            state.island_transition_state,
                            [
                                lambda state: jax.lax.cond(
                                    state.river_alternation_length <= 1,
                                    lambda state: (jnp.array(10), jnp.array(1)),
                                    lambda state: (state.river_alternation_length, state.island_transition_state),
                                    operand=state
                                ),
                                lambda state: jax.lax.cond(
                                    state.river_alternation_length <= 1,
                                    lambda state: (state.river_alternation_length, jnp.array(2)),
                                    lambda state: (state.river_alternation_length, state.island_transition_state),
                                    operand=state
                                ),
                                lambda state: (state.river_alternation_length, state.island_transition_state)
                            ],
                            state
                        ))
                    return state.replace(river_left=new_river_left,
                                          river_right=new_river_right,
                                          river_alternation_length= new_alternation_length,
                                          island_transition_state=new_island_transition_state)

                def initiate_split(state: RiverraidState) -> RiverraidState:
                    new_river_inner_left = state.river_inner_left.at[0].set(self.consts.SCREEN_WIDTH // 2 - 1)
                    new_river_inner_right = state.river_inner_right.at[0].set(self.consts.SCREEN_WIDTH // 2 + 1)
                    new_river_left = state.river_left.at[0].set(state.river_left[1])
                    new_river_right = state.river_right.at[0].set(state.river_right[1])

                    new_alternation_length = jax.random.randint(alter_length_key, (), 4, 12)
                    new_island_transition_state = jnp.array(0)
                    new_river_state = jnp.array(1)
                    new_island_present = jnp.array(1)

                    return state.replace(river_inner_left=new_river_inner_left,
                                          river_inner_right=new_river_inner_right,
                                          river_left=new_river_left,
                                          river_right=new_river_right,
                                          river_alternation_length=new_alternation_length,
                                          island_transition_state=new_island_transition_state,
                                          river_state=new_river_state,
                                          river_island_present=new_island_present)

                return lax.switch(
                    state.island_transition_state,
                    [straight, straight, initiate_split],
                    state
                )

            def terminate_island(state: RiverraidState) -> RiverraidState:
                def remove_island(state: RiverraidState) -> RiverraidState:
                    new_river_inner_left = state.river_inner_left.at[0].set(state.river_inner_left[1] + 3)
                    new_river_inner_right = state.river_inner_right.at[0].set(state.river_inner_right[1] - 3)
                    new_left = state.river_left.at[0].set(state.river_left[1])
                    new_right = state.river_right.at[0].set(state.river_right[1])

                    new_island_present, new_island_transition_state, new_alternation_length = jax.lax.cond(
                        new_river_inner_left[0] >= new_river_inner_right[0],
                        lambda _: (jnp.array(4), jnp.array(4), jnp.array(14)),
                        lambda state: (state.river_island_present, state.island_transition_state, jnp.array(1)),
                        operand=state
                    )


                    return state.replace(
                        river_inner_left=new_river_inner_left,
                        river_inner_right=new_river_inner_right,
                        river_left=new_left,
                        river_right=new_right,
                        river_island_present=new_island_present,
                        island_transition_state=new_island_transition_state,
                        river_alternation_length=new_alternation_length
                    )

                def straight(state: RiverraidState) -> RiverraidState:
                    new_river_left = state.river_left.at[0].set(state.river_left[1])
                    new_river_right = state.river_right.at[0].set(state.river_right[1])

                    new_river_state, new_island_transition_state, new_island_present = (
                                                    jax.lax.cond(state.river_alternation_length <= 1,
                                                    lambda _: (jnp.array(2), jnp.array(0), jnp.array(0)),
                                                    lambda state: (state.river_state, state.island_transition_state, state.river_island_present),
                                                    operand=state
                                                    ))

                    new_alternation_length = jax.lax.cond(
                        new_island_present == 0,
                        lambda _: jax.random.randint(alter_length_key, (), 3, 8),
                        lambda _: state.river_alternation_length,
                        operand=None
                    )

                    return state.replace(river_left=new_river_left,
                                          river_right=new_river_right,
                                          island_transition_state=new_island_transition_state,
                                          river_alternation_length=new_alternation_length,
                                          river_state=new_river_state,
                                          river_island_present=new_island_present)

                return jax.lax.cond(
                    state.island_transition_state == 4,
                    lambda state: straight(state),
                    lambda state: remove_island(state),
                    operand=state
                )

            return jax.lax.cond(jnp.logical_or(state.river_island_present == 0, state.river_island_present == 2),
                            lambda state: spawn_island(state),
                            lambda state: terminate_island(state),
                            operand=state)



        new_river_island_present = jax.lax.cond(
            jnp.logical_and(
                state.river_island_present >= 2,
                jnp.logical_and(
                    state.river_inner_left[0] <= state.river_inner_right[0],
                    state.river_right[1] - state.river_left[1] < self.consts.MIN_RIVER_WIDTH * 2 + 20
                )
            ),
            lambda state: jnp.array(0),
            lambda state: state.river_island_present,
            operand=state
        )
        state = state.replace(river_island_present=new_river_island_present)

        # main handler for what branch to use
        state = jax.lax.cond(
            state.river_island_present == 0,
            lambda state: no_island_branch(state),
            lambda state: jax.lax.cond(
                state.river_island_present == 1,
                lambda state: island_branch(state),
                lambda state: island_transition(state),
                operand=state
            ),
            operand=state
        )

        def no_island_clamping(state: RiverraidState) -> RiverraidState:
            min_river_width = self.consts.MIN_RIVER_WIDTH
            max_river_width = self.consts.MAX_RIVER_WIDTH
            proposed_left = state.river_left[0]
            proposed_right = state.river_right[0]

            proposed_width = (proposed_right - proposed_left).astype(jnp.float32)
            proposed_center = (proposed_left + proposed_right).astype(jnp.float32) / 2.0

            clamped_width = lax.clamp(min_river_width, proposed_width.astype(jnp.int32), max_river_width)

            min_center = clamped_width / 2.0
            max_center = self.consts.SCREEN_WIDTH - (clamped_width / 2.0)
            clamped_center = lax.clamp(min_center, proposed_center, max_center)

            final_left = jnp.round(clamped_center - clamped_width / 2.0).astype(jnp.int32)
            final_right = (final_left + jnp.round(clamped_width).astype(jnp.int32))

            new_left = state.river_left.at[0].set(final_left)
            new_right = state.river_right.at[0].set(final_right)

            return state.replace(
                river_left=new_left,
                river_right=new_right,
            )

        def yes_island_clamping(state: RiverraidState) -> RiverraidState:
            outer_left = state.river_left[0]
            outer_right = state.river_right[0]
            proposed_inner_left = state.river_inner_left[0]
            proposed_inner_right = state.river_inner_right[0]
            min_river_width = self.consts.MIN_RIVER_WIDTH

            new_inner_left = jax.lax.cond(
                (proposed_inner_left - outer_left).astype(jnp.float32) < min_river_width,
                lambda state: state.river_inner_left.at[0].set((outer_left + min_river_width).astype(jnp.int32)),
                lambda state: state.river_inner_left,
                operand=state
            )

            new_inner_right = jax.lax.cond(
                (outer_right - proposed_inner_right).astype(jnp.float32) < min_river_width,
                lambda state: state.river_inner_right.at[0].set((outer_right - min_river_width).astype(jnp.int32)),
                lambda state: state.river_inner_right,
                operand=state
            )

            return state.replace(
                river_inner_left=new_inner_left,
                river_inner_right=new_inner_right,
            )

        state = jax.lax.cond(state.river_island_present == 0,
                            lambda state: no_island_clamping(state),
                            lambda state: yes_island_clamping(state),
                            operand=state
                            )
        return state.replace(master_key=key)

    @partial(jax.jit, static_argnums=(0,))
    def generate_straight_river(self, state: RiverraidState) -> RiverraidState:
        scrolled_left = jnp.roll(state.river_left, 1)
        scrolled_right = jnp.roll(state.river_right, 1)
        scrolled_inner_left = jnp.roll(state.river_inner_left, 1)
        scrolled_inner_right = jnp.roll(state.river_inner_right, 1)
        scrolled_inner_left = scrolled_inner_left.at[0].set(-1)
        scrolled_inner_right = scrolled_inner_right.at[0].set(-1)

        current_width = scrolled_right[1] - scrolled_left[1]
        should_expand = current_width < self.consts.MIN_RIVER_WIDTH * 2

        new_top_left = jnp.where(should_expand, scrolled_left[1] - 3, scrolled_left[1])
        new_top_right = jnp.where(should_expand, scrolled_right[1] + 3, scrolled_right[1])
        new_river_left = scrolled_left.at[0].set(new_top_left)
        new_river_right = scrolled_right.at[0].set(new_top_right)

        return state.replace(
            river_left=new_river_left,
            river_right=new_river_right,
            river_inner_left=scrolled_inner_left,
            river_inner_right=scrolled_inner_right,
            river_state=jnp.array(0)
        )

    @partial(jax.jit, static_argnums=(0,))
    def generate_segment_transition(self, state: RiverraidState) -> RiverraidState:
        def scroll_empty_island(state: RiverraidState) -> RiverraidState:
            scrolled_inner_left = jnp.roll(state.river_inner_left, 1)
            scrolled_inner_right = jnp.roll(state.river_inner_right, 1)
            scrolled_inner_left = scrolled_inner_left.at[0].set(-1)
            scrolled_inner_right = scrolled_inner_right.at[0].set(-1)
            return state.replace(river_inner_left=scrolled_inner_left,
                                    river_inner_right=scrolled_inner_right)

        def first_call(state: RiverraidState) -> RiverraidState:
            new_island_present, new_segment_transition_state = jax.lax.cond(state.river_island_present == 0,
                                            lambda state: (jnp.array(0), jnp.array(2)),
                                            lambda state: (jnp.array(3), jnp.array(1)),
                                            operand=state)
            # let the island removale be done by main river method by hard coding the river_island_present state to 3 (remove)
            return self.generate_altering_river(state.replace(river_island_present=new_island_present,
                                                  segment_transition_state=new_segment_transition_state))

        # let main method handle as explained above
        def remove_island(state: RiverraidState) -> RiverraidState:
            new_state = self.generate_altering_river(state)
            new_segment_transition_state = jax.lax.cond(jnp.logical_and(new_state.river_island_present == 0, new_state.river_state == 0),
                                                        lambda state: jnp.array(2),
                                                        lambda state: state.segment_transition_state,
                                                        operand=state)
            return new_state.replace(segment_transition_state=new_segment_transition_state)

        def shrink_to_damsize(state: RiverraidState) -> RiverraidState:
            scrolled_left = jnp.roll(state.river_left, 1)
            scrolled_right = jnp.roll(state.river_right, 1)

            new_river_left = scrolled_left.at[0].set(scrolled_left[1] + 3)
            new_river_right = scrolled_right.at[0].set(scrolled_right[1] - 3)

            new_segment_transition_state = jax.lax.cond(new_river_right[0] - new_river_left[0] <= self.consts.MIN_RIVER_WIDTH + 12,
                                                        lambda state: jnp.array(3),
                                                        lambda state: state.segment_transition_state,
                                                        operand=state
                                                        )
            new_state = scroll_empty_island(state)
            return new_state.replace(river_left=new_river_left,
                                    river_right=new_river_right,
                                    segment_transition_state=new_segment_transition_state)

        def straight_until_dam(state: RiverraidState) -> RiverraidState:
            scrolled_left = jnp.roll(state.river_left, 1)
            scrolled_right = jnp.roll(state.river_right, 1)
            new_river_left = scrolled_left.at[0].set(state.river_left[0])
            new_river_right = scrolled_right.at[0].set(state.river_right[0])
            new_segment_straight_counter = state.segment_straigt_counter - 1
            new_transition_state = jax.lax.cond(new_segment_straight_counter <= 0,
                                                        lambda state: jnp.array(4),
                                                        lambda state: state.segment_transition_state,
                                                        operand=state)
            new_state = scroll_empty_island(state)
            return new_state.replace(river_left=new_river_left,
                                  river_right=new_river_right,
                                  segment_straigt_counter=new_segment_straight_counter,
                                  segment_transition_state=new_transition_state)

        # place the dam and reset the states for next segment
        def dam_into_new_segment(state: RiverraidState) -> RiverraidState:
            new_state = scroll_empty_island(state)
            new_segment_straight_counter = jnp.array(8)
            dam_position = state.dam_position.at[0].set(1)

            new_river_state = jnp.array(0)
            new_alternation_length = jnp.array(10)
            new_alternation_cooldown = jnp.array(0)
            new_segment_state = state.segment_state + 1
            new_segment_transition_state = jnp.array(0)
            return new_state.replace(segment_state=new_segment_state,
                                  segment_transition_state=new_segment_transition_state,
                                  dam_position=dam_position,
                                  segment_straigt_counter=new_segment_straight_counter,
                                  river_state=new_river_state,
                                  river_alternation_length=new_alternation_length,
                                  alternation_cooldown=new_alternation_cooldown,
                                  dam_explosion_cooldown=jnp.array(15)
                                  )

        return jax.lax.switch(state.segment_transition_state,[first_call,
                                                                    remove_island,
                                                                    shrink_to_damsize,
                                                                    straight_until_dam,
                                                                    dam_into_new_segment],
                                                                   operand=state)

    @partial(jax.jit, static_argnums=(0,))
    def update_river_banks(self, state: RiverraidState) -> RiverraidState:
        return jax.lax.switch(state.segment_state, [lambda state: self.generate_altering_river(state),
                                                    lambda state: self.generate_segment_transition(state),
                                                    lambda state: self.generate_straight_river(state),
                                                    lambda state: self.generate_segment_transition(state)],
                                                    operand=state)

    @partial(jax.jit, static_argnums=(0,))
    def river_generation(self, state: RiverraidState) -> RiverraidState:
        new_segment_state = jax.lax.cond(
            (state.turn_step % self.consts.SEGMENT_LENGTH) == 0,
            lambda state: state.segment_state + 1,
            lambda state: state.segment_state,
            operand=state
        )
        new_state = state.replace(segment_state=new_segment_state % 4)
        new_state = jax.lax.fori_loop(0, new_state.player_speed, lambda i, new_state: self.update_river_banks(new_state),
                                      new_state)
        new_turn_step = jax.lax.cond(new_state.turn_step % self.consts.SEGMENT_LENGTH < self.consts.SEGMENT_LENGTH - 15,
                                     lambda state: state.turn_step + state.player_speed - 1,
                                     lambda state: state.turn_step,
                                     operand=new_state
                                     )
        new_state = new_state.replace(turn_step=new_turn_step)
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def handle_dam(self, state: RiverraidState) -> RiverraidState:
        new_dam_position = jnp.roll(state.dam_position, state.player_speed)
        new_dam_position = jax.lax.fori_loop(0, state.player_speed, lambda i, pos: pos.at[i].set(0), new_dam_position)

        # Kollisionsprüfung Kugel <-> Damm
        bullet_y = jnp.round(state.player_bullet_y).astype(jnp.int32)
        bullet_x = jnp.round(state.player_bullet_x).astype(jnp.int32)

        dam_mask = new_dam_position == 1
        dam_exists = jnp.any(dam_mask)
        dam_top_y = jnp.argmax(dam_mask) - self.consts.DAM_OFFSET
        bullet_on_screen = state.player_bullet_y >= 0

        bullet_above_dam = jnp.logical_and(
            dam_exists,
            jnp.logical_and(
                bullet_on_screen,  # <-- This is the new condition
                jnp.logical_and(bullet_y < dam_top_y + self.consts.DAM_OFFSET, bullet_x > 1)
            )
        )

        river_left = state.river_left[bullet_y]
        river_right = state.river_right[bullet_y]

        new_bullet_x = jnp.where(bullet_above_dam, -1.0, state.player_bullet_x)
        new_bullet_y = jnp.where(bullet_above_dam, -1.0, state.player_bullet_y)

        player_above_dam = jnp.logical_and(dam_exists, state.player_y < dam_top_y + 25)
        new_player_state = jax.lax.cond(
            player_above_dam,
            lambda state: jnp.array(1),
            lambda state: state.player_state,
            operand=state
        )

        new_dam_position = jax.lax.cond(
            bullet_above_dam,
            lambda state: new_dam_position.at[dam_top_y + self.consts.DAM_OFFSET].set(2),
            lambda state: new_dam_position,
            operand=state
        )
        new_score = jax.lax.cond(
            bullet_above_dam,
            lambda state: state.player_score + 500,
            lambda state: state.player_score,
            operand=state
        )
        return state.replace(
            dam_position=new_dam_position,
            player_bullet_x=new_bullet_x,
            player_bullet_y=new_bullet_y,
            player_state=new_player_state,
            player_score=new_score)

    @partial(jax.jit, static_argnums=(0,))
    def player_movement(self, state: RiverraidState, action: chex.Array) -> RiverraidState:
        #input
        press_right = jnp.any(jnp.array([
            action == Action.RIGHT,
            action == Action.RIGHTFIRE,
            action == Action.UPRIGHT,
            action == Action.DOWNRIGHT,
            action == Action.UPRIGHTFIRE,
            action == Action.DOWNRIGHTFIRE
        ]))
        press_left = jnp.any(jnp.array([
            action == Action.LEFT,
            action == Action.LEFTFIRE,
            action == Action.UPLEFT,
            action == Action.DOWNLEFT,
            action == Action.UPLEFTFIRE,
            action == Action.DOWNLEFTFIRE
        ]))

        new_direction = 1 + press_right - press_left
        velocity_change = (press_right * self.consts.PLAYER_ACCELERATION) - (press_left * self.consts.PLAYER_ACCELERATION)
        current_velocity = jnp.where(
            (press_left == 0) & (press_right == 0),
            0.0,
            state.player_velocity + velocity_change
        )

        min_vel = jnp.where(press_right, 0.0, -self.consts.PLAYER_MAX_SPEED)
        max_vel = jnp.where(press_left, 0.0, self.consts.PLAYER_MAX_SPEED)
        new_velocity = jnp.clip(current_velocity, min_vel, max_vel)

        # move
        new_x = state.player_x + new_velocity

        # check collisions
        hitbox_left = new_x
        hitbox_right = new_x + self.consts.PLAYER_WIDTH
        hitbox_top_y = jnp.clip(jnp.round(state.player_y).astype(jnp.int32), 0, self.consts.SCREEN_HEIGHT - 1)
        hitbox_bottom_y = jnp.clip(jnp.round(state.player_y + self.consts.PLAYER_HEIGHT).astype(jnp.int32), 0, self.consts.SCREEN_HEIGHT - 1)

        collision_top_banks = (hitbox_left <= state.river_left[hitbox_top_y]) | \
                              (hitbox_right >= state.river_right[hitbox_top_y])
        collision_bottom_banks = (hitbox_left <= state.river_left[hitbox_bottom_y]) | \
                                 (hitbox_right >= state.river_right[hitbox_bottom_y])

        island_present_top = state.river_inner_left[hitbox_top_y] >= 0
        collision_with_island_top = island_present_top & \
                                    (hitbox_right >= state.river_inner_left[hitbox_top_y]) & \
                                    (hitbox_left <= state.river_inner_right[hitbox_top_y])

        island_present_bottom = state.river_inner_left[hitbox_bottom_y] >= 0
        collision_with_island_bottom = island_present_bottom & \
                                       (hitbox_right >= state.river_inner_left[hitbox_bottom_y]) & \
                                       (hitbox_left <= state.river_inner_right[hitbox_bottom_y])

        any_collision = collision_top_banks | collision_bottom_banks | \
                        collision_with_island_top | collision_with_island_bottom


        # kill player if collision
        new_player_state = jnp.where(any_collision, 1, state.player_state)

        return state.replace(
            player_x=new_x,
            player_velocity=new_velocity,
            player_state=new_player_state,
            player_direction=new_direction
        )

    @partial(jax.jit, static_argnums=(0,))
    def player_shooting(self, state, action: chex.Array):
        shooting = jnp.any(
            jnp.array([
                action == Action.FIRE,
                action == Action.LEFTFIRE,
                action == Action.RIGHTFIRE,
                action == Action.UPFIRE,
                action == Action.DOWNFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE
            ])
        )
        new_bullet_x, new_bullet_y = jax.lax.cond(
            jnp.logical_and(
                shooting,
                state.player_bullet_y < 0),
            lambda state: ((state.player_x + 3).astype(jnp.float32), (state.player_y - 0).astype(jnp.float32)),
            lambda state: (state.player_bullet_x, (state.player_bullet_y - 5).astype(jnp.float32)),
            operand=state
        )

        # collision with outer banks
        new_bullet_x, new_bullet_y = jax.lax.cond(
            jnp.logical_or(new_bullet_x <= state.river_left[new_bullet_y.astype(jnp.int32)],
                            new_bullet_x >= state.river_right[new_bullet_y.astype(jnp.int32)]),
            lambda state: (jnp.array(-1, dtype=jnp.float32), jnp.array(-1, dtype=jnp.float32)),
            lambda state: (new_bullet_x, new_bullet_y),
            operand=state
        )

        # collision with island
        new_bullet_x, new_bullet_y = jax.lax.cond(
            jnp.logical_and(new_bullet_x >= state.river_inner_left[new_bullet_y.astype(jnp.int32)],
                            new_bullet_x <= state.river_inner_right[new_bullet_y.astype(jnp.int32)]),
            lambda state: (jnp.array(-1, dtype=jnp.float32), jnp.array(-1, dtype=jnp.float32)),
            lambda state: (new_bullet_x, new_bullet_y),
            operand=state
        )

        return state.replace(player_bullet_x=new_bullet_x,
                                  player_bullet_y=new_bullet_y)

    @partial(jax.jit, static_argnums=(0,))
    def spawn_enemies(self, state):
        key, spawn_key, x_key = jax.random.split(state.master_key, 3)

        # 0 boat, 1 helicopter, 2 plane
        new_single_enemy_type = jax.random.randint(spawn_key, (), 0, 3)
        free_enemy_idx = jax.lax.cond(
            jnp.any(state.enemy_state == 0),
            lambda state: jnp.argmax(state.enemy_state == 0),
            lambda _: jnp.array(-1, dtype=jnp.int32),
            operand=state
        )

        new_enemy_type = state.enemy_type.at[free_enemy_idx].set(new_single_enemy_type)
        new_enemy_state = state.enemy_state.at[free_enemy_idx].set(jnp.array(1))

        new_enemy_x = jax.lax.cond(
            free_enemy_idx >= 0,
            lambda state: jax.lax.cond(
                new_enemy_type[free_enemy_idx] <= 1, # logic for boat or helicopter (spawn in water)
                lambda state: jax.lax.cond(
                    state.river_inner_left[0] >= 0,
                    lambda state: jax.lax.cond(
                        jax.random.bernoulli(x_key, 0.5),
                                lambda state: jax.random.randint(x_key, (), state.river_left[0] + 4, state.river_inner_left[0] - 4),
                                lambda state: jax.random.randint(x_key, (), state.river_inner_right[0] + 4, state.river_right[0] - 4),
                        operand=state
                    ),
                    lambda state: jax.random.randint(x_key, (), state.river_left[0] + 8, state.river_right[0] - 8),
                    operand=state
                ),
                lambda state: jax.lax.cond( # logic for plane (select a screenside)
                    jax.random.bernoulli(x_key, 0.5),
                    lambda _: jnp.array(-10, dtype=jnp.int32),
                    lambda _: jnp.array(self.consts.SCREEN_WIDTH + 10, dtype=jnp.int32),
                    operand=None
                ),
                operand=state
            ),
            lambda state: jnp.array(-1, dtype=jnp.int32),
            operand=state
        )
        new_enemy_y = jax.lax.cond(
            free_enemy_idx >= 0,
            lambda _: jnp.array(-20, dtype=jnp.float32), # spawn enemies outside the window so they don't pop up suddenly
            lambda _: state.enemy_y[free_enemy_idx],
            operand=None
        )
        new_enemy_direction = jax.lax.cond(
            free_enemy_idx >= 0,
            lambda _: jax.random.randint(spawn_key, (), 0, 2), # 0 left, 1 right
            lambda _: state.enemy_direction[free_enemy_idx],
            operand=None
        )

        new_state = jax.lax.cond(
            free_enemy_idx >= 0,
            lambda new_state: new_state.replace(
                enemy_x=state.enemy_x.at[free_enemy_idx].set(new_enemy_x.astype(jnp.float32)),
                enemy_y=state.enemy_y.at[free_enemy_idx].set(new_enemy_y.astype(jnp.float32)),
                enemy_direction=state.enemy_direction.at[free_enemy_idx].set(new_enemy_direction),
                enemy_state=new_enemy_state,
                enemy_type=new_enemy_type
            ),
            lambda state: state,
            operand=state
        )
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def spawn_fuel(self, state: RiverraidState) -> RiverraidState:
        key, x_key = jax.random.split(state.master_key, 2)

        free_fuel_idx = jax.lax.cond(
            jnp.any(state.fuel_state == 0),
            lambda state: jnp.argmax(state.fuel_state == 0),
            lambda _: jnp.array(-1, dtype=jnp.int32),
            operand=state
        )

        new_fuel_state = state.fuel_state.at[free_fuel_idx].set(jnp.array(1))

        new_fuel_x = jax.lax.cond(
            free_fuel_idx >= 0,
            lambda state: jax.lax.cond(
                    state.river_inner_left[0] >= 0,
                    lambda state: jax.lax.cond(
                        jax.random.bernoulli(x_key, 0.5),
                                lambda state: jax.random.randint(x_key, (), state.river_left[0] + 1, state.river_inner_left[0] - 8),
                                lambda state: jax.random.randint(x_key, (), state.river_inner_right[0] + 1, state.river_right[0] - 8),
                        operand=state
                    ),
                    lambda state: jax.random.randint(x_key, (), state.river_left[0] + 8, state.river_right[0] - 8),
                    operand=state
                ),
            lambda state: jnp.array(-1, dtype=jnp.int32),
            operand=state
        )
        new_fuel_y = jax.lax.cond(
            free_fuel_idx >= 0,
            lambda _: jnp.array(-20, dtype=jnp.float32), # spawn fuels outside the window so they don't pop up suddenly
            lambda _: state.fuel_y[free_fuel_idx],
            operand=None
        )

        return state.replace(fuel_state=new_fuel_state,
                                fuel_x=state.fuel_x.at[free_fuel_idx].set(new_fuel_x.astype(jnp.float32)),
                                fuel_y=state.fuel_y.at[free_fuel_idx].set(new_fuel_y.astype(jnp.float32)),
                                master_key=key)

    @partial(jax.jit, static_argnums=(0,))
    def spawn_entities(self, state: RiverraidState) -> RiverraidState:
        key, subkey1, subkey2 = jax.random.split(state.master_key, 3)

        def spawn_entity(state: RiverraidState) -> RiverraidState:
            spawn_fuel_flag = jax.random.bernoulli(subkey2, self.consts.FUEL_SPAWN_PROP)
            return jax.lax.cond(
                spawn_fuel_flag,
                lambda state: self.spawn_fuel(state),
                lambda state: self.spawn_enemies(state),
                operand=state
            )
        # only spawn if no dam in the top 10 rows
        dam_at_top = jnp.any(state.dam_position[:50] >= 1)
        spawn_new_entity = jnp.logical_and(
            jax.random.bernoulli(subkey1, self.consts.ENTITY_SPAWN_PROP * state.player_speed),
            ~dam_at_top
        )

        new_state = jax.lax.cond(
            jnp.logical_and(state.spawn_cooldown <= 0, spawn_new_entity),
            lambda state: spawn_entity(state),
            lambda state: state,
            operand=state
        )

        new_spawn_cooldown = jax.lax.cond(
            jnp.logical_and(state.spawn_cooldown <= 0, spawn_new_entity),
            lambda _: jnp.array(self.consts.MINIMUM_SPAWN_COOLDOWN),
            lambda _: state.spawn_cooldown - 1 * state.player_speed,
            operand=None
        )

        return new_state.replace(master_key=key,
                                  spawn_cooldown=new_spawn_cooldown)

    @partial(jax.jit, static_argnums=(0,))
    def scroll_entities(self, state: RiverraidState) -> RiverraidState:
        limit = self.consts.SCREEN_HEIGHT - self.consts.UI_HEIGHT
        new_enemy_y = state.enemy_y + state.player_speed
        new_enemy_state = jnp.where(new_enemy_y > limit, 0, state.enemy_state)
        new_enemy_x = jnp.where(new_enemy_y > limit, -1, state.enemy_x)

        new_fuel_y = state.fuel_y + state.player_speed
        new_fuel_state = jnp.where(new_fuel_y > limit, 0, state.fuel_state)
        new_fuel_x = jnp.where(new_fuel_y > limit, -1, state.fuel_x)

        return state.replace(
            enemy_y=new_enemy_y,
            enemy_state=new_enemy_state,
            enemy_x=new_enemy_x,
            fuel_y=new_fuel_y,
            fuel_state=new_fuel_state,
            fuel_x=new_fuel_x
        )

    @partial(jax.jit, static_argnums=(0,))
    def enemy_collision(self, state: RiverraidState) -> RiverraidState:
        def handle_bullet_collision(state: RiverraidState) -> RiverraidState:
            enemy_hitboxes = jnp.array([
                13,  # boat
                9,   # helicopter
                6    # plane
            ])
            active_enemy_mask = state.enemy_state == 1

            # hitbox for boat and plane
            hitbox_widths = enemy_hitboxes[state.enemy_type]
            hitbox_starts_x = state.enemy_x
            hitbox_ends_x = state.enemy_x + hitbox_widths

            # specific hitbox for helicopter
            is_helicopter_mask = state.enemy_type == 1
            # Directions: 0/2 are left-facing, 1/3 are right-facing
            is_facing_right_mask = (state.enemy_direction == 1) | (state.enemy_direction == 3)
            should_offset_helicopter = is_helicopter_mask & is_facing_right_mask
            helicopter_offset = 4.0
            hitbox_starts_x = jnp.where(should_offset_helicopter, hitbox_starts_x + helicopter_offset, hitbox_starts_x)
            hitbox_ends_x = jnp.where(should_offset_helicopter, hitbox_ends_x + helicopter_offset, hitbox_ends_x)

            # collision check
            x_collision_mask = (state.player_bullet_x > hitbox_starts_x) & (state.player_bullet_x < hitbox_ends_x)
            y_collision_mask = (state.player_bullet_y < state.enemy_y + 6) & (state.player_bullet_y + 1 > state.enemy_y)

            collision_mask = active_enemy_mask & x_collision_mask & y_collision_mask
            collision_present = jnp.any(collision_mask)
            hit_index = jnp.argmax(collision_mask)

            new_enemy_state = jnp.where(
                collision_present,
                state.enemy_state.at[hit_index].set(2),
                state.enemy_state
            )
            new_score = jnp.where(
                collision_present,
                state.player_score + jax.lax.switch(
                    state.enemy_type[hit_index],
                    [
                        lambda: 30,  # ship
                        lambda: 60,  # helicopter
                        lambda: 100  # plane
                    ]
                ),
                state.player_score
            )

            new_bullet_x = jnp.where(collision_present, -1.0, state.player_bullet_x)
            new_bullet_y = jnp.where(collision_present, -1.0, state.player_bullet_y)

            return state.replace(
                enemy_state=new_enemy_state,
                player_bullet_x=new_bullet_x,
                player_bullet_y=new_bullet_y,
                player_score=new_score
            )

        # Bullet - Enemy Collision only when bullet present
        new_state = jax.lax.cond(
            state.player_bullet_y >= 0,
            lambda state: handle_bullet_collision(state),
            lambda state: state,
            state
        )

        # Player - Enemy Collision
        active_enemy_mask = new_state.enemy_state == 1

        x_collision = (new_state.player_x < new_state.enemy_x + 8) & (new_state.player_x + 8 > new_state.enemy_x)
        y_collision = (new_state.player_y < new_state.enemy_y + 8) & (new_state.player_y + 8 > new_state.enemy_y)
        collision_mask = active_enemy_mask & x_collision & y_collision

        collision_present = jnp.any(collision_mask)
        new_player_state = jnp.where(collision_present, 1, new_state.player_state)
        return new_state.replace(player_state=new_player_state)

    @partial(jax.jit, static_argnums=(0,))
    def handle_animations(self, state: RiverraidState) -> RiverraidState:
        # enemies
        enemy_animation_mask = state.enemy_state > 1
        enemy_cooldown_positive_mask = state.enemy_animation_cooldowns > 0

        new_enemy_cooldowns = state.enemy_animation_cooldowns - 1

        enemy_state_change_mask = enemy_animation_mask & (new_enemy_cooldowns <= 0)

        new_enemy_state = jnp.where(enemy_state_change_mask & (state.enemy_state == 2), 3, state.enemy_state)
        new_enemy_state = jnp.where(enemy_state_change_mask & (state.enemy_state == 3), 4, new_enemy_state)
        new_enemy_state = jnp.where(enemy_state_change_mask & (state.enemy_state == 4), 0, new_enemy_state)

        new_enemy_cooldowns = jnp.where(enemy_state_change_mask, 10, new_enemy_cooldowns)

        # fuel
        fuel_animation_mask = state.fuel_state > 1
        new_fuel_cooldowns = state.fuel_animation_cooldowns - 1
        fuel_state_change_mask = fuel_animation_mask & (new_fuel_cooldowns <= 0)

        new_fuel_state = jnp.where(fuel_state_change_mask & (state.fuel_state == 2), 3, state.fuel_state)
        new_fuel_state = jnp.where(fuel_state_change_mask & (state.fuel_state == 3), 4, new_fuel_state)
        new_fuel_state = jnp.where(fuel_state_change_mask & (state.fuel_state == 4), 0, new_fuel_state)

        new_fuel_cooldowns = jnp.where(fuel_state_change_mask, 10, new_fuel_cooldowns)

        # dam
        new_dam_explosion_cooldown = jax.lax.cond(
            jnp.any(state.dam_position == 2) & (state.dam_explosion_cooldown > 0),
            lambda: state.dam_explosion_cooldown - 1,
            lambda: state.dam_explosion_cooldown
        )

        return state.replace(
            enemy_state=new_enemy_state,
            enemy_animation_cooldowns=new_enemy_cooldowns,
            fuel_state=new_fuel_state,
            fuel_animation_cooldowns=new_fuel_cooldowns,
            dam_explosion_cooldown=new_dam_explosion_cooldown
        )

    @partial(jax.jit, static_argnums=(0,))
    def handle_fuel(self, state: RiverraidState) -> RiverraidState:
        active_fuel_mask = state.fuel_state == 1

        # player collision
        x_collision_mask = (state.player_x < state.fuel_x + 8) & (state.player_x + 8 > state.fuel_x)
        y_collision_mask = (state.player_y < state.fuel_y + 8) & (state.player_y + 8 > state.fuel_y)
        player_collision_mask = active_fuel_mask & x_collision_mask & y_collision_mask
        player_collision_present = jnp.any(player_collision_mask)

        # bullet collision
        bullet_x_collision_mask = (state.player_bullet_x < state.fuel_x + 12) & (state.player_bullet_x + 0 > state.fuel_x)
        bullet_y_collision_mask = (state.player_bullet_y < state.fuel_y + 24) & (state.player_bullet_y > state.fuel_y)
        bullet_collision_mask = active_fuel_mask & bullet_x_collision_mask & bullet_y_collision_mask & (state.player_bullet_y >= 0)
        bullet_collision_present = jnp.any(bullet_collision_mask)
        bullet_hit_index = jnp.argmax(bullet_collision_mask)

        new_fuel_state = jnp.where(
            bullet_collision_present,
            state.fuel_state.at[bullet_hit_index].set(2),
            state.fuel_state
        )
        new_score = jnp.where(
            bullet_collision_present,
            state.player_score + 80,
            state.player_score
        )

        new_bullet_x = jnp.where(bullet_collision_present, -1.0, state.player_bullet_x)
        new_bullet_y = jnp.where(bullet_collision_present, -1.0, state.player_bullet_y)

        new_player_fuel = jax.lax.cond(
            player_collision_present,
            lambda state: jax.lax.cond(
                state.turn_step % 2 == 0,
                lambda state: jnp.clip(state.player_fuel + 1, 0, self.consts.MAX_FUEL),
                lambda state: state.player_fuel,
                operand=state
            ),
            lambda state: jax.lax.cond(
                state.turn_step % 50 == 0,
                lambda state: jnp.clip(state.player_fuel - 1, 0, self.consts.MAX_FUEL),
                lambda state: state.player_fuel,
                operand=state
            ),
            operand=state
        )

        new_player_state = jax.lax.cond(state.player_fuel <= 0,
                                        lambda state: 1,
                                        lambda state: state.player_state,
                                        operand=state)

        return state.replace(
            fuel_state=new_fuel_state,
            player_bullet_x=new_bullet_x,
            player_bullet_y=new_bullet_y,
            player_fuel=new_player_fuel,
            player_score=new_score,
            player_state=new_player_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_enemy_movement_status(self, state: RiverraidState) -> RiverraidState:
        active_static_mask = (state.enemy_state == 1) & (state.enemy_direction <= 1)
        key, subkey = jax.random.split(state.master_key, 2)

        base_prob = self.consts.ENEMY_START_MOVING_PROP * state.player_speed
        plane_prob_multiplier = 10.0
        individual_probs = jnp.where(
            state.enemy_type == 2,  # plane
            base_prob * plane_prob_multiplier,
            base_prob
        )
        individual_probs = jnp.clip(individual_probs, 0.0, 1.0)
        should_change = jax.random.bernoulli(subkey, individual_probs, shape=(self.consts.MAX_ENEMIES,))
        should_change = should_change & active_static_mask

        new_direction = jnp.where(
            state.enemy_direction == 0,
            jnp.array(2),  # left static -> left moving
            jnp.array(3)  # right static -> right moving
        )

        updated_direction = jnp.where(
            should_change,
            new_direction,
            state.enemy_direction
        )

        return state.replace(enemy_direction=updated_direction, master_key=key)

    @partial(jax.jit, static_argnums=(0,))
    def enemy_movement(self, state: RiverraidState) -> RiverraidState:
        move_left_mask = (state.enemy_state == 1) & (state.enemy_direction == 2)
        move_right_mask = (state.enemy_state == 1) & (state.enemy_direction == 3)

        new_enemy_x = state.enemy_x.copy()
        new_enemy_x = jnp.where(move_left_mask, new_enemy_x - 0.5, new_enemy_x)
        new_enemy_x = jnp.where(move_right_mask, new_enemy_x + 0.5, new_enemy_x)

        enemy_y = state.enemy_y.astype(jnp.int32)
        enemy_width = 8

        # Collision with the outer river banks
        collides_with_outer_left = (new_enemy_x <= state.river_left[enemy_y]) & move_left_mask
        collides_with_outer_right = (new_enemy_x + enemy_width >= state.river_right[enemy_y]) & move_right_mask

        # Collision with the inner island shores
        island_present_at_y = state.river_inner_left[enemy_y] >= 0
        was_left_of_island = state.enemy_x + enemy_width < state.river_inner_left[enemy_y]
        collides_with_inner_left_shore = island_present_at_y & was_left_of_island & \
                                         (new_enemy_x + enemy_width >= state.river_inner_left[enemy_y]) & move_right_mask

        was_right_of_island = state.enemy_x > state.river_inner_right[enemy_y]
        collides_with_inner_right_shore = island_present_at_y & was_right_of_island & \
                                          (new_enemy_x <= state.river_inner_right[enemy_y]) & move_left_mask

        left_side_collision = (collides_with_outer_left | collides_with_inner_right_shore)
        right_side_collision = (collides_with_outer_right | collides_with_inner_left_shore)

        change_direction_mask = left_side_collision | right_side_collision
        new_enemy_x = jnp.where(
            left_side_collision & (state.enemy_type != 2),
            jnp.maximum(new_enemy_x + 1.0,
                        jnp.where(collides_with_inner_right_shore,
                                  state.river_inner_right[enemy_y] + 1.0,
                                  state.river_left[enemy_y] + 1.0)),
            new_enemy_x
        )

        new_enemy_x = jnp.where(
            right_side_collision & (state.enemy_type != 2),
            jnp.minimum(new_enemy_x - 1.0,
                        jnp.where(collides_with_inner_left_shore,
                                  state.river_inner_left[enemy_y] - enemy_width - 1.0,
                                  state.river_right[enemy_y] - enemy_width - 1.0)),
            new_enemy_x
        )

        new_enemy_direction = jnp.where(
            change_direction_mask & (state.enemy_type != 2),  # Planes (type 2) fly off-screen and don't turn
            jnp.where(state.enemy_direction == 2, 3, 2),  # Flip direction from left (2) to right (3) or vice-versa
            state.enemy_direction
        )
        return state.replace(enemy_x=new_enemy_x, enemy_direction=new_enemy_direction)

    @partial(jax.jit, static_argnums=(0,))
    def handle_housetree(self, state: RiverraidState) -> RiverraidState:
        new_housetree_y = state.housetree_y + state.player_speed

        # remove off-screen housetrees
        off_screen_mask = new_housetree_y > self.consts.SCREEN_HEIGHT - self.consts.UI_HEIGHT
        new_housetree_state = jnp.where(off_screen_mask, 0, state.housetree_state)

        new_state = state.replace(
            housetree_y=new_housetree_y,
            housetree_state=new_housetree_state,
            housetree_cooldown=state.housetree_cooldown - 1 * state.player_speed
        )

        spawn_key, side_key, direction_key = jax.random.split(state.master_key, 3)

        def spawn_housetree(state: RiverraidState) -> RiverraidState:
            free_idx = jnp.argmax(state.housetree_state == 0)
            side = jax.random.choice(side_key, jnp.array([1, 2]))  # 1: left, 2: right
            direction = jax.random.choice(direction_key, jnp.array([1, 2]))  # 1: left-facing, 2: right-facing
            x_pos = jax.lax.cond(side == 1,
                                 lambda _: 2,
                                 lambda _: self.consts.SCREEN_WIDTH - 18,
                                 operand=None)

            # spawn the new housetree
            updated_x = state.housetree_x.at[free_idx].set(x_pos)
            updated_y = state.housetree_y.at[free_idx].set(-15)
            updated_state = state.housetree_state.at[free_idx].set(1)
            updated_side = state.housetree_side.at[free_idx].set(side)
            updated_direction = state.housetree_direction.at[free_idx].set(direction)

            return state.replace(
                housetree_x=updated_x,
                housetree_y=updated_y,
                housetree_state=updated_state,
                housetree_side=updated_side,
                housetree_direction=updated_direction,
                housetree_cooldown=jnp.array(25)
            )

        free_slot = jnp.any(new_state.housetree_state == 0) # free slot?
        free_area = jnp.logical_and(
            state.river_right[0] - state.river_left[0] < self.consts.MAX_RIVER_WIDTH - 30,
            ~jnp.any(state.dam_position[:50] >= 1)
        )
        can_spawn = jnp.logical_and(free_slot, free_area)
        spawn_new_housetree = jax.random.bernoulli(spawn_key, p=0.05)

        new_state = jax.lax.cond(
            jnp.logical_and(spawn_new_housetree,
                            jnp.logical_and(new_state.housetree_cooldown <= 0,
                                            can_spawn)),
            lambda state: spawn_housetree(state),
            lambda state: state,
            operand=new_state
        )
        return new_state.replace(master_key=spawn_key)

    @partial(jax.jit, static_argnums=(0,))
    def adjust_player_speed(self, state: RiverraidState, action: chex.Array) -> RiverraidState:
        press_up = jnp.any(jnp.array([
            action == Action.UP,
            action == Action.UPFIRE,
            action == Action.UPRIGHT,
            action == Action.UPLEFT,
            action == Action.UPRIGHTFIRE,
            action == Action.UPLEFTFIRE
        ]))

        press_down = jnp.any(jnp.array([
            action == Action.DOWN,
            action == Action.DOWNFIRE,
            action == Action.DOWNRIGHT,
            action == Action.DOWNLEFT,
            action == Action.DOWNRIGHTFIRE,
            action == Action.DOWNLEFTFIRE
        ]))

        can_change_speed = state.player_speedchange_cooldown <= 0
        speed_change = jax.lax.cond(
            can_change_speed,
            lambda: jax.lax.cond(
                press_up,
                lambda: jnp.array(1),
                lambda: jax.lax.cond(
                    press_down,
                    lambda: jnp.array(-1),
                    lambda: jnp.array(0)
                )
            ),
            lambda: jnp.array(0)
        )

        new_speed = jnp.clip(state.player_speed + speed_change, 1, 3)
        speed_changed = new_speed != state.player_speed

        new_cooldown = jax.lax.cond(
            speed_changed,
            lambda: jnp.array(self.consts.SPEED_CHANGE_COOLDOWN),  # Cooldown von 30 Frames
            lambda: jnp.maximum(state.player_speedchange_cooldown - 1, 0)
        )

        return state.replace(
            player_speed=new_speed,
            player_speedchange_cooldown=new_cooldown
        )

    def render(self, state: RiverraidState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: RiverraidState) -> RiverraidObservation:
        c = self.consts
        w, h = int(c.SCREEN_WIDTH), int(c.SCREEN_HEIGHT)

        # --- Player ---
        # Map direction 0=Left, 1=Straight, 2=Right
        p_ori = jax.lax.select(
            state.player_direction == 0, 270.0,
            jax.lax.select(state.player_direction == 2, 90.0, 0.0)
        ).astype(jnp.float32)
        
        player = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.player_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(state.player_y, dtype=jnp.int32), 0, h),
            width=jnp.array(c.PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.array(c.PLAYER_HEIGHT, dtype=jnp.int32),
            active=((state.player_state == 0).astype(jnp.int32)),
            orientation=jnp.array(p_ori, dtype=jnp.float32)
        )

        # --- Enemies ---
        # Orientation: 0/2=Left, 1/3=Right
        e_dirs = state.enemy_direction
        e_ori = jnp.where((e_dirs == 1) | (e_dirs == 3), 90.0, 270.0).astype(jnp.float32)
        
        enemy_widths = jnp.array([13, 9, 6])[state.enemy_type]

        enemies = ObjectObservation.create(
            x=jnp.clip(state.enemy_x.astype(jnp.int32), 0, w),
            y=jnp.clip(state.enemy_y.astype(jnp.int32), 0, h),
            width=enemy_widths.astype(jnp.int32),
            height=jnp.full((c.MAX_ENEMIES,), 6, dtype=jnp.int32),
            active=(state.enemy_state == 1).astype(jnp.int32),
            visual_id=state.enemy_type.astype(jnp.int32),
            orientation=e_ori
        )

        # --- Fuel Tanks ---
        fuel_tanks = ObjectObservation.create(
            x=jnp.clip(state.fuel_x.astype(jnp.int32), 0, w),
            y=jnp.clip(state.fuel_y.astype(jnp.int32), 0, h),
            width=jnp.full((c.MAX_ENEMIES,), 8, dtype=jnp.int32),
            height=jnp.full((c.MAX_ENEMIES,), 12, dtype=jnp.int32),
            active=(state.fuel_state == 1).astype(jnp.int32)
        )

        # --- Player Projectile ---
        bullet_active = (state.player_bullet_y > -1).astype(jnp.int32)
        player_projectile = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.player_bullet_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(state.player_bullet_y, dtype=jnp.int32), 0, h),
            width=jnp.array(2, dtype=jnp.int32),
            height=jnp.array(4, dtype=jnp.int32),
            active=jnp.array(bullet_active, dtype=jnp.int32)
        )

        # --- House/Trees ---
        house_trees = ObjectObservation.create(
            x=jnp.clip(state.housetree_x.astype(jnp.int32), 0, w),
            y=jnp.clip(state.housetree_y.astype(jnp.int32), 0, h),
            width=jnp.full((c.MAX_HOUSE_TREES,), 8, dtype=jnp.int32),
            height=jnp.full((c.MAX_HOUSE_TREES,), 8, dtype=jnp.int32),
            active=(state.housetree_state == 1).astype(jnp.int32)
        )

        return RiverraidObservation(
            player=player,
            enemies=enemies,
            fuel_tanks=fuel_tanks,
            player_projectile=player_projectile,
            house_trees=house_trees,
            river_left_bank=state.river_left,
            river_right_bank=state.river_right,
            island_left_bank=state.river_inner_left,
            island_right_bank=state.river_inner_right,
            dam_indicator=state.dam_position,
            player_fuel=state.player_fuel,
            player_lives=state.player_lives,
            player_score=state.player_score,
            player_speed=state.player_speed
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3),
            dtype=jnp.uint8
        )

    def reset(self, key=None) -> Tuple[RiverraidObservation, RiverraidState]:
        river_start_x = (self.consts.SCREEN_WIDTH - self.consts.DEFAULT_RIVER_WIDTH) // 2
        river_end_x = river_start_x + self.consts.DEFAULT_RIVER_WIDTH
        initial_key = jax.random.PRNGKey(1)

        state = RiverraidState(turn_step=0,
                               turn_step_linear=0,
                               river_left=jnp.full((self.consts.SCREEN_HEIGHT,), river_start_x, dtype=jnp.int32),
                               river_right=jnp.full((self.consts.SCREEN_HEIGHT,), river_end_x, dtype=jnp.int32),
                               river_inner_left=jnp.full((self.consts.SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                               river_inner_right=jnp.full((self.consts.SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                               river_state=jnp.array(0),
                               river_alternation_length=jnp.array(0),
                               master_key=initial_key,
                               river_ongoing_alternation=jnp.array(0),
                               river_island_present=jnp.array(0),
                               alternation_cooldown=jnp.array(10),
                               island_transition_state=jnp.array(0),
                               segment_state=jnp.array(0),
                               segment_transition_state=jnp.array(0),
                               segment_straigt_counter=jnp.array(8),
                               dam_position= jnp.full((self.consts.SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                               player_x= jnp.array(self.consts.SCREEN_WIDTH // 2 - 2, dtype=jnp.float32),
                               player_y=jnp.array(self.consts.SCREEN_HEIGHT - 20 - self.consts.UI_HEIGHT, dtype=jnp.float32),
                               player_velocity=jnp.array(0, dtype=jnp.float32),
                               player_direction=jnp.array(1),
                               player_state= jnp.array(0),
                               player_bullet_x= jnp.array(-1, dtype=jnp.float32),
                               player_bullet_y= jnp.array(-1, dtype=jnp.float32),
                               player_speed= jnp.array(1),
                               player_speedchange_cooldown= jnp.array(10),
                               enemy_x=jnp.full((self.consts.MAX_ENEMIES,), -1, dtype=jnp.float32),
                               enemy_y=jnp.full((self.consts.MAX_ENEMIES,), self.consts.SCREEN_HEIGHT + 1, dtype=jnp.float32),
                               enemy_state=jnp.full((self.consts.MAX_ENEMIES,), 0, dtype=jnp.int32),
                               enemy_type= jnp.full((self.consts.MAX_ENEMIES,), 0, dtype=jnp.int32),
                               enemy_direction= jnp.full((self.consts.MAX_ENEMIES,), 0, dtype=jnp.int32),
                               fuel_x=jnp.full((self.consts.MAX_ENEMIES,), -1, dtype=jnp.float32),
                               fuel_y=jnp.full((self.consts.MAX_ENEMIES,), self.consts.SCREEN_HEIGHT + 1, dtype=jnp.float32),
                               fuel_state=jnp.full((self.consts.MAX_ENEMIES,), 0, dtype=jnp.int32),
                               player_fuel=jnp.array(self.consts.MAX_FUEL),
                               spawn_cooldown=jnp.array(50),
                               player_score=jnp.array(0),
                               player_lives=jnp.array(3),
                               housetree_x=jnp.full((self.consts.MAX_HOUSE_TREES,), -1, dtype=jnp.float32),
                               housetree_y=jnp.full((self.consts.MAX_HOUSE_TREES,), -1, dtype=jnp.float32),
                               housetree_state=jnp.full((self.consts.MAX_HOUSE_TREES,), 0, dtype=jnp.int32),
                               housetree_side=jnp.full((self.consts.MAX_HOUSE_TREES,), 0, dtype=jnp.int32),
                               housetree_direction=jnp.full((self.consts.MAX_HOUSE_TREES,), 0, dtype=jnp.int32),
                               housetree_cooldown=jnp.array(20),
                               enemy_animation_cooldowns=jnp.full(self.consts.MAX_ENEMIES, 3),
                               fuel_animation_cooldowns=jnp.full(self.consts.MAX_ENEMIES, 3),
                               death_cooldown=jnp.array(self.consts.DEATH_COOLDOWN),
                               dam_explosion_cooldown=jnp.array(15)
                               )
        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: RiverraidState, action: int) -> Tuple[RiverraidObservation, RiverraidState, RiverraidInfo]:
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))
        
        def player_alive(state: RiverraidState) -> RiverraidState:
            new_state = self.adjust_player_speed(state, atari_action)
            new_state = self.handle_dam(new_state)
            new_state = self.river_generation(new_state)
            new_state = self.player_movement(new_state, atari_action)
            new_state = self.player_shooting(new_state, atari_action)
            new_state = self.spawn_entities(new_state)
            new_state = self.scroll_entities(new_state)
            new_state = self.enemy_collision(new_state)
            new_state = self.handle_animations(new_state)
            new_state = self.update_enemy_movement_status(new_state)
            new_state = self.enemy_movement(new_state)
            new_state = self.handle_fuel(new_state)
            new_state = self.handle_housetree(new_state)

            # new life every 10.000 points
            previous_life_threshold = state.player_score // 10000
            current_life_threshold = new_state.player_score // 10000
            earned_extra_life = current_life_threshold > previous_life_threshold
            new_player_lives = jnp.where(earned_extra_life, state.player_lives + 1, state.player_lives)
            new_state = new_state.replace(player_lives=new_player_lives)
            return new_state

        def respawn(state: RiverraidState) -> RiverraidState:
            river_start_x = (self.consts.SCREEN_WIDTH - self.consts.DEFAULT_RIVER_WIDTH) // 2
            river_end_x = river_start_x + self.consts.DEFAULT_RIVER_WIDTH
            initial_key = jax.random.PRNGKey(1)
            new_segment_state = jax.lax.cond(
                jnp.logical_or(state.segment_state == 0, state.segment_state == 1), # if altering or about to change
                lambda _: jnp.array(0), # set to the end of altering (end ensuured by turn step)
                lambda _: jnp.array(2), # else to the end of straight
                operand=None
            )
            new_turn_step = state.turn_step - (state.turn_step % self.consts.SEGMENT_LENGTH + 50) # respawn at the start of the last segment + some offset

            new_state = RiverraidState(turn_step= new_turn_step,
                                   turn_step_linear=state.turn_step_linear,
                                   river_left=jnp.full((self.consts.SCREEN_HEIGHT,), river_start_x, dtype=jnp.int32),
                                   river_right=jnp.full((self.consts.SCREEN_HEIGHT,), river_end_x, dtype=jnp.int32),
                                   river_inner_left=jnp.full((self.consts.SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                                   river_inner_right=jnp.full((self.consts.SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                                   river_state=jnp.array(0),
                                   river_alternation_length=jnp.array(0),
                                   master_key=initial_key,
                                   river_ongoing_alternation=jnp.array(0),
                                   river_island_present=jnp.array(0),
                                   alternation_cooldown=jnp.array(10),
                                   island_transition_state=jnp.array(0),
                                   segment_state=new_segment_state,
                                   segment_transition_state=jnp.array(0),
                                   segment_straigt_counter=jnp.array(8),
                                   dam_position=jnp.full((self.consts.SCREEN_HEIGHT,), -1, dtype=jnp.int32),
                                   player_x=jnp.array(self.consts.SCREEN_WIDTH // 2 - 2, dtype=jnp.float32),
                                   player_y=jnp.array(self.consts.SCREEN_HEIGHT - 20 - self.consts.UI_HEIGHT, dtype=jnp.float32),
                                   player_velocity=jnp.array(0, dtype=jnp.float32),
                                   player_direction=jnp.array(1),
                                   player_state=jnp.array(0),
                                   player_bullet_x=jnp.array(-1, dtype=jnp.float32),
                                   player_bullet_y=jnp.array(-1, dtype=jnp.float32),
                                   player_speed=jnp.array(1),
                                   player_speedchange_cooldown=jnp.array(10),
                                   enemy_x=jnp.full((self.consts.MAX_ENEMIES,), -1, dtype=jnp.float32),
                                   enemy_y=jnp.full((self.consts.MAX_ENEMIES,), self.consts.SCREEN_HEIGHT + 1, dtype=jnp.float32),
                                   enemy_state=jnp.full((self.consts.MAX_ENEMIES,), 0, dtype=jnp.int32),
                                   enemy_type=jnp.full((self.consts.MAX_ENEMIES,), 0, dtype=jnp.int32),
                                   enemy_direction=jnp.full((self.consts.MAX_ENEMIES,), 0, dtype=jnp.int32),
                                   fuel_x=jnp.full((self.consts.MAX_ENEMIES,), -1, dtype=jnp.float32),
                                   fuel_y=jnp.full((self.consts.MAX_ENEMIES,), self.consts.SCREEN_HEIGHT + 1, dtype=jnp.float32),
                                   fuel_state=jnp.full((self.consts.MAX_ENEMIES,), 0, dtype=jnp.int32),
                                   player_fuel=jnp.array(self.consts.MAX_FUEL),
                                   spawn_cooldown=jnp.array(50),
                                   player_score=state.player_score,
                                   player_lives= state.player_lives - 1,
                                   housetree_x=jnp.full((self.consts.MAX_HOUSE_TREES,), -1, dtype=jnp.float32),
                                   housetree_y=jnp.full((self.consts.MAX_HOUSE_TREES,), -1, dtype=jnp.float32),
                                   housetree_state=jnp.full((self.consts.MAX_HOUSE_TREES,), 0, dtype=jnp.int32),
                                   housetree_side=jnp.full((self.consts.MAX_HOUSE_TREES,), 0, dtype=jnp.int32),
                                   housetree_direction=jnp.full((self.consts.MAX_HOUSE_TREES,), 0, dtype=jnp.int32),
                                   housetree_cooldown=jnp.array(20),
                                   enemy_animation_cooldowns=jnp.full(self.consts.MAX_ENEMIES, 3),
                                   fuel_animation_cooldowns=jnp.full(self.consts.MAX_ENEMIES, 3),
                                   death_cooldown=jnp.array(self.consts.DEATH_COOLDOWN),
                                   dam_explosion_cooldown=jnp.array(15)
                                   )
            return new_state

        def delay_respawn(state: RiverraidState) -> RiverraidState:
            new_death_cooldown = jnp.maximum(state.death_cooldown - 1, 0)
            return jax.lax.cond(
                new_death_cooldown <= 0,
                lambda state: respawn(state),
                lambda state: state.replace(death_cooldown=new_death_cooldown),
                operand=state
            )

        new_state = state.replace(turn_step=state.turn_step + 1,
                                   turn_step_linear=state.turn_step_linear + 1)

        new_player_state = jax.lax.cond(state.player_lives <= 0,
                                        lambda _: jnp.array(2),
                                        lambda _: new_state.player_state,
                                        operand=None
                                        )
        new_state = new_state.replace(player_state=new_player_state)  # game over

        new_state = jax.lax.cond(
            new_state.player_state == 0,
            lambda new_state: player_alive(new_state),
            lambda new_state: jax.lax.cond(
                new_state.player_state == 1,
                lambda new_state: delay_respawn(new_state),
                lambda new_state: new_state,
                operand=new_state
            ),
            operand = new_state
        )

        observation = self._get_observation(new_state)
        env_reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: RiverraidState) -> bool:
        return state.player_lives <= 0

    def observation_space(self) -> spaces.Dict:
        # Use concrete Python ints to avoid TracerArrayConversionError in Box.
        h = int(self.consts.SCREEN_HEIGHT)
        w = int(self.consts.SCREEN_WIDTH)
        screen_size = (h, w)
        
        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)
        
        return spaces.Dict({
            "player": single_obj,
            "enemies": spaces.get_object_space(n=self.consts.MAX_ENEMIES, screen_size=screen_size),
            "fuel_tanks": spaces.get_object_space(n=self.consts.MAX_ENEMIES, screen_size=screen_size),
            "player_projectile": single_obj,
            "house_trees": spaces.get_object_space(n=self.consts.MAX_HOUSE_TREES, screen_size=screen_size),
            
            "river_left_bank": spaces.Box(low=0, high=w, shape=(h,), dtype=jnp.int32),
            "river_right_bank": spaces.Box(low=0, high=w, shape=(h,), dtype=jnp.int32),
            "island_left_bank": spaces.Box(low=-1, high=w, shape=(h,), dtype=jnp.int32),
            "island_right_bank": spaces.Box(low=-1, high=w, shape=(h,), dtype=jnp.int32),
            "dam_indicator": spaces.Box(low=-1, high=2, shape=(h,), dtype=jnp.int32),
            
            "player_fuel": spaces.Box(low=0, high=self.consts.MAX_FUEL + 50, shape=(), dtype=jnp.float32),
            "player_lives": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
            "player_score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "player_speed": spaces.Box(low=0, high=10, shape=(), dtype=jnp.float32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: RiverraidState, state: RiverraidState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: RiverraidState) -> RiverraidInfo:
        """
        Creates the info object.
        """
        return RiverraidInfo(time=state.turn_step_linear)


class RiverraidRenderer(JAXGameRenderer):
    def __init__(self, consts: RiverraidConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or RiverraidConstants()
        super().__init__(self.consts)
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Define sprite path
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "riverraid")

        # 3. Get asset config from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 4. Load, process, and set up all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND, # This is just a 1x1 black pixel
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

        # 5. Get specific color IDs for procedural drawing
        self.GREEN_BANKS_ID = self.COLOR_TO_ID.get((26, 132, 26), 0)
        self.BLUE_RIVER_ID = self.COLOR_TO_ID.get((42, 42, 189), 0)
        self.RED_RIVER_ID = self.COLOR_TO_ID.get((189, 42, 42), 0)
        self.UI_COLOR_ID = self.COLOR_TO_ID.get((128, 128, 128), 0)
        self.BLACK_ID = self.COLOR_TO_ID.get((0, 0, 0), 0)

        # 6. Cache stacked animation frames
        self.SPRITE_PLAYER_STACK = self.SHAPE_MASKS['player']
        self.SPRITE_HELICOPTER_STACK = self.SHAPE_MASKS['enemy_helicopter']
        self.SPRITE_ENEMY_EXPLOSION_STACK = self.SHAPE_MASKS['enemy_explosion']
        self.SPRITE_DAM_EXPLOSION_STACK = self.SHAPE_MASKS['dam_explosion']

    @partial(jax.jit, static_argnums=(0,))
    def _render_procedural_background(self, state: RiverraidState) -> chex.Array:
        """
        Creates the procedural river/bank background as a 2D ID raster.
        This is JIT-compatible and respects downscaling.
        """
        # 1. Get target dimensions (respects downscaling)
        # Since self is static, we can access config properties directly
        target_h = self.config.downscale[0] if self.config.downscale else self.consts.SCREEN_HEIGHT
        target_w = self.config.downscale[1] if self.config.downscale else self.consts.SCREEN_WIDTH
        height_scale = jnp.array(self.config.height_scaling, dtype=jnp.float32)
        width_scale = jnp.array(self.config.width_scaling, dtype=jnp.float32)
        
        # 2. Get color IDs
        green_banks = self.GREEN_BANKS_ID
        river_color = jax.lax.cond(
            jnp.logical_and(
                state.dam_explosion_cooldown > 0,
                jnp.logical_and(
                    state.dam_explosion_cooldown < 15,
                    state.dam_explosion_cooldown % 2 == 0
                )
            ),
            lambda: self.RED_RIVER_ID,
            lambda: self.BLUE_RIVER_ID,
        )

        # 3. Create coordinates at target resolution
        # (W,)
        x_coords = jnp.arange(target_w)
        # (H,)
        y_coords = jnp.arange(target_h)

        # 4. Scale state arrays to target resolution
        # Scale y-coordinates to sample from state arrays
        y_indices = jnp.clip((y_coords / height_scale).astype(jnp.int32), 0, self.consts.SCREEN_HEIGHT - 1)
        
        # Scale river boundaries to target width
        river_left_scaled = (state.river_left[y_indices] * width_scale).astype(jnp.int32)
        river_right_scaled = (state.river_right[y_indices] * width_scale).astype(jnp.int32)
        river_inner_left_scaled = (state.river_inner_left[y_indices] * width_scale).astype(jnp.int32)
        river_inner_right_scaled = (state.river_inner_right[y_indices] * width_scale).astype(jnp.int32)

        # 5. Create boolean masks for river and banks
        # (H, 1) > (W,) -> (H, W)
        is_left_bank = x_coords <= river_left_scaled[:, None] 
        is_right_bank = x_coords >= river_right_scaled[:, None]
        is_island = (river_inner_left_scaled >= 0)[:, None] & (x_coords >= river_inner_left_scaled[:, None]) & (x_coords <= river_inner_right_scaled[:, None])
        is_bank = is_left_bank | is_right_bank | is_island
        
        # 6. Create base raster of river color at target resolution
        raster = jnp.full((target_h, target_w), river_color, dtype=jnp.uint8)
        
        # 7. Paint banks on top
        raster = jnp.where(is_bank, green_banks, raster)
        
        # 8. Mask out the UI area at the bottom (scaled)
        ui_height_scaled = jnp.round(self.consts.UI_HEIGHT * height_scale).astype(jnp.int32)
        ui_mask = y_coords[:, None] >= (target_h - ui_height_scaled)
        raster = jnp.where(ui_mask, self.UI_COLOR_ID, raster)
        
        # 9. Draw black line between UI and game
        ui_line_y = target_h - ui_height_scaled - 1
        raster = jax.lax.cond(
            ui_line_y >= 0,
            lambda r: r.at[ui_line_y].set(self.BLACK_ID),
            lambda r: r,
            raster
        )
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: RiverraidState):
        # Get scaling factors as JAX arrays for use in JIT context
        height_scale = jnp.array(self.config.height_scaling, dtype=jnp.float32)
        width_scale = jnp.array(self.config.width_scaling, dtype=jnp.float32)
        
        # 1. Create the procedural background (river, banks, UI bar)
        raster = self._render_procedural_background(state)

        # 2. Render the dam sprite
        dam_y = jnp.argmax(state.dam_position >= 1).astype(jnp.int32) - self.consts.DAM_OFFSET
        has_dam = jnp.max(state.dam_position) >= 1

        def render_dam(r):
            # Calculate dam_x in game coordinates
            # Sprite width is already scaled, so we need to unscale it to get original width for centering
            original_sprite_width = jnp.round(self.SHAPE_MASKS['full_dam'].shape[1] / width_scale).astype(jnp.int32)
            dam_x = self.consts.SCREEN_WIDTH // 2 - original_sprite_width // 2 + 5
            
            sprite_to_render = jax.lax.cond(
                state.dam_position[dam_y + self.consts.DAM_OFFSET] == 1,
                lambda: self.SHAPE_MASKS['full_dam'],
                lambda: self.SHAPE_MASKS['outer_dam'],
            )
            
            r = jax.lax.cond(
                has_dam,
                lambda r_in: self.jr.render_at_clipped(r_in, dam_x, dam_y, sprite_to_render, flip_offset=self.FLIP_OFFSETS['full_dam']),
                lambda r_in: r_in,
                r
            )
            street = self.SHAPE_MASKS['street']
            r = jax.lax.cond(
                has_dam,
                lambda r_in: self.jr.render_at_clipped(r_in, 0, dam_y, street, flip_offset=self.FLIP_OFFSETS['street']),
                lambda r_in: r_in,
                r
            )
            explosion_idx = jax.lax.cond(
                (state.dam_explosion_cooldown >= 5) & (state.dam_explosion_cooldown <= 10),
                lambda: 1, # dam_explosion_2
                lambda: 0, # dam_explosion_1
            )
            explosion_to_render = self.SPRITE_DAM_EXPLOSION_STACK[explosion_idx]
            # Center explosion in game coordinates
            explosion_x = self.consts.SCREEN_WIDTH // 2 - 8
            r = jax.lax.cond(
                jnp.logical_and(state.dam_position[dam_y + self.consts.DAM_OFFSET] == 2,
                                state.dam_explosion_cooldown > 0),
                lambda r_in: self.jr.render_at_clipped(r_in, explosion_x, dam_y, explosion_to_render, flip_offset=self.FLIP_OFFSETS['dam_explosion']),
                lambda r_in: r_in,
                r
            )
            return r

        raster = render_dam(raster)

        # 3. Player
        px = jnp.round(state.player_x).astype(jnp.int32)
        py = jnp.round(state.player_y).astype(jnp.int32)
        
        # Render player if alive, explosion if dead
        # Map player_direction (0=left, 1=straight, 2=right) to sprite stack indices (0=center, 1=left, 2=right)
        player_frame_mask = jax.lax.switch(
            state.player_direction,
            [
                lambda: self.SPRITE_PLAYER_STACK[1],  # direction 0 (left) -> sprite index 1 (left)
                lambda: self.SPRITE_PLAYER_STACK[0],  # direction 1 (straight) -> sprite index 0 (center)
                lambda: self.SPRITE_PLAYER_STACK[2],  # direction 2 (right) -> sprite index 2 (right)
            ]
        )
        raster = jax.lax.cond(
            state.player_state == 0, # alive
            lambda r: self.jr.render_at_clipped(r, px, py, player_frame_mask, flip_offset=self.FLIP_OFFSETS['player']),
            lambda r: self.jr.render_at_clipped(r, px, py, self.SHAPE_MASKS['player_explosion'], flip_offset=self.FLIP_OFFSETS['player_explosion']),
            raster
        )

        # 4. Player Bullet
        bullet_frame = self.SHAPE_MASKS['bullet']
        bx = jnp.round(state.player_bullet_x).astype(jnp.int32)
        by = jnp.round(state.player_bullet_y).astype(jnp.int32)
        raster = self.jr.render_at_clipped(raster, bx, by, bullet_frame, flip_offset=self.FLIP_OFFSETS['bullet'])

        # 5. Enemies (Boats, Helis, Airplanes)
        def render_single_enemy_on_raster(i, r_acc):
            enemy_state = state.enemy_state[i]
            enemy_type = state.enemy_type[i]
            enemy_direction = state.enemy_direction[i]
            enemy_x = state.enemy_x[i].astype(jnp.int32)
            enemy_y = state.enemy_y[i].astype(jnp.int32)

            # Helper to get flip flag
            flip_h = jnp.logical_and(enemy_state <= 1, jnp.logical_or(enemy_direction == 0, enemy_direction == 2))
            
            # Render based on enemy type and state
            def render_boat(r):
                # Render explosion or boat sprite directly
                r = jax.lax.cond(
                    enemy_state == 2,
                    lambda r_in: self.jr.render_at_clipped(r_in, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[0], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                    lambda r_in: jax.lax.cond(
                        enemy_state == 3,
                        lambda r_in2: self.jr.render_at_clipped(r_in2, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[1], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                        lambda r_in2: jax.lax.cond(
                            enemy_state == 4,
                            lambda r_in3: self.jr.render_at_clipped(r_in3, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[0], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                            lambda r_in3: self.jr.render_at_clipped(r_in3, enemy_x, enemy_y, self.SHAPE_MASKS['enemy_boat'], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_boat']),
                            r_in2
                        ),
                        r_in
                    ),
                    r
                )
                return r
            
            def render_helicopter(r):
                helicopter_frame_idx = jnp.where((state.turn_step % 20) < 10, 0, 1)
                helicopter_sprite = self.SPRITE_HELICOPTER_STACK[helicopter_frame_idx]
                # Render explosion or helicopter sprite directly
                r = jax.lax.cond(
                    enemy_state == 2,
                    lambda r_in: self.jr.render_at_clipped(r_in, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[0], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                    lambda r_in: jax.lax.cond(
                        enemy_state == 3,
                        lambda r_in2: self.jr.render_at_clipped(r_in2, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[1], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                        lambda r_in2: jax.lax.cond(
                            enemy_state == 4,
                            lambda r_in3: self.jr.render_at_clipped(r_in3, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[0], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                            lambda r_in3: self.jr.render_at_clipped(r_in3, enemy_x, enemy_y, helicopter_sprite, flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_helicopter']),
                            r_in2
                        ),
                        r_in
                    ),
                    r
                )
                return r
            
            def render_airplane(r):
                # Render explosion or airplane sprite directly
                r = jax.lax.cond(
                    enemy_state == 2,
                    lambda r_in: self.jr.render_at_clipped(r_in, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[0], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                    lambda r_in: jax.lax.cond(
                        enemy_state == 3,
                        lambda r_in2: self.jr.render_at_clipped(r_in2, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[1], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                        lambda r_in2: jax.lax.cond(
                            enemy_state == 4,
                            lambda r_in3: self.jr.render_at_clipped(r_in3, enemy_x, enemy_y, self.SPRITE_ENEMY_EXPLOSION_STACK[0], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                            lambda r_in3: self.jr.render_at_clipped(r_in3, enemy_x, enemy_y, self.SHAPE_MASKS['enemy_airplane'], flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['enemy_airplane']),
                            r_in2
                        ),
                        r_in
                    ),
                    r
                )
                return r
            
            # Select which render function to use based on enemy type
            return jax.lax.cond(
                enemy_state > 0,
                lambda r: jax.lax.switch(enemy_type, [render_boat, render_helicopter, render_airplane], r),
                lambda r: r,
                r_acc
            )
        
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES, render_single_enemy_on_raster, raster)

        # 6. Fuel Tanks
        def render_fuel(i, r_acc):
            fuel_state = state.fuel_state[i]
            fuel_x = state.fuel_x[i].astype(jnp.int32)
            fuel_y = state.fuel_y[i].astype(jnp.int32)

            # Render explosion or fuel sprite directly
            return jax.lax.cond(
                fuel_state > 0,
                lambda r_in: jax.lax.cond(
                    fuel_state == 2,
                    lambda r_in2: self.jr.render_at_clipped(r_in2, fuel_x, fuel_y, self.SPRITE_ENEMY_EXPLOSION_STACK[0], flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                    lambda r_in2: jax.lax.cond(
                        fuel_state == 3,
                        lambda r_in3: self.jr.render_at_clipped(r_in3, fuel_x, fuel_y, self.SPRITE_ENEMY_EXPLOSION_STACK[1], flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                        lambda r_in3: jax.lax.cond(
                            fuel_state == 4,
                            lambda r_in4: self.jr.render_at_clipped(r_in4, fuel_x, fuel_y, self.SPRITE_ENEMY_EXPLOSION_STACK[0], flip_offset=self.FLIP_OFFSETS['enemy_explosion']),
                            lambda r_in4: self.jr.render_at_clipped(r_in4, fuel_x, fuel_y, self.SHAPE_MASKS['fuel'], flip_offset=self.FLIP_OFFSETS['fuel']),
                            r_in3
                        ),
                        r_in2
                    ),
                    r_in
                ),
                lambda r_in: r_in,
                r_acc
            )
        
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES, render_fuel, raster)

        # 7. House/Tree Scenery
        def render_single_housetree(i, r_acc):
            is_active = state.housetree_state[i] == 1
            
            def draw_it(r):
                hx = state.housetree_x[i].astype(jnp.int32)
                hy = state.housetree_y[i].astype(jnp.int32)
                housetree_sprite = self.SHAPE_MASKS['house_tree']
                flip_h = state.housetree_direction[i] == 1
                
                return self.jr.render_at_clipped(r, hx, hy, housetree_sprite, flip_horizontal=flip_h, flip_offset=self.FLIP_OFFSETS['house_tree'])
            
            return jax.lax.cond(is_active, draw_it, lambda r: r, r_acc)
        
        raster = jax.lax.fori_loop(0, self.consts.MAX_HOUSE_TREES, render_single_housetree, raster)

        # Clean up UI area (overwrite sprites that went too low)
        target_h = raster.shape[0]
        y_coords = jnp.arange(target_h)
        ui_height_scaled = jnp.round(self.consts.UI_HEIGHT * height_scale).astype(jnp.int32)
        ui_mask = y_coords[:, None] >= (target_h - ui_height_scaled)
        raster = jnp.where(ui_mask, self.UI_COLOR_ID, raster)

        # Redraw black line

        ui_line_y = target_h - ui_height_scaled - 1
        raster = jax.lax.cond(
            ui_line_y >= 0,
            lambda r: r.at[ui_line_y].set(self.BLACK_ID),
            lambda r: r,
            raster
        )

        # 8. UI elements (Fuel display, Score, Lives, Logo)
        
        # Fuel Display - calculate in game coordinates
        fuel_frame = self.SHAPE_MASKS['fuel_display']
        original_fuel_frame_width = jnp.round(fuel_frame.shape[1] / width_scale).astype(jnp.int32)
        original_fuel_frame_height = jnp.round(fuel_frame.shape[0] / height_scale).astype(jnp.int32)
        fuel_display_x = self.consts.SCREEN_WIDTH // 2 - original_fuel_frame_width // 2
        fuel_display_y = self.consts.SCREEN_HEIGHT - self.consts.UI_HEIGHT // 2 - original_fuel_frame_height // 2
        raster = self.jr.render_at(raster, fuel_display_x, fuel_display_y, fuel_frame, flip_offset=self.FLIP_OFFSETS['fuel_display'])

        # Fuel Indicator
        fuel_indicator_frame = self.SHAPE_MASKS['fuel_indicator']
        fuel_fill = state.player_fuel
        indicator_x = fuel_display_x + fuel_fill + 3
        raster = self.jr.render_at(raster, indicator_x, fuel_display_y + 4, fuel_indicator_frame, flip_offset=self.FLIP_OFFSETS['fuel_indicator'])

        # Score
        num_digits = jnp.maximum(1, jnp.ceil(jnp.log10(state.player_score + 1)).astype(jnp.int32))
        num_digits = jnp.minimum(num_digits, self.consts.MAX_SCORE_DIGITS) # Cap at max
        
        score_digits = self.jr.int_to_digits(state.player_score, max_digits=self.consts.MAX_SCORE_DIGITS)
        digit_masks = self.SHAPE_MASKS['score_digits']
        
        # Use render_label_selective like other games to avoid scalar predicate issues
        score_x = fuel_display_x
        score_y = fuel_display_y - 10
        start_index = self.consts.MAX_SCORE_DIGITS - num_digits
        raster = self.jr.render_label_selective(
            raster,
            score_x,
            score_y,
            score_digits,
            digit_masks,
            start_index,
            num_digits,
            spacing=12,
            max_digits_to_render=self.consts.MAX_SCORE_DIGITS
        )

        # Lives
        lives_frame = digit_masks[state.player_lives]
        lives_x = fuel_display_x - 8
        # Use original frame height for positioning
        lives_y = fuel_display_y + original_fuel_frame_height
        raster = self.jr.render_at(raster, lives_x, lives_y, lives_frame, flip_offset=self.FLIP_OFFSETS['score_digits'])

        # Activision Logo - calculate in game coordinates
        activision_sprite = self.SHAPE_MASKS['activision']
        original_activision_width = jnp.round(activision_sprite.shape[1] / width_scale).astype(jnp.int32)
        original_activision_height = jnp.round(activision_sprite.shape[0] / height_scale).astype(jnp.int32)
        activision_x = self.consts.SCREEN_WIDTH // 2 - original_activision_width // 2
        activision_y = self.consts.SCREEN_HEIGHT - (original_activision_height + 1)
        raster = self.jr.render_at(
            raster, activision_x, activision_y, 
            activision_sprite,
            flip_offset=self.FLIP_OFFSETS['activision']
        )
        
        # 9. Final Palette Lookup
        return self.jr.render_from_palette(raster, self.PALETTE)
