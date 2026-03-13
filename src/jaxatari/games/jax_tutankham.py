from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Discrete, Box
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
import chex
import jax.lax
import os
from functools import partial
from PIL import Image
import numpy as np
import jaxatari.spaces as spaces
import time


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for BankHeist.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    # Define file lists for groups
    # tombs = [f"tomb_{i + 1}.npy" for i in range(4)]

    # Define the sprites
    config = (
        # Backgrounds (loaded as a group)
        # Note: The 'background' type is not used here, as the city map is the primary background.
        # We will treat 'tombs' as our base background sprites.
        # {'name': 'tombs', 'type': 'group', 'files': tombs},

        # Roomparts
        {'name': 'room_floor', 'type': 'single', 'file': 'room_floor.npy'},

        # Player (loaded as single sprites for manual padding)
        # {'name': 'archeologist ', 'type': 'single', 'file': 'archeologist.npy'},
        {'name': 'player', 'type': 'single', 'file': 'player_idle.npy'},
        {'name': 'player_move_00 ', 'type': 'single', 'file': 'player_move_00.npy'},
        {'name': 'player_move_01 ', 'type': 'single', 'file': 'player_move_01.npy'},
        {'name': 'player_death ', 'type': 'single', 'file': 'player_death.npy'},
        {'name': 'bullet', 'type': 'single', 'file': 'bullet_00.npy'},

        # Creatures (loaded as single sprites for manual padding)
        {'name': 'snake', 'type': 'single', 'file': 'creature_snake_00.npy'},
        {'name': 'scorpion', 'type': 'single', 'file': 'creature_scorpion_00.npy'},
        {'name': 'bat', 'type': 'single', 'file': 'creature_bat_00.npy'},
        # {'name': 'turtle', 'type': 'single', 'file': 'turtle.npy'},
        # {'name': 'jackel', 'type': 'single', 'file': 'jackel.npy'},
        # {'name': 'condor', 'type': 'single', 'file': 'condor.npy'},
        # {'name': 'lion', 'type': 'single', 'file': 'lion.npy'},
        # {'name': 'moth', 'type': 'single', 'file': 'moth.npy'},
        # {'name': 'virus', 'type': 'single', 'file': 'virus.npy'},
        # {'name': 'monkey', 'type': 'single', 'file': 'monkey.npy'},
        # {'name': 'mystery', 'type': 'single', 'file': 'mystery.npy'},
        # {'name': 'weapon', 'type': 'single', 'file': 'weapon.npy'},

        # Treasures
        # {'name': 'key', 'type': 'single', 'file': 'key.npy'},
        # {'name': 'crown', 'type': 'single', 'file': 'crown.npy'},
        # {'name': 'ring', 'type': 'single', 'file': 'ring.npy'},
        # {'name': 'ruby', 'type': 'single', 'file': 'ruby.npy'},
        # {'name': 'chalice', 'type': 'single', 'file': 'chalice.npy'},
        # {'name': 'emerald', 'type': 'single', 'file': 'emerald.npy'},
        # {'name': 'goblet', 'type': 'single', 'file': 'goblet.npy'},
        # {'name': 'bust', 'type': 'single', 'file': 'bust.npy'},
        # {'name': 'trident', 'type': 'single', 'file': 'trident.npy'},
        # {'name': 'herb', 'type': 'single', 'file': 'herb.npy'},
        # {'name': 'diamond', 'type': 'single', 'file': 'diamond.npy'},
        # {'name': 'candelabra', 'type': 'single', 'file': 'candelabra.npy'},
        # {'name': 'amulet', 'type': 'single', 'file': 'amulet.npy'},
        # {'name': 'fan', 'type': 'single', 'file': 'fan.npy'},
        # {'name': 'crystal', 'type': 'single', 'file': 'crystal.npy'},
        # {'name': 'zircon', 'type': 'single', 'file': 'zircon.npy'},
        # {'name': 'dagger', 'type': 'single', 'file': 'dagger.npy'},

        # UI
        # {'name': 'lives', 'type': 'single', 'pattern': 'lives.npy'},
        # {'name': 'flashbangs', 'type': 'single', 'pattern': 'flashbangs.npy'},
        # {'name': 'points', 'type': 'digits', 'pattern': 'lives.npy'},
        # {'name': 'time', 'type': 'single', 'pattern': 'time.npy'},
        {'name': 'background', 'type': 'background', 'file': 'background_full.npy'},
    )
    return config


class Entity(NamedTuple):
    position: jnp.ndarray
    direction: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
class TutankhamConstants(NamedTuple):
    # Game Window
    WIDTH: int = 160
    HEIGHT: int = 210
    
    # Player constants
    PLAYER_SPEED: float = 2.5
    PLAYER_SIZE: chex.Array = jnp.array([5, 10], dtype=jnp.int32)
    PLAYER_LIVES: int = 3

    # Missile constants
    BULLET_SIZE: chex.Array = jnp.array([1, 2], dtype=jnp.int32)
    BULLET_SPEED: int = 8
    AMMO_SUPPLY: int = 900  # frames until ammo runs out

    MAX_LASER_FLASHES: int = 3
    LASER_FLASH_COOLDOWN: int = 60  # frames

    # Creature constants -------------------------------------

    # Creature Types
    SNAKE: int = 0
    SCORPION: int = 1
    BAT: int = 2
    TURTLE: int = 3
    JACKEL: int = 4
    CONDOR: int = 5
    LION: int = 6
    MOTH: int = 7
    VIRUS: int = 8
    MONKEY: int = 9
    MYSTERY: int = 10
    WEAPON: int = 11

    CREATURE_SIZE: chex.Array = jnp.array([10, 10], dtype=jnp.int32)

    INACTIVE: int = 0
    ACTIVE: int = 1

    CREATURE_SPEED: chex.Array = jnp.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                           dtype=jnp.int32)  # speed for each creature type
    CREATURE_POINTS: chex.Array = jnp.array([1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 0, 3],
                                            dtype=jnp.int32)  # points for defeating each creature type

    MAX_CREATURES: int = 2  # max number of creatures on screen at once

    # Item constants ----------------------------------------------------

    # Item Types
    KEY: int = 0
    CROWN_01: int = 1
    RING: int = 2
    RUBY: int = 3
    CHALICE: int = 4
    CROWN_02: int = 5

    ITEM_SIZE: chex.Array = jnp.array([5, 5], dtype=jnp.int32)

    ITEM_POINTS: chex.Array = jnp.array([50, 100, 75, 150], dtype=jnp.int32)  # points for collecting each item type


    # Asset config baked into constants
    ASSET_CONFIG: tuple = _get_default_asset_config()


    # Level -----------------------------------------

    # Define which creatures can spawn in each level 
    LEVEL_CREATURES: chex.Array = jnp.array([
        [SNAKE, BAT, BAT],
        [TURTLE, JACKEL, CONDOR],
        [LION, MOTH, VIRUS],
        [MONKEY, MYSTERY, WEAPON]
    ])
    


# ---------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------
class TutankhamState(NamedTuple):
    # key state for creature spawn randomness
    rng_key: int

    level: int  # current level (up to 16 Levels)

    player_x: chex.Array
    player_y: chex.Array
    player_lives: int
    tutankham_score: int  # current score


    bullet_state: chex.Array  # (, 4) array with (x, y, bullet_rotation, bullet_active)
    laser_flash_count: int  # number of laser flashes that can be fired
    laser_flash_cooldown: int  # cooldown timer for next laser flash
    amonition_timer: int  # if timer runs out, player can not fire again

    creature_states: chex.Array  # (3, 4) array with (x, y, creature_type, active) for each creature
    last_creature_spawn: int  # time since last creature spawn

    item_states: chex.Array # (N, 4) array with (x, y, item_type, collected) for each item

    has_key: bool  # whether the player has collected the key or not

    last_directional_action: int  # last action that had a directional component


# ---------------------------------------------------------------------
# Renderer (No JAX)
# ---------------------------------------------------------------------

class TutankhamRenderer(JAXGameRenderer):
    def __init__(self, consts: TutankhamConstants = None):
        super().__init__()
        self.consts = consts or TutankhamConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tutankham"

        # 1. Configure the rendering utility
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 3. Make one call to load and process all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

    # ---------------------------------------------------------
    # Main render() method
    # ---------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TutankhamState):
        ZERO_FLIP = jnp.array([0, 0], dtype=jnp.int32)
        indices_to_update = 0
        new_color_ids = 0

        # 1. Start with the static blue background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self.jr.render_at(
            raster,
            0,  # x
            0,  # y
            self.SHAPE_MASKS["room_floor"],
            flip_offset=ZERO_FLIP
        )

        # raster = jax.lax.cond(
        #    floor_checks[0] & not_vanishing,
        #    lambda r: self.jr.render_at_clipped(
        #        r, state.ghost[0], state.ghost[1] - camera_offset,
        #        self.SHAPE_MASKS['ghost_group'][ghost_frame],
        #        flip_offset=self.FLIP_OFFSETS['ghost_group']
        #    ),
        #    lambda r: r,
        #    raster
        # )
        # 2. Render Player
        raster = self.jr.render_at(
            raster,
            state.player_x,
            state.player_y,
            # - camera_offset,
            self.SHAPE_MASKS["player"],
            flip_offset=ZERO_FLIP
            # self.FLIP_OFFSETS['player_group'],
        )
        #creatures
        raster = self.jr.render_at(
            raster,
            state.creature_states[0][0],
            state.creature_states[0][1],
            # - camera_offset,
            self.SHAPE_MASKS["snake"],
            flip_offset=ZERO_FLIP
            # self.FLIP_OFFSETS['player_group'],
        )
        raster = self.jr.render_at(
            raster,
            state.creature_states[1][0],
            state.creature_states[1][1],
            # - camera_offset,
            self.SHAPE_MASKS["bat"],
            flip_offset=ZERO_FLIP
            # self.FLIP_OFFSETS['player_group'],
        )
        # player_frame = jnp.where(state.stun_duration > 0, state.stun_duration % 8 + 1, state.player_direction[1])
        # player_mask = self.SHAPE_MASKS['player_group'][player_frame]
        # raster = self.jr.render_at(
        #    raster, state.player[0], state.player[1] - camera_offset,
        #    player_mask, flip_offset=self.FLIP_OFFSETS['player_group']
        # )
        # 2.5 Animations
        # 3. Render Walls
        # 4. Render Teleporter and Spawner
        # 5. Render Treasures
        # 6. Render Bullets
        raster = jax.lax.cond(state.bullet_state[3] == 1,
                              lambda r: self.jr.render_at(
                                  r,
                                  state.bullet_state[0],
                                  state.bullet_state[1],
                                  # - camera_offset,
                                  self.SHAPE_MASKS["bullet"],
                                  flip_offset=ZERO_FLIP
                                  # self.FLIP_OFFSETS['player_group'],
                              ),
                              lambda r: r,
                              raster)
        """# 7. Render Enemies
        creatures = jnp.stack([state.creature_states[0], state.creature_states[1]])
        snake_mask = self.SHAPE_MASKS["snake"]
        scorpion_mask = self.SHAPE_MASKS["scorpion"]
        bat_mask = self.SHAPE_MASKS["bat"]
        all_masks = (snake_mask, scorpion_mask, bat_mask)

        def render_creature(i, r):
            creature_pos = creatures[i]
            active = creatures[i][3]
            creature_type = creatures[i][2]
            mask = jax.lax.switch(
                creature_type,
                [
                    lambda: all_masks[0],
                    lambda: all_masks[1],
                    lambda: all_masks[2]
                ]
            )

            # Use the single uniform offset for the group
            return jax.lax.cond(
                active == 1,
                lambda r_in: self.jr.render_at_clipped(
                    r_in,
                    creature_pos[0],
                    creature_pos[1],
                    # - camera_offset,
                    mask,
                    flip_offset=ZERO_FLIP  # self.ITEM_OFFSET  Use the single group offset
                ),
                lambda r_in: r_in,
                r
            )

        raster = render_creature(0, raster)
        raster = render_creature(1, raster)"""


        # 8. Render UI
        # 9. Final Palette Lookup
        return self.jr.render_from_palette(
            raster,
            self.PALETTE,
            indices_to_update=indices_to_update,
            new_color_ids=new_color_ids
        )


# ---------------------------------------------------------------------
# Environment (No JAX)
# ---------------------------------------------------------------------
class JaxTutankham(JaxEnvironment):
    def __init__(self):
        consts = TutankhamConstants()
        super().__init__(consts)
        self.renderer = TutankhamRenderer()
        self.consts = consts

        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE
        ]

    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, state: TutankhamState) -> TutankhamState:
        return state._replace()

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, key=None):
        # Generate random seed based on current time for creature spawn randomness
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        level = 0
        start_x = self.consts.WIDTH // 2
        start_y = self.consts.HEIGHT // 2
        tutankham_score = 0
        player_lives = self.consts.PLAYER_LIVES
        amonition_timer = self.consts.AMMO_SUPPLY
        bullet_state = jnp.array([0, 0, 0, 0], dtype=jnp.int32) # TODO bullet state [3] is Flase still?
        creature_states = jnp.zeros((self.consts.MAX_CREATURES, 4))  # (x, y, creature_type, active)
        item_states = jnp.zeros((4, 4))  # (x, y, item_type, collected) for each item # TODO

        last_creature_spawn = 0
        laser_flash_count = self.consts.MAX_LASER_FLASHES
        laser_flash_cooldown = self.consts.LASER_FLASH_COOLDOWN
        has_key = False


        state = TutankhamState(level=level,
                               player_x=start_x,
                                player_y=start_y,
                                tutankham_score=tutankham_score,
                                player_lives=player_lives,
                                bullet_state=bullet_state,
                                amonition_timer=amonition_timer,
                                creature_states=creature_states,
                                item_states=item_states,
                                last_creature_spawn=last_creature_spawn,
                                laser_flash_count=laser_flash_count,
                                laser_flash_cooldown=laser_flash_cooldown,
                                has_key=has_key,
                                last_directional_action=0,
                                rng_key=key
                               )
        return state, state #TODO: (EnvObs, EnvState)
    
    
    # Player Step
    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, player_x, player_y, action, last_directional_action):
        speed = self.consts.PLAYER_SPEED

        dx = jnp.array([
        0,          # 0  NOOP
        0,          # 1  FIRE
        0,          # 2  UP
        speed,      # 3  RIGHT
        -speed,     # 4  LEFT
        0,          # 5  DOWN
        0,          # 6  UPRIGHT
        0,          # 7  UPLEFT
        0,          # 8  DOWNRIGHT
        0,          # 9  DOWNLEFT
        0,          # 10 UPFIRE
        0,          # 11 RIGHTFIRE
        0,          # 12 LEFTFIRE
        0,          # 13 DOWNFIRE
        0,          # 14 UPRIGHTFIRE
        0,          # 15 UPLEFTFIRE
        0,          # 16 DOWNRIGHTFIRE
        0,          # 17 DOWNLEFTFIRE
        ])

        dy = jnp.array([
            0,          # 0  NOOP
            0,          # 1  FIRE
            -speed,     # 2  UP
            0,          # 3  RIGHT
            0,          # 4  LEFT
            speed,      # 5  DOWN
            0,     # 6  UPRIGHT
            0,     # 7  UPLEFT
            0,      # 8  DOWNRIGHT
            0,      # 9  DOWNLEFT
            0,     # 10 UPFIRE
            0,          # 11 RIGHTFIRE
            0,          # 12 LEFTFIRE
            0,      # 13 DOWNFIRE
            0,     # 14 UPRIGHTFIRE
            0,     # 15 UPLEFTFIRE
            0,      # 16 DOWNRIGHTFIRE
            0,      # 17 DOWNLEFTFIRE
        ])

        # If the current action has no directional component, fall back to the last directional action
        has_movement = (dx[action] != 0) | (dy[action] != 0)
        effective_action = jnp.where(has_movement, action, last_directional_action)
        new_last_directional_action = jnp.where(has_movement, action, last_directional_action)

        player_x = player_x + dx[effective_action]
        player_y = player_y + dy[effective_action]

        player_x = jnp.clip(player_x, 0, self.consts.WIDTH - 1)
        player_y = jnp.clip(player_y, 0, self.consts.HEIGHT - 1)

        return player_x, player_y, new_last_directional_action
    

    #Bullet Step
    @partial(jax.jit, static_argnums=(0,))
    def bullet_step(self, bullet_state, player_x, player_y, amonition_timer, action):
        
        
        def get_rotation(action):
            return jax.lax.cond(
                action == Action.RIGHTFIRE,
                lambda _: 1, # bullet travels right
                lambda _: jax.lax.cond(
                    action == Action.LEFTFIRE,
                    lambda _: -1, # bullet travels left
                    lambda _: 0, # default if firing up/down/etc
                    operand=None
                ),
                operand=None
            )


        space = jnp.logical_or(action == Action.LEFTFIRE, action == Action.RIGHTFIRE)

        new_bullet = bullet_state #array with (x, y, bullet_rotation, bullet_active)


        # --- update bullet x position if active (bullet only travels horizontal so no vertical movement) ---        
        bullet_x = jax.lax.cond(
            bullet_state[3] == 1,  # if bullet is active
            lambda x: x + self.consts.BULLET_SPEED * bullet_state[2],
            lambda x: x,
            bullet_state[0],
        )
        new_bullet = new_bullet.at[0].set(bullet_x)
        
        # Deactivate if out of bounds
        # TODO: Wall collision detection
        new_bullet = jax.lax.cond(
            (bullet_x < 0) | (bullet_x >= self.consts.WIDTH),
            lambda _: jnp.zeros((4,), dtype=new_bullet.dtype),
            lambda bullet: bullet,
            operand=new_bullet
            )



        # --- firing logic ---
        bullet_rdy = 1 - bullet_state[3]  # 1 if bullet inactive, 0 if active

        new_bullet = jax.lax.cond(
            (space & bullet_rdy & (amonition_timer > 0)) == 1, # if firing action & bullet is inactive & amonition available
            lambda _: jnp.array([player_x, player_y, get_rotation(action), 1], dtype=jnp.int32), # shoot bullet at player position,
            lambda bullet: bullet, # don't shoot bullet
            operand=new_bullet
        )

        amonition_timer -= 1 # TODO: adjust amonition timer

        return new_bullet, amonition_timer

    @partial(jax.jit, static_argnums=(0,))
    def laser_flash_step(self, creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn, action):

        # decrease cooldown timer in every step
        laser_flash_cooldown = jnp.maximum(laser_flash_cooldown - 1, 0)

        # check if laser flash is being fired this step
        is_firing = (action == Action.UPFIRE) & (laser_flash_count > 0) & (laser_flash_cooldown == 0)

        # if firing, set all creatures to inactive, decrease flash count by 1, reset cooldown and reset creature spawn timer
        new_laser_flash_count = jnp.where(is_firing, laser_flash_count - 1, laser_flash_count)
        new_laser_flash_cooldown = jnp.where(is_firing, self.consts.LASER_FLASH_COOLDOWN, laser_flash_cooldown)
        new_last_creature_spawn = jnp.where(is_firing, 0, last_creature_spawn)

        # active mask contains 1 for active creatures and 0 for inactive creatures, if firing set all to 0
        active_mask = creature_states[:, -1]
        # if firing, set all active_mask values to 0, otherwise keep them as they are
        new_active_mask = jnp.where(is_firing, 0, active_mask)

        new_creature_states = creature_states.at[:, -1].set(new_active_mask)

        return new_creature_states, new_laser_flash_cooldown, new_laser_flash_count, new_last_creature_spawn


    # creature step
    @partial(jax.jit, static_argnums=(0,))
    def creature_step(self, creature_states):

        # Update creature position if active
        def move_creature(creature_state):
            x, y, creature_type, active = creature_state

            speed = self.consts.CREATURE_SPEED[creature_type.astype(int)]

            x_new = x + speed * active # TODO: If creature active, Move right for simplicity

            # Deactivate if out of bounds
            active_new = jnp.where(
                x_new >= self.consts.WIDTH,
                self.consts.INACTIVE,
                active
            )

            # If inactive, reset position to (0, 0)
            x_new = x_new * active_new
            y_new = y * active_new

            return jnp.array([x_new, y_new, creature_type, active_new], dtype=jnp.int32)


        # iterate over creatures and move them
        move_creature_vmapped = jax.vmap(move_creature)
        new_creature_states = move_creature_vmapped(creature_states)

        return new_creature_states

    # creature spawner step
    @partial(jax.jit, static_argnums=(0,))
    def spawner_step(self, creature_states, last_creature_spawn, level, rng_key):
        GROWTH = 0.0003
        MAX_PROB = 0.8

        # Increment timer every step
        new_last_creature_spawn = last_creature_spawn + 1

        # Spawn chance grows linearly with time, capped at MAX_PROB
        spawn_chance = jnp.clip(new_last_creature_spawn * GROWTH, 0.0, MAX_PROB)

        # Split key: one for the probability roll, one for the creature type
        rng_key, key_roll, key_type = jax.random.split(rng_key, 3)
        roll = jax.random.uniform(key_roll)
        should_spawn = roll < spawn_chance

        # Only spawn if there is a free slot
        inactive_mask = creature_states[:, 3] == self.consts.INACTIVE
        has_free_slot = jnp.any(inactive_mask)
        first_free_slot = jnp.argmax(inactive_mask)  # index of first inactive creature

        # Random creature type
        new_type = jax.random.choice(key_type, self.consts.LEVEL_CREATURES[level])  # creature type depends on current level
        new_creature = jnp.array([0, 60, new_type, self.consts.ACTIVE], dtype=jnp.int32)

        do_spawn = should_spawn & has_free_slot

        # if do spawn is true, insert new creature at first free slot, otherwise keep creature states unchanged
        new_row = jnp.where(do_spawn, new_creature, creature_states[first_free_slot])
        new_creature_states = creature_states.at[first_free_slot].set(new_row)

        # Reset timer on spawn
        final_last_creature_spawn = jnp.where(do_spawn, jnp.int32(0), new_last_creature_spawn)

        return new_creature_states, final_last_creature_spawn, rng_key



    # score update based on creature deaths & item collections
    @partial(jax.jit, static_argnums=(0,))
    def update_score(self, score, prev_creature_states, new_creature_states, prev_item_states, new_item_states,
                     laser_flash_cooldown):

        def detect_creature_deaths(prev_creature_states, new_creature_states):
            """Erkennt Todes-Events: active: 1 → 0."""
            old_active = prev_creature_states[:, 3] == 1.0
            new_active = new_creature_states[:, 3] == 1.0
            died = jnp.logical_and(old_active, jnp.logical_not(new_active))  # TODO: translate to pure python
            return died  # shape: (MAX_CREATURES,)

        new_score = score

        # if TRUE then laser flash was used this frame and all creature deaths will not be counted
        # only bullet kills count towards score
        if not laser_flash_cooldown == self.consts.LASER_FLASH_COOLDOWN:

            # detect creature deaths
            deaths = detect_creature_deaths(prev_creature_states, new_creature_states)

            # update score based on creature deaths
            for i in range(deaths.shape[0]):
                if deaths[i]:
                    creature_type = int(prev_creature_states[i, 2])
                    points = int(self.consts.CREATURE_POINTS[creature_type])
                    new_score += points

        # TODO: item collection logic

        return new_score

    @partial(jax.jit, static_argnums=(0,))
    def check_entity_collision(self, x1, y1, size1, x2, y2, size2):
        '''Check AABB collision between two entities.'''
        return (
            (x1 < x2 + size2[0]) & (x1 + size1[0] > x2) &
            (y1 < y2 + size2[1]) & (y1 + size1[1] > y2)
        )


    @partial(jax.jit, static_argnums=(0,))
    def respawn_player(self, player_x, player_y, player_lives):
        '''
        Resets player position to last checkpoint and decreases lives by 1
        Sets bullet state and creature states to default values
        Resets creature spawn timer to 0
        '''
        
        # TODO: Replace with hardcoded per-room checkpoints
        respawn_x = jnp.int32(30)
        respawn_y = jnp.int32(30)

        creature_states = jnp.zeros((self.consts.MAX_CREATURES, 4), dtype=jnp.int32)
        bullet_state = jnp.zeros(4, dtype=jnp.int32)
        last_creature_spawn = jnp.int32(0)

        return respawn_x, respawn_y, bullet_state, creature_states, player_lives - 1, last_creature_spawn



    @partial(jax.jit, static_argnums=(0,))
    def resolve_bullet_collisions(self, creature_states, bullet_state):
        active_mask = creature_states[:, 3] == self.consts.ACTIVE

        # check bullet-creature collisions (vectorized over all creatures)
        def bullet_hits_creature(creature):
            return self.check_entity_collision(
                bullet_state[0], bullet_state[1], self.consts.BULLET_SIZE,
                creature[0], creature[1], self.consts.CREATURE_SIZE,
            )

        bullet_hits = jax.vmap(bullet_hits_creature)(creature_states)
        bullet_hits = bullet_hits & active_mask & (bullet_state[3] == 1)

        any_bullet_hit = jnp.any(bullet_hits)
        # Deactivate only the first hit creature (cumsum trick: first True stays 1)
        first_bullet_hit = (jnp.cumsum(bullet_hits) == 1) & bullet_hits

        new_creature_active = jnp.where(first_bullet_hit, self.consts.INACTIVE, creature_states[:, 3])
        new_creature_states = creature_states.at[:, 3].set(new_creature_active)
        new_bullet_state = jnp.where(any_bullet_hit, jnp.zeros(4, dtype=bullet_state.dtype), bullet_state)

        return new_creature_states, new_bullet_state


    @partial(jax.jit, static_argnums=(0,))
    def resolve_player_creature_collisions(self, player_x, player_y, creature_states, bullet_state, player_lives, last_creature_spawn):

        # check player-creature collisions (vectorized over all creatures)
        def player_hits_creature(creature):
            return self.check_entity_collision(
                player_x, player_y, self.consts.PLAYER_SIZE,
                creature[0], creature[1], self.consts.CREATURE_SIZE,
            )

        player_hits = jax.vmap(player_hits_creature)(creature_states)
        player_hits = player_hits & (creature_states[:, 3] == self.consts.ACTIVE)

        player_hit = jnp.any(player_hits) # is true if player collides with any active creature

        # Compute respawn state unconditionally, then select with jnp.where
        (respawn_x, respawn_y,
         respawn_bullet, respawn_creatures,
         respawn_lives, respawn_spawn) = self.respawn_player(player_x, player_y, player_lives)

        final_player_x = jnp.where(player_hit, respawn_x, player_x)
        final_player_y = jnp.where(player_hit, respawn_y, player_y)
        final_bullet_state = jnp.where(player_hit, respawn_bullet, bullet_state)
        final_creature_states = jnp.where(player_hit, respawn_creatures, creature_states)
        final_player_lives = jnp.where(player_hit, respawn_lives, player_lives)
        final_last_creature_spawn = jnp.where(player_hit, respawn_spawn, last_creature_spawn)

        return (final_player_x, final_player_y,
                final_bullet_state, final_creature_states,
                final_player_lives, final_last_creature_spawn)

    # @partial(jax.jit, static_argnums=(0,))
    # def resolve_player_item_collisions(self, player_x, player_y, item_states):

    #     # check player-creature collisions (vectorized over all creatures)
    #     def player_hits_creature(creature):
    #         return self.check_entity_collision(
    #             player_x, player_y, self.consts.PLAYER_SIZE,
    #             creature[0], creature[1], self.consts.CREATURE_SIZE,
    #         )

    #     player_hits = jax.vmap(player_hits_creature)(creature_states)
    #     player_hits = player_hits & (creature_states[:, 3] == self.consts.ACTIVE)

    #     player_hit = jnp.any(player_hits) # is true if player collides with any active creature

    #     # Compute respawn state unconditionally, then select with jnp.where
    #     (respawn_x, respawn_y,
    #      respawn_bullet, respawn_creatures,
    #      respawn_lives, respawn_spawn) = self.respawn_player(player_x, player_y, player_lives)

    #     final_player_x = jnp.where(player_hit, respawn_x, player_x)
    #     final_player_y = jnp.where(player_hit, respawn_y, player_y)
    #     final_bullet_state = jnp.where(player_hit, respawn_bullet, bullet_state)
    #     final_creature_states = jnp.where(player_hit, respawn_creatures, creature_states)
    #     final_player_lives = jnp.where(player_hit, respawn_lives, player_lives)
    #     final_last_creature_spawn = jnp.where(player_hit, respawn_spawn, last_creature_spawn)

    #     return (final_player_x, final_player_y,
    #             final_bullet_state, final_creature_states,
    #             final_player_lives, final_last_creature_spawn)



    # Step logic
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TutankhamState, action: int):
        level=state.level
        player_x = state.player_x
        player_y = state.player_y
        tutankham_score = state.tutankham_score
        bullet_state = state.bullet_state
        creature_states = state.creature_states
        item_states = state.item_states
        laser_flash_count = state.laser_flash_count
        last_creature_spawn = state.last_creature_spawn
        laser_flash_cooldown = state.laser_flash_cooldown
        amonition_timer = state.amonition_timer
        player_lives = state.player_lives
        has_key = state.has_key
        last_directional_action = state.last_directional_action
        rng_key = state.rng_key

        player_x, player_y, last_directional_action = self.player_step(player_x, player_y, action, last_directional_action)

        bullet_state, amonition_timer =self.bullet_step(bullet_state, player_x, player_y, amonition_timer, action)

        creature_states = self.creature_step(creature_states)

        creature_states, last_creature_spawn, rng_key = self.spawner_step(creature_states, last_creature_spawn, level, rng_key)

        # laser flash step should go after creature step to immediately remove creatures
        creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn = self.laser_flash_step(creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn, action)

        # temporary store previous lives and creature states for respawn & collision detection
        prev_creature_states = creature_states.copy()
        # TODO: add ITEM collisions (+key collection) -> has_key update

        creature_states, bullet_state = self.resolve_bullet_collisions(creature_states, bullet_state)

        (player_x, player_y, 
         bullet_state, creature_states, 
         player_lives, last_creature_spawn
         ) = self.resolve_player_creature_collisions(
             player_x, player_y, 
             creature_states, bullet_state, 
             player_lives, last_creature_spawn
             )
  

        # TODO:Update score based on creature deaths & items collected
        # score_update() bekommt prev_creature_states & creature_states und prev_item_states & item_states



        state = TutankhamState(level=level,
                               player_x=player_x,
                               player_y=player_y,
                               tutankham_score=tutankham_score,
                               player_lives=player_lives,
                               bullet_state=bullet_state,
                               amonition_timer=amonition_timer,
                               creature_states=creature_states,
                               item_states=item_states,
                               last_creature_spawn=last_creature_spawn,
                               laser_flash_count=laser_flash_count,
                               laser_flash_cooldown=laser_flash_cooldown,
                               has_key=has_key,
                               last_directional_action=last_directional_action,
                               rng_key=rng_key
                               )

        reward = 0.0
        done = self._get_done(state)
        info = 0

        # return observation, new_state, env_reward, done, info
        return state, state, reward, done, info

        # @partial(jax.jit, static_argnums=(0,))
        # def check_wall_collision(self, pos, size):
        # """Check collision between an entity and the wall"""####

        # Because the wall sprite is not at (0,0)
        # pos = jnp.array([pos[0], pos[1] - self.consts.WALL_Y_OFFSET])##

        # collision_top_left = self.consts.WALL[pos[1]][pos[0]]
        # collision_top_right = self.consts.WALL[pos[1]][pos[0] + size[0] - 1]
        # collision_bottom_left = self.consts.WALL[pos[1] + size[1] - 1][pos[0]]
        # collision_bottom_right = self.consts.WALL[pos[1] + size[1] - 1][pos[0] + size[0] - 1]

        # return jnp.any(
        #    jnp.array([collision_top_left, collision_top_right, collision_bottom_right, collision_bottom_left]))
        # return False

    # -----------------------------
    # Rendering
    # -----------------------------
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TutankhamState) -> jnp.ndarray:
        return self.renderer.render(state)

    # -----------------------------
    # Action & Observation Space
    # -----------------------------
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self):
        return Box(
            low=0,
            high=max(self.consts.WIDTH, self.consts.HEIGHT),
            shape=(2,),
            dtype=np.int32,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TutankhamState) -> bool:
        game_over = state.player_lives <= 0
        beat_game = False  # TODO: replace game winning condition later
        return jnp.logical_or(game_over, beat_game)
