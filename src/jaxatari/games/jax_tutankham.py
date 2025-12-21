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
        {'name': 'bullet ', 'type': 'single', 'file': 'bullet_00.npy'},

        # Creatures (loaded as single sprites for manual padding)
        {'name': 'snake', 'type': 'single', 'file': 'creature_snake_00.npy'},
        # {'name': 'scorpion', 'type': 'single', 'file': 'scorpion.npy'},
        # {'name': 'bat', 'type': 'single', 'file': 'bat.npy'},
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
    WIDTH: int = 160
    HEIGHT: int = 210
    SPEED: int = 4
    PIXEL_COLOR: chex.Array = jnp.array([255, 255, 255], dtype=jnp.int32)  # white

    PLAYER_SIZE: chex.Array = jnp.array([5, 10], dtype=jnp.int32)

    PLAYER_LIVES: int = 3

    # Missile constants
    BULLET_SIZE: chex.Array = jnp.array([1, 2], dtype=jnp.int32)
    BULLET_SPEED: int = 8
    AMMO_SUPPLY: int = 900  # frames until ammo runs out

    MAX_LASER_FLASHES: int = 3
    LASER_FLASH_COOLDOWN: int = 60  # frames

    # Creature constants -------------------------------------
    CREATURE_SIZE: chex.Array = jnp.array([10, 10], dtype=jnp.int32)

    INACTIVE: int = 0
    ACTIVE: int = 1

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

    CREATURE_SPEED: chex.Array = jnp.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                           dtype=jnp.int32)  # speed for each creature type
    CREATURE_POINTS: chex.Array = jnp.array([1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 0, 3],
                                            dtype=jnp.int32)  # points for each creature type

    MAX_CREATURES: int = 3  # max number of creatures on screen at once

    # Item constants
    ITEM_SIZE: chex.Array = jnp.array([5, 5], dtype=jnp.int32)

    # Item Types
    KEY: int = 0
    CROWN_01: int = 1
    RING: int = 2
    RUBY: int = 3
    CHALICE: int = 4
    CROWN_02: int = 5

    # KEY_
    ITEM_POINTS: chex.Array = jnp.array([50, 100, 75, 150], dtype=jnp.int32)  # points for each item type

    RESPAWN_CHECKPOINT_UPDATE_INTERVAL: int = 180  # frames between respawn checkpoint updates

    # Asset config baked into constants
    ASSET_CONFIG: tuple = _get_default_asset_config()


# ---------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------
class TutankhamState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_lives: int
    tutankham_score: int  # current score

    checkpoint_x: int  # respawn checkpoint x
    checkpoint_y: int  # respawn checkpoint y

    bullet_state: chex.Array  # (, 4) array with (x, y, bullet_rotation, bullet_active)
    laser_flash_count: int  # number of laser flashes that can be fired
    laser_flash_cooldown: int  # cooldown timer for next laser flash
    amonition_timer: int  # if timer runs out, player can not fire again

    creature_states: chex.Array  # (3, 4) array with (x, y, creature_type, active) for each creature
    last_creature_spawn: int  # time since last creature spawn

    # item_states: chex.Array = None  # (N, 4) array with (x, y, item_type, collected) for each item (optional)

    respawn_step_counter: int  # counts the number of steps taken in the game

    has_key: bool  # whether the player has collected the key or not


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
        # pseudo for no errors
        player_mask = 0
        camera_offset = 0
        not_vanishing = True
        floor_checks = [1]
        ghost_frame = 0
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
        raster = jax.lax.cond(state.bullet_state[3],
                              self.jr.render_at(
                                  raster,
                                  state.bullet_state[0],
                                  state.bullet_state[1],
                                  # - camera_offset,
                                  self.SHAPE_MASKS["bullet"],
                                  flip_offset=ZERO_FLIP
                                  # self.FLIP_OFFSETS['player_group'],
                              ))
        # 7. Render Enemies
        creatures = jnp.stack([state.creature_states[0], state.creature_states[1]])
        creature_names = ["snake", "scorpion", "bat", "turtle", "jackel", "condor", "lion", "moth", "virus", "monkey",
                          "mystery", "weapon"]

        def render_creature(i, r):
            creature_pos = creatures[i]
            active = creatures[i][3]
            creature_type = creatures[i][2]

            # Use the single uniform offset for the group
            return jax.lax.cond(
                active,
                lambda r_in: self.jr.render_at_clipped(
                    r_in, creature_pos[0], creature_pos[1],
                    # - camera_offset,
                    self.SHAPE_MASKS[creature_names[creature_type]],
                    flip_offset=ZERO_FLIP  # self.ITEM_OFFSET  Use the single group offset
                ),
                lambda r_in: r_in,
                r
            )

        raster = render_creature(0, 1)
        raster = render_creature(1, 1)
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
        start_x = self.consts.WIDTH // 2
        start_y = self.consts.HEIGHT // 2
        checkpoint_x = start_x
        checkpoint_y = start_y
        tutankham_score = 0
        player_lives = self.consts.PLAYER_LIVES
        amonition_timer = self.consts.AMMO_SUPPLY
        bullet_state = jnp.array([0, 0, 0, 0], dtype=jnp.int32)  # TODO bullet state [3] is Flase still?
        creature_states = jnp.zeros((self.consts.MAX_CREATURES, 4))  # (x, y, creature_type, active)
        creature_states = jnp.array([[50, 50, 0, 0], [50, 100, 1, 0]], dtype=jnp.int32) # TODO delete
        last_creature_spawn = 0
        laser_flash_count = self.consts.MAX_LASER_FLASHES
        laser_flash_cooldown = self.consts.LASER_FLASH_COOLDOWN
        respawn_step_counter = 0
        has_key = False

        state = TutankhamState(player_x=start_x,
                               player_y=start_y,
                               checkpoint_x=checkpoint_x,
                               checkpoint_y=checkpoint_y,
                               tutankham_score=tutankham_score,
                               player_lives=player_lives,
                               bullet_state=bullet_state,
                               amonition_timer=amonition_timer,
                               creature_states=creature_states,
                               last_creature_spawn=last_creature_spawn,
                               laser_flash_count=laser_flash_count,
                               laser_flash_cooldown=laser_flash_cooldown,
                               respawn_step_counter=respawn_step_counter,
                               has_key=has_key
                               )
        return state, state  # TODO: (EnvObs, EnvState)

    # Player Step

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, player_x, player_y, action):
        speed = self.consts.SPEED

        dx = jnp.array([-speed, speed, 0, 0])
        dy = jnp.array([0, 0, -speed, speed])

        player_x = player_x + dx[action]
        player_y = player_y + dy[action]

        player_x = jnp.clip(player_x, 0, self.consts.WIDTH - 1)
        player_y = jnp.clip(player_y, 0, self.consts.HEIGHT - 1)

        return player_x, player_y

    # Bullet Step
    @partial(jax.jit, static_argnums=(0,))
    def bullet_step(self, bullet_state, player_x, player_y, amonition_timer, action):

        def get_rotation(action):
            if action == Action.RIGHTFIRE: return 1
            if action == Action.LEFTFIRE: return -1
            return 0  # default if firing up/down/etc

        space = (
                (action == Action.LEFTFIRE)
                or (action == Action.RIGHTFIRE)
        )

        new_bullet = bullet_state.copy()  # array with (x, y, bullet_rotation, bullet_active)

        # --- update existing bullets ---
        if bullet_state[3]:
            bullet_x = bullet_state[0] + self.consts.BULLET_SPEED * bullet_state[2]
            new_bullet[0] = bullet_x

            # Deactivate if out of bounds
            if not (0 <= bullet_x < self.consts.WIDTH):
                new_bullet = [0, 0, 0, False]

        # --- firing logic ---
        bullet_rdy = not bullet_state[3]

        if space and bullet_rdy and amonition_timer > 0:
            new_bullet = np.array([player_x, player_y, get_rotation(action), True])

        amonition_timer -= 1  # TODO: adjust amonition timer

        return new_bullet, amonition_timer

    @partial(jax.jit, static_argnums=(0,))
    def laser_flash_step(self, creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn, action):

        laser_flash_cooldown = max(laser_flash_cooldown - 1, 0)
        if action == Action.UPFIRE and laser_flash_count > 0 and laser_flash_cooldown == 0:
            new_laser_flash_count = laser_flash_count - 1  # use one laser flash
            new_laser_flash_cooldown = self.consts.LASER_FLASH_COOLDOWN  # reset cooldown
            last_creature_spawn = 0  # reset creature spawn timer on laser flash use

            new_creature_states = creature_states.copy()
            new_creature_states[:, -1] = 0  # set all creatures to inactive

            return new_creature_states, new_laser_flash_cooldown, new_laser_flash_count, last_creature_spawn

        return creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn

    # creature step
    @partial(jax.jit, static_argnums=(0,))
    def creature_step(self, creature_states, last_creature_spawn):

        def spawn_creature(creature_states, last_creature_spawn):
            # last_creature_spawn = vergangene Zeit (in Sekunden) seit letztem Frame

            # Parameter # TODO: REMOVE HARDCODED VALUES
            MAX_ACTIVE = 3
            GROWTH = 0.0003  # Chance steigt pro Sekunde um 5%
            MAX_PROB = 0.8  # Deckelung (optional)

            # 1) aktive Creatures zählen
            active_count = np.sum(creature_states[:, 3] == self.consts.ACTIVE)
            if active_count >= MAX_ACTIVE:
                return creature_states, last_creature_spawn  # nichts tun, Limit erreicht

            # 2) Spawn-Timer erhöhen
            last_creature_spawn += 1

            # 3) Spawn-Chance berechnen
            spawn_chance = last_creature_spawn * GROWTH
            spawn_chance = min(spawn_chance, MAX_PROB)

            # 4) treffen wir den Zufall?
            if np.random.random() > spawn_chance:
                return creature_states, last_creature_spawn  # nein → nichts machen

            # 5) Ja → wir spawnen einen!
            new_creature_states = creature_states.copy()

            for i in range(self.consts.MAX_CREATURES):
                x, y, creature_type, active = creature_states[i]
                if active == self.consts.INACTIVE:  # TODO: find correct spawner x,y
                    new_x = 0
                    new_y = np.random.randint(0, self.consts.HEIGHT)
                    new_creature_type = np.random.randint(0, len(self.consts.CREATURE_POINTS))
                    new_creature_states[i] = np.array([new_x, new_y, new_creature_type, self.consts.ACTIVE])

                    # Timer zurücksetzen: Start von vorne
                    last_creature_spawn = 0
                    break

            return new_creature_states, last_creature_spawn

        def move_creature(creature_state):
            x, y, creature_type, active = creature_state

            if active:
                speed = self.consts.CREATURE_SPEED[int(creature_type)]
                x += speed  # Move right for simplicity

                # Deactivate if out of bounds
                if x >= self.consts.WIDTH:
                    active = self.consts.INACTIVE

            return np.array([x, y, creature_type, active])

        creature_states, last_creature_spawn = spawn_creature(creature_states, last_creature_spawn)
        new_creature_states = creature_states.copy()

        for i in range(self.consts.MAX_CREATURES):
            new_creature_states[i] = move_creature(creature_states[i])

        return new_creature_states, last_creature_spawn

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
        """Check collision between two single entities"""
        # Calculate edges for rectangle 1
        rect1_left = x1
        rect1_right = x1 + size1[0]
        rect1_top = y1
        rect1_bottom = y1 + size1[1]

        # Calculate edges for rectangle 2
        rect2_left = x2
        rect2_right = x2 + size2[0]
        rect2_top = y2
        rect2_bottom = y2 + size2[1]

        # Check overlap
        horizontal_overlap = (
                rect1_left < rect2_right and
                rect1_right > rect2_left
        )

        vertical_overlap = (
                rect1_top < rect2_bottom and
                rect1_bottom > rect2_top
        )

        return horizontal_overlap and vertical_overlap

    @partial(jax.jit, static_argnums=(0,))
    def resolve_collisions(self, player_x, player_y, creature_states, bullet_state, player_lives):

        # check bullet/creature collision
        if bullet_state[3]:  # bullet is active

            for idx, creature in enumerate(creature_states):
                creature_x, creature_y, creature_type, active = creature

                if active == self.consts.ACTIVE:
                    collision = self.check_entity_collision(
                        bullet_state[0], bullet_state[1], self.consts.BULLET_SIZE,
                        creature_x, creature_y, self.consts.CREATURE_SIZE
                    )

                    if collision:
                        # Deactivate bullet and creature
                        bullet_state = np.array([0, 0, 0, False])
                        # TODO: kill creature logic (score etc)
                        creature_states[idx] = np.array([creature_x, creature_y, creature_type, self.consts.INACTIVE])
                        break  # Bullet can only hit one creature

        # check player/creature collision
        for idx, creature in enumerate(creature_states):
            creature_x, creature_y, creature_type, active = creature

            if active == self.consts.ACTIVE:
                collision = self.check_entity_collision(
                    player_x, player_y, self.consts.PLAYER_SIZE,
                    creature_x, creature_y, self.consts.CREATURE_SIZE
                )

                if collision:
                    player_lives -= 1
                    # TODO: respawn player logic
                    creature_states[idx] = np.array(
                        [creature_x, creature_y, creature_type, self.consts.INACTIVE])  # Deactivate creature
                    # Optionally, reset player position or apply other penalties
                    break  # Handle one collision at a time

        return bullet_state, creature_states, player_lives

    @partial(jax.jit, static_argnums=(0,))
    def respawn_player(self, player_x, player_y, checkpoint_x, checkpoint_y, prev_player_lives, current_player_lives,
                       creature_states, bullet_state, last_creature_spawn, respawn_step_counter):
        # TODO: Implement hardcoded checkpoints instead of last checkpoint position
        if current_player_lives < prev_player_lives:
            # Respawn player at checkpoint
            player_x = checkpoint_x
            player_y = checkpoint_y

            # Reset creature states
            creature_states = np.zeros((self.consts.MAX_CREATURES, 4))  # (x, y, creature_type, active)

            # Reset bullet state
            bullet_state = np.array([0, 0, 0, False])

            # reset spawn timer
            last_creature_spawn = 0

            # reset respawn step counter
            respawn_step_counter = 0

        elif respawn_step_counter == self.consts.RESPAWN_CHECKPOINT_UPDATE_INTERVAL:
            checkpoint_x = player_x
            checkpoint_y = player_y
            respawn_step_counter = 0
        else:
            respawn_step_counter += 1

        return player_x, player_y, checkpoint_x, checkpoint_y, creature_states, bullet_state, respawn_step_counter, last_creature_spawn

    # -----------------------------
    # Step logic (pure Python)
    # -----------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TutankhamState, action: int):

        player_x = state.player_x
        player_y = state.player_y
        checkpoint_x = state.checkpoint_x
        checkpoint_y = state.checkpoint_y
        tutankham_score = state.tutankham_score
        bullet_state = state.bullet_state
        creature_states = state.creature_states
        laser_flash_count = state.laser_flash_count
        last_creature_spawn = state.last_creature_spawn
        laser_flash_cooldown = state.laser_flash_cooldown
        amonition_timer = state.amonition_timer
        player_lives = state.player_lives
        respawn_step_counter = state.respawn_step_counter
        has_key = state.has_key

        player_x, player_y = self.player_step(player_x, player_y, action)

        "bullet_state, amonition_timer =self.bullet_step(bullet_state, player_x, player_y, amonition_timer, action)"

        "creature_states, last_creature_spawn = self.creature_step(creature_states, last_creature_spawn)"

        # laser flash step should go after creature step to immediately remove creatures
        "creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn = self.laser_flash_step(creature_states, laser_flash_cooldown, laser_flash_count, last_creature_spawn, action)"

        # temporary store previous lives and creature states for respawn & collision detection
        "prev_player_lives = player_lives"
        "prev_creature_states = creature_states.copy()"
        # TODO: add ITEM collisions (+key collection) -> has_key update
        "bullet_state, creature_states, player_lives = self.resolve_collisions(player_x, player_y, creature_states, bullet_state, player_lives)"

        # TODO:Update score based on creature deaths & items collected
        # score_update() bekommt prev_creature_states & creature_states und prev_item_states & item_states

        """player_x, player_y, checkpoint_x, checkpoint_y, creature_states, bullet_state, respawn_step_counter, last_creature_spawn = self.respawn_player(
            player_x, player_y,
            checkpoint_x, checkpoint_y,
            prev_player_lives, player_lives,
            creature_states,
            bullet_state,
            last_creature_spawn,
            respawn_step_counter
        )"""

        state = TutankhamState(player_x=player_x,
                               player_y=player_y,
                               checkpoint_x=checkpoint_x,
                               checkpoint_y=checkpoint_y,
                               tutankham_score=tutankham_score,
                               player_lives=player_lives,
                               bullet_state=bullet_state,
                               amonition_timer=amonition_timer,
                               creature_states=creature_states,
                               last_creature_spawn=last_creature_spawn,
                               laser_flash_count=laser_flash_count,
                               laser_flash_cooldown=laser_flash_cooldown,
                               respawn_step_counter=respawn_step_counter,
                               has_key=has_key
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
