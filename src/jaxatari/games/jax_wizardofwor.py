from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.lax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        # Characters: each 7 frames (4 walk + 3 death) as 'group'
        {'name': 'player', 'type': 'group', 'files': [
            'player/player_0.npy', 'player/player_1.npy',
            'player/player_2.npy', 'player/player_3.npy',
            'player/death_0.npy', 'player/death_1.npy', 'player/death_2.npy',
        ]},
        {'name': 'burwor', 'type': 'group', 'files': [
            'enemies/burwor/burwor_0.npy', 'enemies/burwor/burwor_1.npy',
            'enemies/burwor/burwor_2.npy', 'enemies/burwor/burwor_3.npy',
            'enemies/burwor/death_0.npy', 'enemies/burwor/death_1.npy',
            'enemies/burwor/death_2.npy',
        ]},
        {'name': 'garwor', 'type': 'group', 'files': [
            'enemies/garwor/garwor_0.npy', 'enemies/garwor/garwor_1.npy',
            'enemies/garwor/garwor_2.npy', 'enemies/garwor/garwor_3.npy',
            'enemies/garwor/death_0.npy', 'enemies/garwor/death_1.npy',
            'enemies/garwor/death_2.npy',
        ]},
        {'name': 'thorwor', 'type': 'group', 'files': [
            'enemies/thorwor/thorwor_0.npy', 'enemies/thorwor/thorwor_1.npy',
            'enemies/thorwor/thorwor_2.npy', 'enemies/thorwor/thorwor_3.npy',
            'enemies/thorwor/death_0.npy', 'enemies/thorwor/death_1.npy',
            'enemies/thorwor/death_2.npy',
        ]},
        {'name': 'worluk', 'type': 'group', 'files': [
            'enemies/worluk/worluk_0.npy', 'enemies/worluk/worluk_1.npy',
            'enemies/worluk/worluk_2.npy', 'enemies/worluk/worluk_3.npy',
            'enemies/worluk/death_0.npy', 'enemies/worluk/death_1.npy',
            'enemies/worluk/death_2.npy',
        ]},
        {'name': 'wizard', 'type': 'group', 'files': [
            'enemies/wizard/wizard_0.npy', 'enemies/wizard/wizard_1.npy',
            'enemies/wizard/wizard_2.npy', 'enemies/wizard/wizard_3.npy',
            'enemies/wizard/death_0.npy', 'enemies/wizard/death_1.npy',
            'enemies/wizard/death_2.npy',
        ]},
        # Enemy bullets: 6 entries indexed by enemy type (0=NONE dummy, 1=BURWOR, ..., 5=WIZARD)
        {'name': 'enemy_bullet', 'type': 'group', 'files': [
            'bullets/burwor.npy', 'bullets/burwor.npy',
            'bullets/garwor.npy', 'bullets/thorwor.npy',
            'bullets/worluk.npy', 'bullets/wizard.npy',
        ]},
        # Player bullet
        {'name': 'bullet', 'type': 'single', 'file': 'bullet.npy'},
        # Walls
        {'name': 'wall_horizontal', 'type': 'single', 'file': 'wall_horizontal.npy'},
        {'name': 'wall_vertical', 'type': 'single', 'file': 'wall_vertical.npy'},
        # Radar blips: 6 entries indexed by enemy type (0=empty, 1=burwor, ..., 5=wizard)
        {'name': 'radar_blip', 'type': 'group', 'files': [
            'radar/radar_empty.npy', 'radar/radar_burwor.npy',
            'radar/radar_garwor.npy', 'radar/radar_thorwor.npy',
            'radar/radar_worluk.npy', 'radar/radar_wizard.npy',
        ]},
        # Score digits
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/score_{}.npy'},
    )


#
# IMPORTANT
# FEATURES THAT WERE NOT IN SCOPE:
# - Pathfinding of the Worluk towards the teleporters
# - Spawning and movement of the Wizard
# - Speed increasing through level progression instead of just time in the level. This is connected to the point below.
# - More than 1 level. Our scope was the first level, the others are just cherries on top. (This can be configured via MAX_LEVEL)
#


class EntityPosition(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    direction: int  # Richtung aus UP, DOWN, LEFT, RIGHT


class WizardOfWorConstants(struct.PyTreeNode):
    # Window size
    WINDOW_WIDTH: int = struct.field(pytree_node=False, default=160)
    WINDOW_HEIGHT: int = struct.field(pytree_node=False, default=210)

    # 4 tuples for each direction
    BULLET_ORIGIN_UP: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 1))
    BULLET_ORIGIN_DOWN: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 7))
    BULLET_ORIGIN_LEFT: Tuple[int, int] = struct.field(pytree_node=False, default=(1, 4))
    BULLET_ORIGIN_RIGHT: Tuple[int, int] = struct.field(pytree_node=False, default=(7, 4))

    # Enemy Speed Up Timers
    SPEED_TIMER_1: int = struct.field(pytree_node=False, default=1500)
    SPEED_TIMER_2: int = struct.field(pytree_node=False, default=3000)
    SPEED_TIMER_3: int = struct.field(pytree_node=False, default=4500)
    SPEED_TIMER_MAX: int = struct.field(pytree_node=False, default=6000)
    SPEED_TIMER_BASE_MOD: int = struct.field(pytree_node=False, default=20)
    SPEED_TIMER_1_MOD: int = struct.field(pytree_node=False, default=16)
    SPEED_TIMER_2_MOD: int = struct.field(pytree_node=False, default=8)
    SPEED_TIMER_3_MOD: int = struct.field(pytree_node=False, default=4)
    SPEED_TIMER_MAX_MOD: int = struct.field(pytree_node=False, default=2)

    # Enemy invisibility timers
    MAX_LAST_SEEN: int = struct.field(pytree_node=False, default=200)
    INVISIBILITY_TIMER_GARWOR: int = struct.field(pytree_node=False, default=100)
    INVISIBILITY_TIMER_THORWOR: int = struct.field(pytree_node=False, default=100)

    # Directions
    NONE: int = Action.NOOP
    UP: int = Action.UP
    DOWN: int = Action.DOWN
    LEFT: int = Action.LEFT
    RIGHT: int = Action.RIGHT

    # Enemy types
    ENEMY_NONE: int = 0
    ENEMY_BURWOR: int = 1
    ENEMY_GARWOR: int = 2
    ENEMY_THORWOR: int = 3
    ENEMY_WORLUK: int = 4
    ENEMY_WIZARD: int = 5

    # POINTS
    POINTS_BURWOR: int = 100
    POINTS_GARWOR: int = 200
    POINTS_THORWOR: int = 500
    POINTS_WORLUK: int = 1000
    POINTS_WIZARD: int = 2500

    # Gameplay constants
    MAX_ENEMIES: int = 6
    MAX_LEVEL: int = 5
    MAX_LIVES: int = 3

    # Tolerance for spotting invisible enemies not quite in the same row/column
    SPOT_TOLERANCE = 4

    # Gameboards
    GAMEBOARD_1_WALLS_HORIZONTAL = jnp.array([
        [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]])

    # GAMEBOARD_1_WALLS_HORIZONTAL: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
    #     [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    #     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]]))

    GAMEBOARD_1_WALLS_VERTICAL = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]])

    GAMEBOARD_2_WALLS_HORIZONTAL = jnp.array([
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]])
    GAMEBOARD_2_WALLS_VERTICAL = jnp.array([
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])

    # Positions of the teleporters
    TELEPORTER_LEFT_POSITION: Tuple[int, int] = (-2, 20)
    TELEPORTER_RIGHT_POSITION: Tuple[int, int] = (108, 20)

    # Empty enemy array
    NO_ENEMY_POSITIONS = jnp.zeros((MAX_ENEMIES, 5), dtype=jnp.int32)

    # Position where the player spawns
    PLAYER_SPAWN_POSITION: Tuple[int, int, int] = (100, 50, LEFT)  # Startposition der Spielfigur

    # How far one walk step is
    STEP_SIZE: int = 1

    # IMPORTANT: About the coordinates
    # The board goes from 0,0 (top-left) to 110,60 (bottom-right)
    BOARD_SIZE: Tuple[int, int] = (110, 60)  # Size of the game board in tiles
    
    # Rendering
    DEATH_ANIMATION_STEPS = [10, 20]
    PLAYER_SIZE: Tuple[int, int] = (8, 8)
    ENEMY_SIZE: Tuple[int, int] = (8, 8)
    BULLET_SIZE: Tuple[int, int] = (2, 2)
    TILE_SIZE: Tuple[int, int] = (8, 8)
    RADAR_BLIP_SIZE: Tuple[int, int] = (2, 2)
    WALL_THICKNESS: int = 2
    RADAR_BLIP_GAP: int = 0
    BOARD_POSITION: Tuple[int, int] = (16, 64)
    GAME_AREA_OFFSET: Tuple[int, int] = (
        BOARD_POSITION[0] + WALL_THICKNESS + TILE_SIZE[0], BOARD_POSITION[1] + WALL_THICKNESS) # 
    LIVES_OFFSET: Tuple[int, int] = (100, 60)  # Offset for lives display
    LIVES_GAP: int = 5  # Gap between lives icons
    RADAR_OFFSET: Tuple[int, int] = (BOARD_POSITION[0] + 53, BOARD_POSITION[1] + 72)  # Offset for radar display
    SCORE_DIGIT_SPACING: int = 8
    SCORE_OFFSET: Tuple[int, int] = (BOARD_POSITION[0] + 80, BOARD_POSITION[1] - 16)  # Offset for score display

    @partial(jax.jit, static_argnums=(0,))
    def _get_wall_position(self, x: int, y: int, horizontal: bool) -> EntityPosition:
        """Returns the position of a wall based on its coordinates.
        :param x: The x-coordinate of the wall.
        :param y: The y-coordinate of the wall.
        :param horizontal: Whether the wall is horizontal or vertical.
        :return: An EntityPosition representing the wall's position.
        """
        return jax.lax.cond(
            horizontal,
            lambda _: EntityPosition(
                x=x * (self.WALL_THICKNESS + self.TILE_SIZE[0]) - self.WALL_THICKNESS,
                y=self.TILE_SIZE[1] + y * (self.WALL_THICKNESS + self.TILE_SIZE[1]),
                width=self.TILE_SIZE[0] + self.WALL_THICKNESS * 2,
                height=self.WALL_THICKNESS,
                direction=self.UP
            ),
            lambda _: EntityPosition(
                x=self.TILE_SIZE[0] + x * (self.WALL_THICKNESS + self.TILE_SIZE[0]),
                y=y * (self.WALL_THICKNESS + self.TILE_SIZE[1]) - self.WALL_THICKNESS,
                width=self.WALL_THICKNESS,
                height=self.TILE_SIZE[1] + self.WALL_THICKNESS * 2,
                direction=self.RIGHT
            ),
            operand=None
        )

    @staticmethod
    def get_walls_for_gameboard(gameboard: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the walls for the specified gameboard.
        :param gameboard: The gameboard for which the walls should be retrieved.
        :return: A tuple with the horizontal and vertical walls.
        """
        return jax.lax.cond(
            gameboard == 1,
            lambda: (
                WizardOfWorConstants.GAMEBOARD_1_WALLS_HORIZONTAL,
                WizardOfWorConstants.GAMEBOARD_1_WALLS_VERTICAL
            ),
            lambda: jax.lax.cond(
                gameboard == 2,
                lambda: (
                    WizardOfWorConstants.GAMEBOARD_2_WALLS_HORIZONTAL,
                    WizardOfWorConstants.GAMEBOARD_2_WALLS_VERTICAL
                ),
                lambda: (
                    jnp.zeros((5, 11), dtype=jnp.int32),
                    jnp.zeros((6, 10), dtype=jnp.int32)
                )
            )
        )


@struct.dataclass
class WizardOfWorObservation:
    player: EntityPosition
    enemies: chex.Array
    bullet: EntityPosition
    enemy_bullet: EntityPosition
    score: chex.Array
    lives: chex.Array


@struct.dataclass
class WizardOfWorInfo:
    all_rewards: chex.Array


@struct.dataclass
class WizardOfWorState:
    player: EntityPosition
    player_death_animation: int
    enemies: chex.Array  # Array of EntityPosition with length WizardOfWorConstants.MAX_ENEMIES
    gameboard: int
    bullet: EntityPosition
    enemy_bullet: EntityPosition  # Position of the enemy bullet, if any. They all share one bullet.
    idx_enemy_bullet_shot_by: int  # Index of the enemy that shot the bullet, if any.
    score: chex.Array
    lives: int
    doubled: bool  # Flag to indicate if the player has the double score power-up. This is only relevant for WORLUK and WIZARD enemies.
    frame_counter: int  # Counter for animations. This may not be needed since animation are tied to board position.
    rng_key: chex.PRNGKey  # Random key for JAX operations
    level: int
    game_over: bool
    teleporter: bool  # Flag to indicate if the teleporter is active.


def update_state(state: WizardOfWorState, player: EntityPosition = None, enemies: chex.Array = None,
                 gameboard: int = None, bullet: EntityPosition = None, enemy_bullet: EntityPosition = None,
                 score: chex.Array = None, idx_enemy_bullet_shot_by: int = None,
                 lives: int = None, doubled: bool = None, frame_counter: int = None, rng_key: chex.PRNGKey = None,
                 level: int = None, game_over: bool = None, teleporter: bool = None,
                 player_death_animation: int = None) -> WizardOfWorState:
    """
    Updates the state of the game. Only this method should be used to mutate the State object.
    Parameters not passed will be taken from the current state.
    :param state: The current state of the game.
    :param player: New position of the player character.
    :param enemies: New positions of the enemies.
    :param gameboard: New gameboard.
    :param bullet: New position of the shot.
    :param enemy_bullet: New position of the enemy bullet.
    :param idx_enemy_bullet_shot_by: Index of the enemy that shot the bullet.
    :param score: New score.
    :param lives: New number of lives.
    :param doubled: Flag indicating whether the player has the double score power-up.
    :param frame_counter: Counter for animations, e.g. player walking animation.
    :param rng_key: Random key for JAX operations.
    :param level: The current level of the game.
    :param game_over: Flag indicating whether the game is over.
    :param teleporter: Flag indicating whether the teleporter is active.
    :return: A new state of the game with the updated values.
    """
    return WizardOfWorState(
        player=player if player is not None else state.player,
        player_death_animation=player_death_animation if player_death_animation is not None else state.player_death_animation,
        enemies=enemies if enemies is not None else state.enemies,
        gameboard=gameboard if gameboard is not None else state.gameboard,
        bullet=bullet if bullet is not None else state.bullet,
        enemy_bullet=enemy_bullet if enemy_bullet is not None else state.enemy_bullet,
        idx_enemy_bullet_shot_by=idx_enemy_bullet_shot_by if idx_enemy_bullet_shot_by is not None else state.idx_enemy_bullet_shot_by,
        score=score if score is not None else state.score,
        lives=lives if lives is not None else state.lives,
        doubled=doubled if doubled is not None else state.doubled,
        frame_counter=frame_counter if frame_counter is not None else state.frame_counter,
        rng_key=rng_key if rng_key is not None else state.rng_key,
        level=level if level is not None else state.level,
        game_over=game_over if game_over is not None else state.game_over,
        teleporter=teleporter if teleporter is not None else state.teleporter
    )


class JaxWizardOfWor(JaxEnvironment[WizardOfWorState, WizardOfWorObservation, WizardOfWorInfo, WizardOfWorConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.FIRE, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.UPFIRE, Action.DOWNFIRE, Action.LEFTFIRE, Action.RIGHTFIRE],
        dtype=jnp.int32,
    )
    def __init__(self, consts: WizardOfWorConstants = None, reward_funcs: list[callable] = None):
        consts = consts or WizardOfWorConstants()
        super().__init__(consts)
        self.renderer = WizardOfWorRenderer(consts=consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[WizardOfWorObservation, WizardOfWorState]:
        """Reset the game to the initial state."""
        state = WizardOfWorState(
            player=EntityPosition(
                x=self.consts.PLAYER_SPAWN_POSITION[0],
                y=self.consts.PLAYER_SPAWN_POSITION[1],
                width=self.consts.PLAYER_SIZE[0],
                height=self.consts.PLAYER_SIZE[1],
                direction=self.consts.PLAYER_SPAWN_POSITION[2]
            ),
            player_death_animation=self.consts.DEATH_ANIMATION_STEPS[1] + 1,
            enemies=jnp.zeros(
                (self.consts.MAX_ENEMIES, 7),  # [x, y, direction, type, death_animation,timer,last_seen]
                dtype=jnp.int32
            ),
            gameboard=1,
            bullet=EntityPosition(
                x=-100,
                y=-100,
                width=self.consts.BULLET_SIZE[0],
                height=self.consts.BULLET_SIZE[1],
                direction=self.consts.NONE
            ),
            enemy_bullet=EntityPosition(
                x=-100,
                y=-100,
                width=self.consts.BULLET_SIZE[0],
                height=self.consts.BULLET_SIZE[1],
                direction=self.consts.NONE
            ),
            score=jnp.array(0),
            lives=self.consts.MAX_LIVES + 1,
            doubled=False,
            frame_counter=0,
            rng_key=jax.random.PRNGKey(666),  # Initialisiere den RNG
            level=0,
            game_over=False,
            teleporter=False,
            idx_enemy_bullet_shot_by=-1  # No enemy has shot a bullet yet
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: WizardOfWorState, action: chex.Array) -> Tuple[
        WizardOfWorObservation, WizardOfWorState, chex.Array, chex.Array, WizardOfWorInfo]:
        """ Advances the game state by one step based on the action taken.
        :param state: The current state of the game.
        :param action: The action taken by the player.
        :return: A tuple containing the new observation, the updated state, the reward, whether the game is done, and additional info.
        """
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        previous_state = state
        new_state = update_state(
            state=state,
            frame_counter=(state.frame_counter + 1) % 360,
            rng_key=jax.random.fold_in(state.rng_key, atari_action),
            # Teleporter is true if the frame_counter is below 180
            teleporter=(state.frame_counter < 180)
        )
        new_state = self._step_level_change(state=new_state)
        new_state = self._step_player_movement(state=new_state, action=atari_action)
        new_state = self._step_bullet_movement(state=new_state)
        new_state = self._step_enemy_movement(state=new_state)
        new_state = self._step_collision_detection(state=new_state)
        new_state = self._step_enemy_level_progression(state=new_state)
        new_state = jax.lax.cond(
            state.game_over,
            lambda: state,
            lambda: new_state
        )
        done = self._get_done(state=new_state)
        env_reward = self._get_reward(previous_state=previous_state, state=new_state)
        all_rewards = self._get_all_reward(previous_state=previous_state, state=new_state)
        info = self._get_info(state=new_state, all_rewards=all_rewards)
        observation = self._get_observation(state=new_state)

        return observation, new_state, env_reward, done, info

    def render(self, state: WizardOfWorState) -> jnp.ndarray:
        """Renders the current game state to an image."""
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        """Returns the action space of the game."""
        return spaces.Discrete(len(self.ACTION_SET))

    def image_space(self) -> spaces.Box:
        """Returns the image space of the game."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH, 3),
            dtype=jnp.uint8
        )

    def _get_done(self, state: WizardOfWorState) -> chex.Array:
        """Checks if the game is over."""
        return jnp.array(state.game_over, dtype=jnp.bool_)

    def _get_all_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState):
        """Calculates all rewards based on the previous and current state."""
        if self.reward_funcs is None:
            return jnp.zeros(1)
        return jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])

    def _get_info(self, state: WizardOfWorState, all_rewards: chex.Array = None) -> WizardOfWorInfo:
        """Returns additional information about the game state."""
        return WizardOfWorInfo(all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState) -> chex.Array:
        """Calculates the reward based on the previous and current state."""
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: WizardOfWorObservation) -> jnp.ndarray:
        return jnp.concatenate([
            jnp.array(obs.player.x).flatten(),
            jnp.array(obs.player.y).flatten(),
            jnp.array(obs.player.width).flatten(),
            jnp.array(obs.player.height).flatten(),
            jnp.array(obs.player.direction).flatten(),
            obs.enemies.flatten(),
            jnp.array(obs.bullet.x).flatten(),
            jnp.array(obs.bullet.y).flatten(),
            jnp.array(obs.bullet.width).flatten(),
            jnp.array(obs.bullet.height).flatten(),
            jnp.array(obs.bullet.direction).flatten(),
            jnp.array(obs.enemy_bullet.x).flatten(),
            jnp.array(obs.enemy_bullet.y).flatten(),
            jnp.array(obs.enemy_bullet.width).flatten(),
            jnp.array(obs.enemy_bullet.height).flatten(),
            jnp.array(obs.enemy_bullet.direction).flatten(),
            obs.score.flatten(),
            obs.lives.flatten()
        ]).astype(jnp.int32)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space of the game."""
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=-100, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=-100, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=-100, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=-100, high=210, shape=(), dtype=jnp.int32),
                "direction": spaces.Box(low=-1, high=4, shape=(), dtype=jnp.int32),  # NONE, UP, DOWN, LEFT, RIGHT
            }),
            "enemies": spaces.Box(low=-100, high=999999, shape=(6, 7), dtype=jnp.int32),
            "bullet": spaces.Dict({
                "x": spaces.Box(low=-100, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=-100, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=-100, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=-100, high=210, shape=(), dtype=jnp.int32),
                "direction": spaces.Box(low=-1, high=4, shape=(), dtype=jnp.int32),  # NONE, UP, DOWN, LEFT, RIGHT
            }),
            "enemy_bullet": spaces.Dict({
                "x": spaces.Box(low=-100, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=-100, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=-100, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=-100, high=210, shape=(), dtype=jnp.int32),
                "direction": spaces.Box(low=-1, high=4, shape=(), dtype=jnp.int32),  # NONE, UP, DOWN, LEFT, RIGHT
            }),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=-1, high=10, shape=(), dtype=jnp.int32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: WizardOfWorState) -> WizardOfWorObservation:
        """Converts the game state into an observation."""
        return WizardOfWorObservation(
            player=state.player,
            enemies=state.enemies,
            bullet=state.bullet,
            enemy_bullet=state.enemy_bullet,
            score=state.score,
            lives=jnp.array(state.lives, dtype=jnp.int32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _is_enemy_dead(self, enemy) -> bool:
        """Checks if an enemy is dead based on its death animation and type."""
        x, y, direction, type, death_animation, timer, last_seen = enemy
        return (death_animation > self.consts.DEATH_ANIMATION_STEPS[1]) | (type == self.consts.ENEMY_NONE)

    @partial(jax.jit, static_argnums=(0,))
    def _get_gameboard_for_level(self, level: int) -> int:
        """Returns the gameboard for the given level."""
        return 1 + ((level + 1) % 2)

    @partial(jax.jit, static_argnums=(0,))
    def _get_start_enemies(self, rng_key) -> chex.Array:
        """Generates the starting enemies for the game."""

        def _generate_single_enemy(rng_key) -> chex.Array:
            x = jax.random.randint(rng_key, shape=(), minval=0, maxval=11) * (
                    self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS)
            y = jax.random.randint(rng_key, shape=(), minval=0, maxval=6) * (
                    self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS)
            direction = jax.random.choice(rng_key, jnp.array(
                [self.consts.UP, self.consts.DOWN, self.consts.LEFT, self.consts.RIGHT]))
            return jnp.array([x, y, direction, self.consts.ENEMY_BURWOR, 0, 0, 0], dtype=jnp.int32)

        return jax.vmap(_generate_single_enemy)(jax.random.split(rng_key, self.consts.MAX_ENEMIES))

    @partial(jax.jit, static_argnums=(0,))
    def _get_bullet_origin_for_direction(self, direction: int) -> Tuple[int, int]:
        """Returns the origin offset for the bullet based on the direction."""
        return jax.lax.cond(
            jnp.equal(direction, self.consts.UP),
            lambda: self.consts.BULLET_ORIGIN_UP,
            lambda: jax.lax.cond(
                jnp.equal(direction, self.consts.DOWN),
                lambda: self.consts.BULLET_ORIGIN_DOWN,
                lambda: jax.lax.cond(
                    jnp.equal(direction, self.consts.LEFT),
                    lambda: self.consts.BULLET_ORIGIN_LEFT,
                    lambda: self.consts.BULLET_ORIGIN_RIGHT
                )
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def _positions_equal(self, pos1: EntityPosition, pos2: EntityPosition) -> bool:
        """Check if two positions are equal.
        :param pos1: The first position to compare.
        :param pos2: The second position to compare.
        :return: True if the positions are equal, False otherwise.
        """
        return jax.lax.cond(
            (pos1.x == pos2.x) & (pos1.y == pos2.y) &
            (pos1.width == pos2.width) & (pos1.height == pos2.height) &
            (pos1.direction == pos2.direction),
            lambda _: True,
            lambda _: False,
            operand=None
        )

    @partial(jax.jit, static_argnums=(0,))
    def _ensure_position_validity(self, state, old_position: EntityPosition,
                                  new_position: EntityPosition) -> EntityPosition:
        """
        Check if the position is valid.
        :param position: The position to check.
        :return: True if the position is valid, False otherwise.
        """
        # check both walls and boundaries using _check_boundaries and _check_walls
        boundary_position = self._check_boundaries(old_position=old_position, new_position=new_position)
        return self._check_walls(state, old_position=old_position, new_position=boundary_position)

    @partial(jax.jit, static_argnums=(0,))
    def _check_boundaries(self, old_position: EntityPosition, new_position: EntityPosition) -> EntityPosition:
        """Check if an entity collides with the boundaries of the gameboard.
        :param old_position: The old position of the entity.
        :param new_position: The new position of the entity.
        :return: The new position of the entity, or the old position if a collision occurs.
        """
        return jax.lax.cond(
            jnp.logical_or(
                jnp.logical_or(
                    new_position.x < 0,
                    new_position.x > self.consts.BOARD_SIZE[0] - new_position.width - self.consts.WALL_THICKNESS
                ),
                jnp.logical_or(
                    new_position.y < 0,
                    new_position.y > self.consts.BOARD_SIZE[1] - new_position.height - self.consts.WALL_THICKNESS
                )),
            lambda _: old_position,
            lambda _: new_position,
            operand=None
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_walls(self, state: WizardOfWorState, old_position: EntityPosition,
                     new_position: EntityPosition) -> EntityPosition:
        """Check if an entity collides with any walls in the gameboard.
        :param state: The current state of the game.
        :param old_position: The old position of the entity.
        :param new_position: The new position of the entity.
        :return: The new position of the entity, or the old position if a collision occurs.
        """
        horizontal_walls, vertical_walls = self.consts.get_walls_for_gameboard(gameboard=state.gameboard)
        H_horizontal, W_horizontal = horizontal_walls.shape
        H_vertical, W_vertical = vertical_walls.shape

        def check_wall_horizontal(idx):
            y = idx // W_horizontal
            x = idx % W_horizontal
            wall_position = self.consts._get_wall_position(x=x, y=y, horizontal=True)
            collision = jax.lax.cond(
                horizontal_walls[y, x] == 1,
                lambda _: self._check_collision(
                    new_position,
                    wall_position
                ),
                lambda _: False,
                operand=None
            )

            return collision

        def check_wall_vertical(idx):
            y = idx // W_vertical
            x = idx % W_vertical
            wall_position = self.consts._get_wall_position(x=x, y=y, horizontal=False)
            collision = jax.lax.cond(
                vertical_walls[y, x] == 1,
                lambda _: self._check_collision(
                    new_position,
                    wall_position
                ),
                lambda _: False,
                operand=None
            )
            return collision

        indices_horizontal = jnp.arange(H_horizontal * W_horizontal)
        indices_vertical = jnp.arange(H_vertical * W_vertical)
        collides_horizontal = jnp.any(jax.vmap(check_wall_horizontal)(indices_horizontal))
        collides_vertical = jnp.any(jax.vmap(check_wall_vertical)(indices_vertical))

        return jax.lax.cond(
            jnp.logical_or(collides_horizontal, collides_vertical),
            lambda _: old_position,  # If there is a collision, return the old position
            lambda _: new_position,  # If there is no collision, return the new position
            operand=None
        )

    def _check_collision(self, box1: EntityPosition, box2: EntityPosition) -> bool:
        """ Check if two boxes collide.
        :param box1: The first box.
        :param box2: The second box.
        :return: True if the boxes collide, False otherwise.
        """
        return jax.lax.cond(
            jnp.logical_not(
                (box1.x + box1.width <= box2.x) |
                (box1.x >= box2.x + box2.width) |
                (box1.y + box1.height <= box2.y) |
                (box1.y >= box2.y + box2.height)
            ),
            lambda: True,
            lambda: False
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_direction_to_player(self, enemy: EntityPosition, player: EntityPosition) -> int:
        """Returns the direction from the enemy to the player."""
        if enemy.x < player.x:
            return self.consts.RIGHT
        elif enemy.x > player.x:
            return self.consts.LEFT
        elif enemy.y < player.y:
            return self.consts.DOWN
        else:
            return self.consts.UP

    @partial(jax.jit, static_argnums=(0,))
    def _step_level_change(self, state):
        """Checks if all enemies are dead and handles level progression.
        :param state: The current state of the game.
        :return: The updated state of the game.
        """
        # if no enemies are left, we increase the level
        # dont assume that this is  if state.enemies == self.consts.NO_ENEMY_POSITIONS.
        # go through the enemies and check if they are all dead
        all_enemies_dead = jnp.all(jax.vmap(self._is_enemy_dead)(state.enemies))
        new_rng_key, rng_key = jax.random.split(state.rng_key)
        state = update_state(state=state, rng_key=new_rng_key)
        return jax.lax.cond(
            all_enemies_dead,
            lambda: jax.lax.cond(
                state.level + 1 > self.consts.MAX_LEVEL,
                lambda: update_state(
                    state=state,
                    game_over=True
                ),
                lambda: update_state(
                    state=state,
                    gameboard=self._get_gameboard_for_level(
                        level=state.level + 1
                    ),
                    level=state.level + 1,
                    player=EntityPosition(
                        x=-100,
                        y=-100,
                        width=self.consts.PLAYER_SIZE[0],
                        height=self.consts.PLAYER_SIZE[1],
                        direction=self.consts.NONE
                    ),
                    bullet=EntityPosition(
                        x=-100,
                        y=-100,
                        width=self.consts.BULLET_SIZE[0],
                        height=self.consts.BULLET_SIZE[1],
                        direction=self.consts.NONE
                    ),
                    enemy_bullet=EntityPosition(
                        x=-100,
                        y=-100,
                        width=self.consts.BULLET_SIZE[0],
                        height=self.consts.BULLET_SIZE[1],
                        direction=self.consts.NONE
                    ),
                    lives=jnp.minimum(state.lives + 1, self.consts.MAX_LIVES + 1),
                    enemies=self._get_start_enemies(rng_key),
                    idx_enemy_bullet_shot_by=-1,
                    player_death_animation=21,
                )
            ),
            lambda: state)

    @partial(jax.jit, static_argnums=(0,))
    def _step_player_movement(self, state, action):
        """
        Updates the player position based on the action taken.
        Handles movement only if player_death_animation == 0.
        If player_death_animation > 0, increments it up to 21.
        If player.direction == NONE and a non-NONE action is given, spawns the player at SPAWN_POSITION.
        """

        def _is_spawn_action(action):
            # Prüft, ob die Aktion eine echte Bewegung oder Schuss ist (kein NOOP)
            return ~jnp.equal(action, Action.NOOP)

        def _spawn_player_at_start():
            spawn_pos = self.consts.PLAYER_SPAWN_POSITION
            return update_state(
                state=state,
                player=EntityPosition(
                    x=spawn_pos[0],
                    y=spawn_pos[1],
                    width=self.consts.PLAYER_SIZE[0],
                    height=self.consts.PLAYER_SIZE[1],
                    direction=spawn_pos[2]
                ),
                lives=state.lives - 1,  # Reduce lives on respawn
                player_death_animation=0,
            )

        def handle_alive():
            def _get_new_position(player: EntityPosition, action: int) -> EntityPosition:
                return jax.lax.cond(
                    jnp.logical_or(jnp.equal(action, Action.UP), jnp.equal(action, Action.UPFIRE)),
                    lambda: EntityPosition(
                        x=player.x,
                        y=player.y - self.consts.STEP_SIZE,
                        width=player.width,
                        height=player.height,
                        direction=self.consts.UP
                    ),
                    lambda: jax.lax.cond(
                        jnp.logical_or(jnp.equal(action, Action.DOWN), jnp.equal(action, Action.DOWNFIRE)),
                        lambda: EntityPosition(
                            x=player.x,
                            y=player.y + self.consts.STEP_SIZE,
                            width=player.width,
                            height=player.height,
                            direction=self.consts.DOWN
                        ),
                        lambda: jax.lax.cond(
                            jnp.logical_or(jnp.equal(action, Action.LEFT),
                                           jnp.equal(action, Action.LEFTFIRE)),
                            lambda: EntityPosition(
                                x=player.x - self.consts.STEP_SIZE,
                                y=player.y,
                                width=player.width,
                                height=player.height,
                                direction=self.consts.LEFT
                            ),
                            lambda: jax.lax.cond(
                                jnp.logical_or(jnp.equal(action, Action.RIGHT),
                                               jnp.equal(action, Action.RIGHTFIRE)),
                                lambda: EntityPosition(
                                    x=player.x + self.consts.STEP_SIZE,
                                    y=player.y,
                                    width=player.width,
                                    height=player.height,
                                    direction=self.consts.RIGHT
                                ),
                                lambda: state.player  # No movement, return current position
                            )
                        ),
                    ),
                )

            proposed_new_position = _get_new_position(player=state.player, action=action)

            # Teleporter EntityPositions
            teleporter_left = EntityPosition(
                x=self.consts.TELEPORTER_LEFT_POSITION[0],
                y=self.consts.TELEPORTER_LEFT_POSITION[1],
                width=self.consts.WALL_THICKNESS,
                height=self.consts.TILE_SIZE[1],
                direction=self.consts.RIGHT
            )
            teleporter_right = EntityPosition(
                x=self.consts.TELEPORTER_RIGHT_POSITION[0],
                y=self.consts.TELEPORTER_RIGHT_POSITION[1],
                width=self.consts.WALL_THICKNESS,
                height=self.consts.TILE_SIZE[1],
                direction=self.consts.LEFT
            )
            # Zielpositionen nach Teleport
            teleporter_left_target = EntityPosition(
                x=self.consts.TELEPORTER_RIGHT_POSITION[0] - state.player.width,
                y=self.consts.TELEPORTER_RIGHT_POSITION[1],
                width=state.player.width,
                height=state.player.height,
                direction=self.consts.LEFT
            )
            teleporter_right_target = EntityPosition(
                x=self.consts.TELEPORTER_LEFT_POSITION[0] + self.consts.WALL_THICKNESS,
                y=self.consts.TELEPORTER_LEFT_POSITION[1],
                width=state.player.width,
                height=state.player.height,
                direction=self.consts.RIGHT
            )

            def teleport_if_needed(pos):
                return jax.lax.cond(
                    jnp.logical_and(
                        state.teleporter,
                        self._check_collision(proposed_new_position, teleporter_left)
                    ),
                    lambda: teleporter_left_target,
                    lambda: jax.lax.cond(
                        jnp.logical_and(
                            state.teleporter,
                            self._check_collision(proposed_new_position, teleporter_right)
                        ),
                        lambda: teleporter_right_target,
                        lambda: pos
                    )
                )

            checked_new_position = self._ensure_position_validity(
                state=state,
                old_position=state.player,
                new_position=proposed_new_position
            )

            checked_new_position = teleport_if_needed(checked_new_position)

            new_player_position = jax.lax.cond(
                self._positions_equal(pos1=proposed_new_position, pos2=checked_new_position),
                lambda _: checked_new_position,
                lambda _: jax.lax.cond(
                    jnp.logical_and(
                        jnp.not_equal(checked_new_position.direction, proposed_new_position.direction),
                        jnp.logical_not(
                            jnp.logical_and(
                                checked_new_position.x % (self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS) == 0,
                                checked_new_position.y % (self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS) == 0
                            )
                        )
                    ),
                    lambda _: _get_new_position(
                        player=checked_new_position,
                        action=checked_new_position.direction
                    ),
                    lambda _: checked_new_position,
                    operand=None
                ),
                operand=None
            )

            checked_new_position = self._ensure_position_validity(
                state=state,
                old_position=state.player,
                new_position=new_player_position
            )
            checked_new_position = teleport_if_needed(checked_new_position)

            # Bullet firing
            # if a fire action is taken and bullet is not currently active, fire a bullet
            new_bullet = jax.lax.cond(
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.equal(action, Action.FIRE),
                        jnp.equal(action, Action.UPFIRE)
                    ),
                    jnp.logical_or(
                        jnp.equal(action, Action.DOWNFIRE),
                        jnp.logical_or(
                            jnp.equal(action, Action.LEFTFIRE),
                            jnp.equal(action, Action.RIGHTFIRE)
                        )
                    )
                ) & (state.bullet.direction == self.consts.NONE),
                lambda: EntityPosition(
                    x=checked_new_position.x + self._get_bullet_origin_for_direction(checked_new_position.direction)[0],
                    y=checked_new_position.y + self._get_bullet_origin_for_direction(checked_new_position.direction)[1],
                    width=self.consts.BULLET_SIZE[0],
                    height=self.consts.BULLET_SIZE[1],
                    direction=checked_new_position.direction
                ),
                lambda: state.bullet
            )

            return update_state(
                state=state,
                player=checked_new_position,
                bullet=new_bullet,
            )

        def handle_death():
            new_animation = jnp.minimum(state.player_death_animation + 1, self.consts.DEATH_ANIMATION_STEPS[1] + 1)
            return update_state(
                state=state,
                player_death_animation=new_animation,
                player=EntityPosition(
                    x=state.player.x,
                    y=state.player.y,
                    width=state.player.width,
                    height=state.player.height,
                    direction=self.consts.NONE
                ),
                bullet=EntityPosition(
                    x=-100,
                    y=-100,
                    width=self.consts.BULLET_SIZE[0],
                    height=self.consts.BULLET_SIZE[1],
                    direction=self.consts.NONE
                )
            )

        def handle_game_over():
            return update_state(
                state=state,
                player_death_animation=self.consts.DEATH_ANIMATION_STEPS[1] + 1,
                player=EntityPosition(
                    x=-100,
                    y=-100,
                    width=self.consts.PLAYER_SIZE[0],
                    height=self.consts.PLAYER_SIZE[1],
                    direction=self.consts.NONE
                ),
                bullet=EntityPosition(
                    x=-100,
                    y=-100,
                    width=self.consts.BULLET_SIZE[0],
                    height=self.consts.BULLET_SIZE[1],
                    direction=self.consts.NONE
                ),
                game_over=True
            )

        def handle_spawn():
            return jax.lax.cond(
                (state.lives - 1) <= 0,
                lambda: handle_game_over(),
                lambda: jax.lax.cond(
                    _is_spawn_action(action),
                    lambda: _spawn_player_at_start(),
                    lambda: state
                )
            )

        return jax.lax.cond(
            state.player_death_animation > self.consts.DEATH_ANIMATION_STEPS[1],
            lambda: handle_spawn(),
            lambda: jax.lax.cond(
                state.player_death_animation > 0,
                lambda: handle_death(),
                lambda: jax.lax.cond(
                    (state.frame_counter % 4 == 0),
                    lambda: handle_alive(),
                    lambda: state,  # No movement if not the right frame
                )
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step_bullet_movement(self, state):
        """Updates the positions of the bullets in the game."""

        # move the bullet in the direction it is facing
        # if the bullet collides with a wall, it is removed
        # there are up to 2 bullets, one for the player and one for the enemy

        def move_bullet(bullet: EntityPosition) -> EntityPosition:
            new_x = bullet.x + self.consts.STEP_SIZE * (bullet.direction == self.consts.RIGHT) - \
                    self.consts.STEP_SIZE * (bullet.direction == self.consts.LEFT)
            new_y = bullet.y + self.consts.STEP_SIZE * (bullet.direction == self.consts.DOWN) - \
                    self.consts.STEP_SIZE * (bullet.direction == self.consts.UP)
            new_bullet = EntityPosition(
                x=new_x,
                y=new_y,
                width=bullet.width,
                height=bullet.height,
                direction=bullet.direction
            )
            # Check if the bullet is out of bounds or collides with a wall
            new_bullet = self._check_walls(
                state=state,
                old_position=bullet,
                new_position=new_bullet
            )
            new_bullet = self._check_boundaries(
                old_position=bullet,
                new_position=new_bullet
            )
            # If bullet == new_bullet, it means the bullet is out of bounds or collided with a wall
            # Such it collided and we remove it by resetting it.
            reset_bullet = EntityPosition(
                x=-100,
                y=-100,
                width=self.consts.BULLET_SIZE[0],
                height=self.consts.BULLET_SIZE[1],
                direction=self.consts.NONE
            )

            return jax.lax.cond(
                bullet.direction == self.consts.NONE,
                lambda _: bullet,
                lambda _: jax.lax.cond(
                    self._positions_equal(pos1=bullet, pos2=new_bullet),
                    lambda _: reset_bullet,  # Reset bullet if it collided with a wall or is out of bounds
                    lambda _: new_bullet,  # Otherwise return the new position
                    operand=None
                ),
                operand=None
            )

        new_bullet = move_bullet(state.bullet)
        new_enemy_bullet = move_bullet(state.enemy_bullet)
        return jax.lax.cond(
            state.frame_counter % 2 == 0,
            lambda: update_state(
                state=state,
                bullet=new_bullet,
                enemy_bullet=new_enemy_bullet
            ),
            lambda: state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step_enemy_movement(self, state):
        """Updates the positions of the enemies in the game."""

        # scan over all enemies and update the state based on their movement

        def move_enemy(carry, enemy_index):
            def _move_alive_default_enemy(state, enemy_index) -> WizardOfWorState:
                enemy = state.enemies[enemy_index]
                x, y, direction, enemy_type, death_animation, timer, last_seen = enemy

                is_on_tile = jnp.logical_and(
                    (x % (self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS) == 0),
                    (y % (self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS) == 0)
                )

                def move_enemy_between_tiles() -> WizardOfWorState:
                    # Move the enemy in the direction it is facing
                    new_x = x + self.consts.STEP_SIZE * (direction == self.consts.RIGHT) - \
                            self.consts.STEP_SIZE * (direction == self.consts.LEFT)
                    new_y = y + self.consts.STEP_SIZE * (direction == self.consts.DOWN) - \
                            self.consts.STEP_SIZE * (direction == self.consts.UP)
                    new_enemy_position = self._ensure_position_validity(
                        state=state,
                        old_position=EntityPosition(
                            x=x,
                            y=y,
                            width=self.consts.ENEMY_SIZE[0],
                            height=self.consts.ENEMY_SIZE[1],
                            direction=direction
                        ),
                        new_position=EntityPosition(
                            x=new_x,
                            y=new_y,
                            width=self.consts.ENEMY_SIZE[0],
                            height=self.consts.ENEMY_SIZE[1],
                            direction=direction
                        )
                    )
                    new_enemy = jnp.array([
                        new_enemy_position.x,
                        new_enemy_position.y,
                        new_enemy_position.direction,
                        enemy_type,
                        death_animation,
                        timer,
                        last_seen
                    ])
                    new_enemies = state.enemies.at[enemy_index].set(new_enemy)
                    return update_state(state=state, enemies=new_enemies)

                def move_enemy_on_tile() -> WizardOfWorState:
                    # Define directions based on the enemy's current direction
                    new_rng_key, rng_key = jax.random.split(state.rng_key)
                    current_direction = state.enemies[enemy_index][2]
                    forward = current_direction
                    left = jax.lax.cond(
                        current_direction == self.consts.UP, lambda: self.consts.LEFT,
                        lambda: jax.lax.cond(
                            current_direction == self.consts.DOWN, lambda: self.consts.RIGHT,
                            lambda: jax.lax.cond(
                                current_direction == self.consts.LEFT, lambda: self.consts.DOWN,
                                lambda: self.consts.UP
                            )
                        )
                    )
                    right = jax.lax.cond(
                        current_direction == self.consts.UP, lambda: self.consts.RIGHT,
                        lambda: jax.lax.cond(
                            current_direction == self.consts.DOWN, lambda: self.consts.LEFT,
                            lambda: jax.lax.cond(
                                current_direction == self.consts.LEFT, lambda: self.consts.UP,
                                lambda: self.consts.DOWN
                            )
                        )
                    )
                    back = jax.lax.cond(
                        current_direction == self.consts.UP, lambda: self.consts.DOWN,
                        lambda: jax.lax.cond(
                            current_direction == self.consts.DOWN, lambda: self.consts.UP,
                            lambda: jax.lax.cond(
                                current_direction == self.consts.LEFT, lambda: self.consts.RIGHT,
                                lambda: self.consts.LEFT
                            )
                        )
                    )

                    # Generate potential positions for all directions
                    def generate_position(direction):
                        new_x = x + self.consts.STEP_SIZE * (direction == self.consts.RIGHT) - \
                                self.consts.STEP_SIZE * (direction == self.consts.LEFT)
                        new_y = y + self.consts.STEP_SIZE * (direction == self.consts.DOWN) - \
                                self.consts.STEP_SIZE * (direction == self.consts.UP)
                        proposed_position = EntityPosition(
                            x=new_x,
                            y=new_y,
                            width=self.consts.ENEMY_SIZE[0],
                            height=self.consts.ENEMY_SIZE[1],
                            direction=direction
                        )
                        teleporter_left = EntityPosition(
                            x=self.consts.TELEPORTER_LEFT_POSITION[0],
                            y=self.consts.TELEPORTER_LEFT_POSITION[1],
                            width=self.consts.WALL_THICKNESS,
                            height=self.consts.TILE_SIZE[1],
                            direction=self.consts.RIGHT
                        )
                        teleporter_right = EntityPosition(
                            x=self.consts.TELEPORTER_RIGHT_POSITION[0],
                            y=self.consts.TELEPORTER_RIGHT_POSITION[1],
                            width=self.consts.WALL_THICKNESS,
                            height=self.consts.TILE_SIZE[1],
                            direction=self.consts.LEFT
                        )
                        # Zielpositionen nach Teleport
                        teleporter_left_target = EntityPosition(
                            x=self.consts.TELEPORTER_RIGHT_POSITION[0] - state.player.width,
                            y=self.consts.TELEPORTER_RIGHT_POSITION[1],
                            width=state.player.width,
                            height=state.player.height,
                            direction=self.consts.LEFT
                        )
                        teleporter_right_target = EntityPosition(
                            x=self.consts.TELEPORTER_LEFT_POSITION[0] + self.consts.WALL_THICKNESS,
                            y=self.consts.TELEPORTER_LEFT_POSITION[1],
                            width=state.player.width,
                            height=state.player.height,
                            direction=self.consts.RIGHT
                        )
                        return jax.lax.cond(
                            jnp.logical_and(
                                state.teleporter,
                                jnp.logical_or(
                                    self._check_collision(proposed_position, teleporter_left),
                                    self._check_collision(proposed_position, teleporter_right)
                                )
                            ),
                            lambda: jax.lax.cond(
                                self._check_collision(proposed_position, teleporter_left),
                                lambda: teleporter_left_target,
                                lambda: teleporter_right_target
                            ),
                            lambda: proposed_position
                        )

                    potential_position_forward = generate_position(forward)
                    potential_position_left = generate_position(left)
                    potential_position_right = generate_position(right)
                    potential_position_back = generate_position(back)

                    # Ensure validity of positions
                    valid_position_forward: EntityPosition = self._ensure_position_validity(state,
                                                                                            old_position=EntityPosition(
                                                                                                x,
                                                                                                y,
                                                                                                self.consts.ENEMY_SIZE[
                                                                                                    0],
                                                                                                self.consts.ENEMY_SIZE[
                                                                                                    1],
                                                                                                current_direction),
                                                                                            new_position=potential_position_forward)
                    valid_position_left: EntityPosition = self._ensure_position_validity(state,
                                                                                         old_position=EntityPosition(x,
                                                                                                                     y,
                                                                                                                     self.consts.ENEMY_SIZE[
                                                                                                                         0],
                                                                                                                     self.consts.ENEMY_SIZE[
                                                                                                                         1],
                                                                                                                     current_direction),
                                                                                         new_position=potential_position_left)
                    valid_position_right: EntityPosition = self._ensure_position_validity(state,
                                                                                          old_position=EntityPosition(x,
                                                                                                                      y,
                                                                                                                      self.consts.ENEMY_SIZE[
                                                                                                                          0],
                                                                                                                      self.consts.ENEMY_SIZE[
                                                                                                                          1],
                                                                                                                      current_direction),
                                                                                          new_position=potential_position_right)
                    valid_position_back: EntityPosition = self._ensure_position_validity(state,
                                                                                         old_position=EntityPosition(x,
                                                                                                                     y,
                                                                                                                     self.consts.ENEMY_SIZE[
                                                                                                                         0],
                                                                                                                     self.consts.ENEMY_SIZE[
                                                                                                                         1],
                                                                                                                     current_direction),
                                                                                         new_position=potential_position_back)

                    # Check for movement
                    moved_forward: bool = ~self._positions_equal(
                        EntityPosition(x, y, self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1], current_direction),
                        valid_position_forward)
                    moved_left: bool = ~self._positions_equal(
                        EntityPosition(x, y, self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1], current_direction),
                        valid_position_left)
                    moved_right: bool = ~self._positions_equal(
                        EntityPosition(x, y, self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1], current_direction),
                        valid_position_right)
                    moved_back: bool = ~self._positions_equal(
                        EntityPosition(x, y, self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1], current_direction),
                        valid_position_back)

                    # Select a random valid state
                    def select_state():
                        return jax.lax.cond(
                            jnp.logical_and(jnp.logical_and(moved_forward, moved_left), moved_right),
                            # All three directions possible
                            lambda: jax.random.choice(
                                rng_key,
                                jnp.array([valid_position_forward, valid_position_left, valid_position_right])
                            ),
                            lambda: jax.lax.cond(
                                jnp.logical_and(moved_forward, moved_left),  # Forward and left possible
                                lambda: jax.random.choice(
                                    rng_key, jnp.array([valid_position_forward, valid_position_left])
                                ),
                                lambda: jax.lax.cond(
                                    jnp.logical_and(moved_forward, moved_right),  # Forward and right possible
                                    lambda: jax.random.choice(
                                        rng_key, jnp.array([valid_position_forward, valid_position_right])
                                    ),
                                    lambda: jax.lax.cond(
                                        jnp.logical_and(moved_left, moved_right),  # Left and right possible
                                        lambda: jax.random.choice(
                                            rng_key, jnp.array([valid_position_left, valid_position_right])
                                        ),
                                        lambda: jax.lax.cond(
                                            moved_forward,  # Only forward possible
                                            lambda: jnp.array(valid_position_forward),
                                            lambda: jax.lax.cond(
                                                moved_left,  # Only left possible
                                                lambda: jnp.array(valid_position_left),
                                                lambda: jax.lax.cond(
                                                    moved_right,  # Only right possible
                                                    lambda: jnp.array(valid_position_right),
                                                    lambda: jnp.array(valid_position_back)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )

                    new_position = select_state()
                    new_position = EntityPosition(*new_position)
                    new_enemy = jnp.array([
                        new_position.x,
                        new_position.y,
                        new_position.direction,
                        enemy_type,
                        death_animation,
                        timer,
                        last_seen
                    ])
                    new_enemies = state.enemies.at[enemy_index].set(new_enemy)
                    return update_state(state=state, enemies=new_enemies, rng_key=new_rng_key)

                new_state = jax.lax.cond(
                    is_on_tile,
                    lambda: move_enemy_on_tile(),
                    lambda: move_enemy_between_tiles()
                )
                return new_state

            def _move_alive_worluk(state, enemy_index) -> WizardOfWorState:
                # worluk tries to path towards teleporter
                # TODO: implement pathfinding
                return _move_alive_default_enemy(state, enemy_index)

            def _move_alive_wizard(state, enemy_index) -> WizardOfWorState:
                # wizard teleports to a random position on the board facing the player
                # TODO: implement teleporting
                return _move_alive_default_enemy(state, enemy_index)

            def move_alive_enemy(state, enemy_index):
                # choose which movement function to use based on the enemy type
                enemy = state.enemies[enemy_index]
                x, y, direction, enemy_type, death_animation, timer, last_seen = enemy
                is_burwor = enemy_type == self.consts.ENEMY_BURWOR
                is_garwor = enemy_type == self.consts.ENEMY_GARWOR
                is_thorwor = enemy_type == self.consts.ENEMY_THORWOR
                is_worluk = enemy_type == self.consts.ENEMY_WORLUK
                is_wizard = enemy_type == self.consts.ENEMY_WIZARD
                is_default = jnp.logical_or(
                    jnp.logical_or(is_burwor, is_garwor),
                    is_thorwor
                )

                def try_fire_bullet(state, enemy_index):
                    enemy = state.enemies[enemy_index]
                    x, y, direction, _, _, _, _ = enemy
                    player = state.player

                    # Check if enemy is facing the player
                    is_facing_player = jax.lax.cond(
                        jnp.logical_and(
                            jnp.logical_or(
                                jnp.logical_and(jnp.equal(direction, self.consts.UP),
                                                jnp.logical_and(jnp.equal(x, player.x), y > player.y)),
                                jnp.logical_or(
                                    jnp.logical_and(jnp.equal(direction, self.consts.DOWN),
                                                    jnp.logical_and(jnp.equal(x, player.x), y < player.y)),
                                    jnp.logical_or(
                                        jnp.logical_and(jnp.equal(direction, self.consts.LEFT),
                                                        jnp.logical_and(jnp.equal(y, player.y), x > player.x)),
                                        jnp.logical_and(jnp.equal(direction, self.consts.RIGHT),
                                                        jnp.logical_and(jnp.equal(y, player.y), x < player.x))
                                    )
                                )
                            ),
                            True
                        ),
                        lambda: True,
                        lambda: False
                    )

                    # Check if enemy_bullet is already active
                    can_fire = jnp.logical_and(
                        state.enemy_bullet.direction == self.consts.NONE,
                        jnp.logical_and(
                            is_facing_player,
                            state.player_death_animation == 0  # Player must be alive to fire
                        )
                    )

                    # Fire bullet if possible
                    new_enemy_bullet = jax.lax.cond(
                        can_fire,
                        lambda: EntityPosition(
                            x=x + self._get_bullet_origin_for_direction(direction)[0],
                            y=y + self._get_bullet_origin_for_direction(direction)[1],
                            width=self.consts.BULLET_SIZE[0],
                            height=self.consts.BULLET_SIZE[1],
                            direction=direction
                        ),
                        lambda: state.enemy_bullet
                    )

                    new_idx_enemy_bullet_shot_by = jax.lax.cond(
                        can_fire,
                        lambda: enemy_index,
                        lambda: state.idx_enemy_bullet_shot_by
                    )

                    return update_state(
                        state=state,
                        enemy_bullet=new_enemy_bullet,
                        idx_enemy_bullet_shot_by=new_idx_enemy_bullet_shot_by
                    )

                return jax.lax.cond(
                    is_default,
                    lambda: try_fire_bullet(_move_alive_default_enemy(state, enemy_index), enemy_index),
                    lambda: jax.lax.cond(
                        is_worluk,
                        lambda: try_fire_bullet(_move_alive_worluk(state, enemy_index), enemy_index),
                        lambda: jax.lax.cond(
                            is_wizard,
                            lambda: try_fire_bullet(_move_alive_wizard(state, enemy_index), enemy_index),
                            lambda: state  # If no valid enemy type, return state unchanged
                        )
                    )
                )

            state = carry
            enemy = state.enemies[enemy_index]
            x, y, direction, enemy_type, death_animation, timer, last_seen = enemy
            timer = jnp.minimum(self.consts.SPEED_TIMER_MAX, timer + 1)  # here we increment the timer of the enemy
            last_seen = jnp.minimum(self.consts.MAX_LAST_SEEN, last_seen + 1)  # increment last seen timer
            state = jax.lax.cond(
                jnp.logical_or(
                    jnp.abs(state.player.x - state.enemies[enemy_index, 0]) <= self.consts.SPOT_TOLERANCE,
                    jnp.abs(state.player.y - state.enemies[enemy_index, 1]) <= self.consts.SPOT_TOLERANCE
                ),
                lambda: update_state(state=state, enemies=state.enemies.at[enemy_index, 6].set(0)),
                lambda: update_state(
                    state=state,
                    enemies=state.enemies.at[enemy_index, 5].set(timer).at[enemy_index, 6].set(last_seen)
                )
            )
            is_none = enemy_type == self.consts.ENEMY_NONE
            is_dying = death_animation > 0
            dying_enemy = jnp.array([x, y, direction, enemy_type, death_animation + 1, timer, last_seen])
            state_with_dying_enemy = update_state(
                state=state,
                enemies=state.enemies.at[enemy_index].set(dying_enemy)
            )

            enemy_step_modulo = jax.lax.cond(
                timer < self.consts.SPEED_TIMER_1,
                lambda: self.consts.SPEED_TIMER_BASE_MOD,
                lambda: jax.lax.cond(
                    timer < self.consts.SPEED_TIMER_2,
                    lambda: self.consts.SPEED_TIMER_1_MOD,
                    lambda: jax.lax.cond(
                        timer < self.consts.SPEED_TIMER_3,
                        lambda: self.consts.SPEED_TIMER_2_MOD,
                        lambda: jax.lax.cond(
                            timer < self.consts.SPEED_TIMER_MAX,
                            lambda: self.consts.SPEED_TIMER_3_MOD,
                            lambda: self.consts.SPEED_TIMER_MAX_MOD
                        )
                    )
                )
            )

            new_state = jax.lax.cond(
                is_none,
                lambda: state,
                lambda: jax.lax.cond(
                    is_dying,
                    lambda: state_with_dying_enemy,
                    lambda: jax.lax.cond(
                        state.frame_counter % enemy_step_modulo == 0,
                        lambda: move_alive_enemy(
                            state=state,
                            enemy_index=enemy_index),
                        lambda: state
                    )
                )
            )
            return new_state, None

        new_state, _ = jax.lax.scan(
            f=move_enemy,
            init=state,
            xs=jnp.arange(state.enemies.shape[0])
        )
        return new_state

    def _step_enemy_level_progression(self, state):
        """
        Handles the enemy level progression and the cleanup of dead enemies.
        """
        consts = self.consts
        enemies = state.enemies
        level = jnp.minimum(state.level, self.consts.MAX_ENEMIES)  # Level für Burwor->Garwor-Promotion, max 6
        score = state.score

        def get_random_tile_position(rng_key):
            x_idx = jax.random.randint(rng_key, shape=(), minval=0, maxval=11)
            y_idx = jax.random.randint(rng_key, shape=(), minval=0, maxval=6)
            x = x_idx * (consts.TILE_SIZE[0] + consts.WALL_THICKNESS)
            y = y_idx * (consts.TILE_SIZE[1] + consts.WALL_THICKNESS)
            direction = jax.random.choice(rng_key, jnp.array([consts.UP, consts.DOWN, consts.LEFT, consts.RIGHT]))
            return jnp.array([x, y, direction, consts.ENEMY_NONE, 0, 0, 0])

        def is_dead(enemy):
            return (enemy[4] > consts.DEATH_ANIMATION_STEPS[1]) & (enemy[3] != consts.ENEMY_NONE)

        def promote_burwor_to_garwor(enemies, idx):
            alive_burwors = jnp.sum(enemies[:, 3] == consts.ENEMY_BURWOR)
            return alive_burwors <= level

        def promote_thorwor_to_worluk(enemies, idx):
            alive_enemies = jnp.sum(enemies[:, 3] != consts.ENEMY_NONE)
            return alive_enemies <= 1

        def spawn_garwor(enemy, rng_key):
            pos = get_random_tile_position(rng_key)
            return jnp.array([pos[0], pos[1], enemy[2], consts.ENEMY_GARWOR, 0, enemy[5], 0])

        def spawn_thorwor(enemy, rng_key):
            pos = get_random_tile_position(rng_key)
            return jnp.array([pos[0], pos[1], enemy[2], consts.ENEMY_THORWOR, 0, enemy[5], 0])

        def spawn_worluk(enemy, rng_key):
            pos = get_random_tile_position(rng_key)
            return jnp.array([pos[0], pos[1], consts.RIGHT, consts.ENEMY_WORLUK, 0, 0, 0])

        def spawn_wizard(rng_key):
            return jax.random.bernoulli(rng_key, 0.5)

        def get_enemy_score(enemy_type, doubled):
            return jax.lax.switch(
                enemy_type,
                [
                    lambda: 0,
                    lambda: jax.lax.cond(doubled, lambda: consts.POINTS_BURWOR * 2, lambda: consts.POINTS_BURWOR),
                    lambda: jax.lax.cond(doubled, lambda: consts.POINTS_GARWOR * 2, lambda: consts.POINTS_GARWOR),
                    lambda: jax.lax.cond(doubled, lambda: consts.POINTS_THORWOR * 2, lambda: consts.POINTS_THORWOR),
                    lambda: jax.lax.cond(doubled, lambda: consts.POINTS_WORLUK * 2, lambda: consts.POINTS_WORLUK),
                    lambda: jax.lax.cond(doubled, lambda: consts.POINTS_WIZARD * 2, lambda: consts.POINTS_WIZARD),
                ]
            )

        def cleanup_enemy(enemy, idx, rng_key):
            # Gibt (neuer Feind, Score-Delta) zurück
            def handle_dead():
                # Punkte für getöteten Feind berechnen
                points = get_enemy_score(enemy[3], state.doubled)
                return jax.lax.cond(
                    (enemy[3] == consts.ENEMY_BURWOR) & promote_burwor_to_garwor(enemies, idx),
                    lambda: (spawn_garwor(enemy, rng_key), points),
                    lambda: jax.lax.cond(
                        enemy[3] == consts.ENEMY_GARWOR,
                        lambda: (spawn_thorwor(enemy, rng_key), points),
                        lambda: jax.lax.cond(
                            ((enemy[3] == consts.ENEMY_THORWOR) & (state.level > 1) & promote_thorwor_to_worluk(enemies,
                                                                                                                idx)),
                            lambda: (
                                spawn_worluk(enemy, rng_key),
                                points
                            ),
                            lambda: jax.lax.cond(
                                ((enemy[3] == consts.ENEMY_WORLUK) & spawn_wizard(rng_key) & (state.level > 1)),
                                lambda: (get_random_tile_position(rng_key).at[2].set(consts.ENEMY_WIZARD), points),
                                lambda: (jnp.array([0, 0, 0, consts.ENEMY_NONE, 0, 0, 0]), points)
                            )
                        )
                    )
                )

            return jax.lax.cond(
                is_dead(enemy),
                handle_dead,
                lambda: (enemy, 0)
            )

        rng_keys = jax.random.split(state.rng_key, consts.MAX_ENEMIES)
        results = jax.vmap(cleanup_enemy, in_axes=(0, 0, 0))(enemies, jnp.arange(consts.MAX_ENEMIES), rng_keys)
        new_enemies = results[0]
        score_delta = jnp.sum(results[1])
        return update_state(state=state, enemies=new_enemies, score=state.score + score_delta)

    @partial(jax.jit, static_argnums=(0,))
    def _step_collision_detection(self, state):
        """Detects and handles collisions between player, enemies, and bullets."""

        def check_player_enemy_collision(player: EntityPosition, enemy: chex.Array) -> bool:
            x, y, direction, enemy_type, death_animation, timer, last_seen = enemy
            return jax.lax.cond(
                (enemy_type == self.consts.ENEMY_NONE) | (death_animation > 0) | (state.player_death_animation != 0),
                lambda: False,
                lambda: self._check_collision(
                    player,
                    EntityPosition(
                        x=x,
                        y=y,
                        width=self.consts.ENEMY_SIZE[0],
                        height=self.consts.ENEMY_SIZE[1],
                        direction=direction
                    )
                )
            )

        # Check if player collides with any enemy
        player_enemy_collisions = jax.vmap(check_player_enemy_collision, in_axes=(None, 0))(state.player, state.enemies)
        player_enemy_collision = jnp.any(player_enemy_collisions)

        def handle_player_enemy_collision(state: WizardOfWorState) -> WizardOfWorState:
            # If player collides with an enemy, set death_animation to 1
            new_player = EntityPosition(
                x=state.player.x,
                y=state.player.y,
                width=state.player.width,
                height=state.player.height,
                direction=state.player.direction
            )
            return update_state(
                state=state,
                player=new_player,
                player_death_animation=1
            )

        # Handle player vs enemy collision
        new_state = jax.lax.cond(
            player_enemy_collision,
            lambda: handle_player_enemy_collision(state),
            lambda: state)

        def check_player_bullet_collision(player: EntityPosition, bullet: EntityPosition) -> bool:
            return jax.lax.cond(
                jnp.logical_or(
                    bullet.direction == self.consts.NONE,
                    state.player_death_animation != 0
                ),
                lambda: False,
                lambda: self._check_collision(
                    player,
                    EntityPosition(
                        x=bullet.x,
                        y=bullet.y,
                        width=bullet.width,
                        height=bullet.height,
                        direction=bullet.direction
                    )
                )
            )

        # Check if player collides with enemy bullet
        player_bullet_collision = check_player_bullet_collision(
            player=new_state.player,
            bullet=new_state.enemy_bullet
        )

        def handle_player_bullet_collision(state: WizardOfWorState) -> WizardOfWorState:
            # If player collides with enemy bullet, set death_animation to 1 and reset bullet
            new_player = EntityPosition(
                x=state.player.x,
                y=state.player.y,
                width=state.player.width,
                height=state.player.height,
                direction=state.player.direction
            )
            new_bullet = EntityPosition(
                x=-100,
                y=-100,
                width=state.enemy_bullet.width,
                height=state.enemy_bullet.height,
                direction=self.consts.NONE
            )
            return update_state(
                state=state,
                player=new_player,
                player_death_animation=1,
                enemy_bullet=new_bullet
            )

        # Handle player vs enemy bullet collision
        new_state = jax.lax.cond(
            player_bullet_collision,
            lambda: handle_player_bullet_collision(new_state),
            lambda: new_state
        )

        # For the enemy with player bullet collision we cant just check for any collision. we have to go through all enemies and check if this enemy collides with the player bullet and handle it in loop.
        def check_and_handle_enemy_bullet_collision(enemy: chex.Array, bullet: EntityPosition) -> chex.Array:
            x, y, direction, enemy_type, death_animation, timer, last_seen = enemy

            def handle_collision():
                # Setze death_animation nur auf 1, wenn es vorher 0 war
                return jax.lax.cond(
                    death_animation == 0,
                    lambda: jnp.array([x, y, direction, enemy_type, 1, timer, last_seen]),
                    lambda: enemy
                )

            def no_collision():
                return enemy

            def check_collision_inner():
                return jax.lax.cond(
                    self._check_collision(
                        EntityPosition(
                            x=x,
                            y=y,
                            width=self.consts.ENEMY_SIZE[0],
                            height=self.consts.ENEMY_SIZE[1],
                            direction=direction
                        ),
                        bullet
                    ),
                    handle_collision,
                    no_collision
                )

            return jax.lax.cond(
                (enemy_type == self.consts.ENEMY_NONE) | (death_animation > self.consts.DEATH_ANIMATION_STEPS[1]),
                no_collision,
                check_collision_inner
            )

        # Check if any enemy collides with player bullet
        new_enemies = jax.vmap(check_and_handle_enemy_bullet_collision, in_axes=(0, None))(new_state.enemies,
                                                                                           new_state.bullet)
        # Reset the bullet if it was hit
        new_bullet = jax.lax.cond(
            jnp.any(new_enemies[:, 4] == 1),  # If any enemy was hit
            lambda: EntityPosition(
                x=-100,
                y=-100,
                width=new_state.bullet.width,
                height=new_state.bullet.height,
                direction=self.consts.NONE
            ),
            lambda: new_state.bullet
        )

        # Wenn der Feind, der den Feind-Bullet abgefeuert hat, getötet wurde, entferne auch den Bullet und setze idx_enemy_bullet_shot_by auf -1
        enemy_bullet_removed = (
                (new_state.idx_enemy_bullet_shot_by >= 0) &
                (new_enemies[new_state.idx_enemy_bullet_shot_by, 4] > 0)  # Check if the enemy is dead
        )
        new_enemy_bullet = jax.lax.cond(
            enemy_bullet_removed,
            lambda: EntityPosition(
                x=-100,
                y=-100,
                width=new_state.enemy_bullet.width,
                height=new_state.enemy_bullet.height,
                direction=self.consts.NONE
            ),
            lambda: new_state.enemy_bullet
        )
        new_idx_enemy_bullet_shot_by = jax.lax.cond(
            enemy_bullet_removed,
            lambda: -100,
            lambda: new_state.idx_enemy_bullet_shot_by
        )

        # Update the state with new enemies and bullet
        new_state = update_state(
            state=new_state,
            enemies=new_enemies,
            bullet=new_bullet,
            enemy_bullet=new_enemy_bullet,
            idx_enemy_bullet_shot_by=new_idx_enemy_bullet_shot_by
        )

        return new_state


class WizardOfWorRenderer(JAXGameRenderer):
    def __init__(self, consts: WizardOfWorConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or WizardOfWorConstants()
        super().__init__(self.consts)

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "wizardofwor")

        final_asset_config = list(_get_default_asset_config())

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        bg_h, bg_w = self.BACKGROUND.shape
        board_x, board_y = self.consts.BOARD_POSITION
        full_bg = jnp.full(
            (self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH),
            self.jr.TRANSPARENT_ID,
            dtype=jnp.asarray(self.BACKGROUND).dtype,
        )
        self.BACKGROUND = full_bg.at[
            board_y:board_y + bg_h, board_x:board_x + bg_w
        ].set(jnp.asarray(self.BACKGROUND))

        self._cache_sprite_references()

    def _cache_sprite_references(self):
        # Character stacks: each (7, H, W) — frames 0-3 walk, 4-6 death
        self.PLAYER_STACK = self.SHAPE_MASKS['player']
        self.PLAYER_OFFSET = self.FLIP_OFFSETS['player']

        # Build unified enemy mega-stack for efficient JIT indexing by enemy_type
        # Index: 0=dummy(ENEMY_NONE), 1=BURWOR, 2=GARWOR, 3=THORWOR, 4=WORLUK, 5=WIZARD
        character_names = ['player', 'burwor', 'garwor', 'thorwor', 'worluk', 'wizard']
        character_stacks = [self.SHAPE_MASKS[n] for n in character_names]
        character_offsets = [self.FLIP_OFFSETS[n] for n in character_names]

        # Pad all to same dimensions
        max_h = max(s.shape[1] for s in character_stacks)
        max_w = max(s.shape[2] for s in character_stacks)

        padded_stacks = []
        for stack in character_stacks:
            h, w = stack.shape[1], stack.shape[2]
            padded = jnp.pad(
                stack,
                ((0, 0), (0, max_h - h), (0, max_w - w)),
                mode="constant",
                constant_values=self.jr.TRANSPARENT_ID
            )
            padded_stacks.append(padded)

        self.ALL_CHARACTER_STACKS = jnp.stack(padded_stacks)  # (6, 7, max_H, max_W)
        self.ALL_CHARACTER_OFFSETS = jnp.stack(character_offsets)  # (6, 2)

        # Other sprite references
        self.ENEMY_BULLET_STACK = self.SHAPE_MASKS['enemy_bullet']  # (6, H, W)
        self.BULLET_MASK = self.SHAPE_MASKS['bullet']  # (H, W)
        self.WALL_H_MASK = self.SHAPE_MASKS['wall_horizontal']  # (H, W)
        self.WALL_V_MASK = self.SHAPE_MASKS['wall_vertical']  # (H, W)
        self.RADAR_BLIP_STACK = self.SHAPE_MASKS['radar_blip']  # (6, H, W)
        self.DIGIT_MASKS = self.SHAPE_MASKS['digits']  # (10, H, W)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: WizardOfWorState):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self._render_gameboard(raster, state)
        raster = self._render_radar(raster, state)
        raster = self._render_player(raster, state)
        raster = self._render_enemies(raster, state)
        raster = self._render_player_bullet(raster, state)
        raster = self._render_enemy_bullet(raster, state)
        raster = self._render_score(raster, state)
        raster = self._render_lives(raster, state)
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def _render_gameboard(self, raster, state: WizardOfWorState):
        def _render_gameboard_walls(raster, state: WizardOfWorState):
            walls_horizontal, walls_vertical = self.consts.get_walls_for_gameboard(gameboard=state.gameboard)

            def _render_horizontal_wall(raster, x, y, is_wall):
                wall_x = self.consts.GAME_AREA_OFFSET[0] + (
                    x * (self.consts.WALL_THICKNESS + self.consts.TILE_SIZE[0]))
                wall_y = self.consts.GAME_AREA_OFFSET[1] + self.consts.TILE_SIZE[1] + (
                    y * (self.consts.WALL_THICKNESS + self.consts.TILE_SIZE[1]))
                return jax.lax.cond(
                    is_wall > 0,
                    lambda _: self.jr.render_at(raster, wall_x, wall_y, self.WALL_H_MASK),
                    lambda _: raster,
                    # raster
                    operand=None
                )

            def _render_vertical_wall(raster, x, y, is_wall):
                wall_x = self.consts.GAME_AREA_OFFSET[0] + self.consts.TILE_SIZE[0] + (
                    x * (self.consts.WALL_THICKNESS + self.consts.TILE_SIZE[0]))
                wall_y = self.consts.GAME_AREA_OFFSET[1] + (
                    y * (self.consts.WALL_THICKNESS + self.consts.TILE_SIZE[1]))
                return jax.lax.cond(
                    is_wall > 0,
                    lambda r: self.jr.render_at(r, wall_x, wall_y, self.WALL_V_MASK),
                    lambda r: r,
                    raster
                )

            def _render_horizontal_walls(raster, grid_vals):
                H, W = grid_vals.shape[:2]
                xs = jnp.repeat(jnp.arange(H)[:, None], W, axis=1)
                ys = jnp.repeat(jnp.arange(W)[None, :], H, axis=0)
                xs_f = xs.ravel()
                ys_f = ys.ravel()
                vals_f = grid_vals.reshape(-1, *grid_vals.shape[2:])

                def body(carry, elem):
                    r = carry
                    row, col, v = elem
                    r = _render_horizontal_wall(raster=r, x=col, y=row, is_wall=v)
                    return r, None

                raster_final, _ = jax.lax.scan(f=body, init=raster, xs=(xs_f, ys_f, vals_f))
                return raster_final

            def _render_vertical_walls(raster, grid_vals):
                H, W = grid_vals.shape[:2]
                xs = jnp.repeat(jnp.arange(H)[:, None], W, axis=1)
                ys = jnp.repeat(jnp.arange(W)[None, :], H, axis=0)
                xs_f = xs.ravel()
                ys_f = ys.ravel()
                vals_f = grid_vals.reshape(-1, *grid_vals.shape[2:])

                def body(carry, elem):
                    r = carry
                    row, col, v = elem
                    r = _render_vertical_wall(raster=r, x=col, y=row, is_wall=v)
                    return r, None

                raster_final, _ = jax.lax.scan(f=body, init=raster, xs=(xs_f, ys_f, vals_f))
                return raster_final

            new_raster = _render_horizontal_walls(raster=raster, grid_vals=walls_horizontal)
            new_raster = _render_vertical_walls(raster=new_raster, grid_vals=walls_vertical)
            return new_raster

        def _render_gameboard_teleporter(raster, state: WizardOfWorState):
            def _render_both_teleporter_walls(raster):
                raster = self.jr.render_at(
                    raster,
                    self.consts.GAME_AREA_OFFSET[0] + self.consts.TELEPORTER_LEFT_POSITION[0],
                    self.consts.GAME_AREA_OFFSET[1] + self.consts.TELEPORTER_LEFT_POSITION[1],
                    self.WALL_V_MASK
                )
                raster = self.jr.render_at(
                    raster,
                    self.consts.GAME_AREA_OFFSET[0] + self.consts.TELEPORTER_RIGHT_POSITION[0],
                    self.consts.GAME_AREA_OFFSET[1] + self.consts.TELEPORTER_RIGHT_POSITION[1],
                    self.WALL_V_MASK
                )
                return raster

            return jax.lax.cond(
                state.teleporter,
                lambda r: r,
                lambda r: _render_both_teleporter_walls(r),
                raster
            )

        # Background is already in the raster via create_object_raster
        new_raster = _render_gameboard_walls(raster=raster, state=state)
        new_raster = _render_gameboard_teleporter(raster=new_raster, state=state)
        return new_raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_radar(self, raster, state: WizardOfWorState):
        def body(carry, enemy):
            r = carry
            x, y, direction, enemy_type, death_animation, timer, last_seen = enemy

            radar_x = jax.lax.cond(
                direction == self.consts.LEFT,
                lambda: jnp.ceil(x / (self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS)),
                lambda: jnp.floor(x / (self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS))
            )
            radar_y = jax.lax.cond(
                direction == self.consts.UP,
                lambda: jnp.ceil(y / (self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS)),
                lambda: jnp.floor(y / (self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS))
            )

            blip_x = (self.consts.RADAR_OFFSET[0] + radar_x * self.consts.RADAR_BLIP_SIZE[0]).astype(jnp.int32)
            blip_y = (self.consts.RADAR_OFFSET[1] + radar_y * self.consts.RADAR_BLIP_SIZE[1]).astype(jnp.int32)
            blip_mask = self.RADAR_BLIP_STACK[enemy_type]

            r = jax.lax.cond(
                (enemy_type == self.consts.ENEMY_NONE) | (death_animation > 0),
                lambda r: r,
                lambda r: self.jr.render_at(r, blip_x, blip_y, blip_mask),
                r
            )
            return r, None

        raster_final, _ = jax.lax.scan(f=body, init=raster, xs=state.enemies)
        return raster_final

    @partial(jax.jit, static_argnums=(0,))
    def _render_character(self, raster, sprite_stack, flip_offset, entity: EntityPosition,
                          death_animation, is_worluk=False):
        direction = entity.direction
        render_x = self.consts.GAME_AREA_OFFSET[0] + entity.x
        render_y = self.consts.GAME_AREA_OFFSET[1] + entity.y

        def render_death_animation(raster):
            frame_index = jax.lax.cond(
                death_animation < self.consts.DEATH_ANIMATION_STEPS[0],
                lambda: 4,
                lambda: jax.lax.cond(
                    death_animation > self.consts.DEATH_ANIMATION_STEPS[1],
                    lambda: 6,
                    lambda: 5,
                )
            )
            mask = sprite_stack[frame_index]
            return self.jr.render_at(raster, render_x, render_y, mask)

        def render_normal(raster):
            frame_offset = ((entity.x + entity.y + 1) // 2) % 2
            # Special case for lives rendering: if y >= 60, force frame_offset to 1
            frame_offset = jax.lax.cond(
                entity.y >= 60,
                lambda: jnp.int32(1),
                lambda: frame_offset.astype(jnp.int32),
            )
            frame_index = jax.lax.cond(
                (direction == self.consts.LEFT) | (direction == self.consts.RIGHT),
                lambda: frame_offset,
                lambda: 2 + frame_offset,
            )
            mask = sprite_stack[frame_index]

            # Worluk sprites are never flipped
            flip_h = jax.lax.select(is_worluk, False, direction == self.consts.RIGHT)
            flip_v = jax.lax.select(is_worluk, False, direction == self.consts.UP)

            return jax.lax.cond(
                direction == self.consts.NONE,
                lambda r: r,
                lambda r: self.jr.render_at(
                    r, render_x, render_y, mask,
                    flip_horizontal=flip_h,
                    flip_vertical=flip_v,
                    flip_offset=flip_offset
                ),
                raster
            )

        return jax.lax.cond(
            death_animation > 0,
            render_death_animation,
            render_normal,
            raster
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_player(self, raster, state: WizardOfWorState):
        return self._render_character(
            raster,
            self.PLAYER_STACK,
            self.PLAYER_OFFSET,
            state.player,
            death_animation=state.player_death_animation,
            is_worluk=False
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemies(self, raster, state: WizardOfWorState):
        def body(carry, enemy):
            r = carry
            x, y, direction, enemy_type, death_animation, timer, last_seen = enemy

            sprite_stack = self.ALL_CHARACTER_STACKS[enemy_type]
            sprite_offset = self.ALL_CHARACTER_OFFSETS[enemy_type]

            is_worluk = enemy_type == self.consts.ENEMY_WORLUK

            # Handle invisibility for garwor and thorwor
            is_invisible = (
                ((enemy_type == self.consts.ENEMY_GARWOR) &
                 (last_seen >= self.consts.INVISIBILITY_TIMER_GARWOR) &
                 (death_animation == 0)) |
                ((enemy_type == self.consts.ENEMY_THORWOR) &
                 (last_seen >= self.consts.INVISIBILITY_TIMER_THORWOR) &
                 (death_animation == 0))
            )

            should_render = (enemy_type != self.consts.ENEMY_NONE) & ~is_invisible

            r = jax.lax.cond(
                should_render,
                lambda r: self._render_character(
                    r, sprite_stack, sprite_offset,
                    EntityPosition(x=x, y=y, direction=direction,
                                   width=self.consts.ENEMY_SIZE[0],
                                   height=self.consts.ENEMY_SIZE[1]),
                    death_animation=death_animation,
                    is_worluk=is_worluk
                ),
                lambda r: r,
                r
            )
            return r, None

        raster_final, _ = jax.lax.scan(f=body, init=raster, xs=state.enemies)
        return raster_final

    @partial(jax.jit, static_argnums=(0,))
    def _render_player_bullet(self, raster, state: WizardOfWorState):
        return jax.lax.cond(
            state.bullet.x >= 0,
            lambda _: self.jr.render_at_clipped(
                raster,
                self.consts.GAME_AREA_OFFSET[0] + state.bullet.x,
                self.consts.GAME_AREA_OFFSET[1] + state.bullet.y,
                self.BULLET_MASK
            ),
            lambda _: raster,
            # raster
            operand=None
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemy_bullet(self, raster, state: WizardOfWorState):
        enemy_type = jax.lax.cond(
            state.idx_enemy_bullet_shot_by >= 0,
            lambda: state.enemies[state.idx_enemy_bullet_shot_by, 3],
            lambda: self.consts.ENEMY_NONE
        )
        bullet_mask = self.ENEMY_BULLET_STACK[enemy_type]
        return jax.lax.cond(
            state.enemy_bullet.x >= 0,
            lambda r: self.jr.render_at(
                r,
                self.consts.GAME_AREA_OFFSET[0] + state.enemy_bullet.x,
                self.consts.GAME_AREA_OFFSET[1] + state.enemy_bullet.y,
                bullet_mask
            ),
            lambda r: r,
            raster
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_score(self, raster, state: WizardOfWorState):
        score_digits = self.jr.int_to_digits(state.score, max_digits=5)
        return self.jr.render_label_selective(
            raster,
            self.consts.SCORE_OFFSET[0],
            self.consts.SCORE_OFFSET[1],
            score_digits,
            self.DIGIT_MASKS,
            start_index=0,
            num_to_render=5,
            spacing=self.consts.SCORE_DIGIT_SPACING,
            max_digits_to_render=5
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_lives(self, raster, state: WizardOfWorState):
        def render_life(carry, i):
            r = carry
            r = jax.lax.cond(
                (state.lives - 1) > i,
                lambda r: self._render_character(
                    raster=r,
                    sprite_stack=self.PLAYER_STACK,
                    flip_offset=self.PLAYER_OFFSET,
                    entity=EntityPosition(
                        x=self.consts.LIVES_OFFSET[0] - (i * (self.consts.PLAYER_SIZE[0] + self.consts.LIVES_GAP)),
                        y=self.consts.LIVES_OFFSET[1],
                        width=self.consts.PLAYER_SIZE[0],
                        height=self.consts.PLAYER_SIZE[1],
                        direction=self.consts.LEFT
                    ),
                    death_animation=0,
                    is_worluk=False
                ),
                lambda r: r,
                r
            )
            return r, None

        indices = jnp.arange(start=0, stop=self.consts.MAX_LIVES, dtype=jnp.int32)
        new_raster, _ = jax.lax.scan(render_life, raster, indices)
        return new_raster