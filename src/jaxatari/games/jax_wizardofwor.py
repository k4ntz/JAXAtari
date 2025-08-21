from functools import partial
import os
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


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


class WizardOfWorConstants(NamedTuple):
    # Window size
    WINDOW_WIDTH: int = 160
    WINDOW_HEIGHT: int = 210

    # 4 tuples for each direction
    BULLET_ORIGIN_UP: Tuple[int, int] = (4, 1)
    BULLET_ORIGIN_DOWN: Tuple[int, int] = (4, 7)
    BULLET_ORIGIN_LEFT: Tuple[int, int] = (1, 4)
    BULLET_ORIGIN_RIGHT: Tuple[int, int] = (7, 4)

    # Enemy Speed Up Timers
    SPEED_TIMER_1 = 1500
    SPEED_TIMER_2 = 3000
    SPEED_TIMER_3 = 4500
    SPEED_TIMER_MAX = 6000
    SPEED_TIMER_BASE_MOD = 20
    SPEED_TIMER_1_MOD = 16
    SPEED_TIMER_2_MOD = 8
    SPEED_TIMER_3_MOD = 4
    SPEED_TIMER_MAX_MOD = 2

    # Enemy invisibility timers
    MAX_LAST_SEEN = 200
    INVISIBILITY_TIMER_GARWOR = 100
    INVISIBILITY_TIMER_THORWOR = 100

    # Directions
    NONE: int = Action.NOOP
    UP: int = Action.UP
    DOWN: int = Action.DOWN
    LEFT: int = Action.LEFT
    RIGHT: int = Action.RIGHT
    FIRE: int = Action.FIRE
    UPFIRE: int = Action.UPFIRE
    DOWNFIRE: int = Action.DOWNFIRE
    LEFTFIRE: int = Action.LEFTFIRE
    RIGHTFIRE: int = Action.RIGHTFIRE

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

    #Positions of the teleporters
    TELEPORTER_LEFT_POSITION: Tuple[int, int] = (-2, 20)
    TELEPORTER_RIGHT_POSITION: Tuple[int, int] = (108, 20)

    # Empty enemy array
    NO_ENEMY_POSITIONS = jnp.zeros((MAX_ENEMIES, 5), dtype=jnp.int32)

    # Position where the player spawns
    PLAYER_SPAWN_POSITION: Tuple[int, int, int] = (100, 50, LEFT)  # Startposition der Spielfigur

    # How far one walk step is
    STEP_SIZE: int = 5

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
        BOARD_POSITION[0] + WALL_THICKNESS + TILE_SIZE[0], BOARD_POSITION[1] + WALL_THICKNESS)
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


class WizardOfWorObservation(NamedTuple):
    player: EntityPosition
    enemies: chex.Array
    bullet: EntityPosition
    enemy_bullet: EntityPosition
    score: chex.Array
    lives: chex.Array


class WizardOfWorInfo(NamedTuple):
    all_rewards: chex.Array


class WizardOfWorState(NamedTuple):
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
        previous_state = state
        new_state = update_state(
            state=state,
            frame_counter=(state.frame_counter + 1) % 360,
            rng_key=jax.random.fold_in(state.rng_key, action),
            # Teleporter is true if the frame_counter is below 180
            teleporter=(state.frame_counter < 180)
        )
        new_state = self._step_level_change(state=new_state)
        new_state = self._step_player_movement(state=new_state, action=action)
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
        return spaces.Discrete(6)  # Platzhalter

    @partial(jax.jit, static_argnums=(0,))
    def observation_space(self) -> spaces.Dict:
        """Returns the observation space of the game."""
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.BOARD_SIZE[0], shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.BOARD_SIZE[1], shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.PLAYER_SIZE[0], shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.PLAYER_SIZE[1], shape=(), dtype=jnp.int32),
                "direction": spaces.Discrete(5),  # NONE, UP, DOWN, LEFT, RIGHT
            }),
            "enemies": spaces.Box(low=0, high=999999, shape=(self.consts.MAX_ENEMIES, 7), dtype=jnp.int32),
            "bullet": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.BOARD_SIZE[0], shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.BOARD_SIZE[1], shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.BULLET_SIZE[0], shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.BULLET_SIZE[1], shape=(), dtype=jnp.int32),
                "direction": spaces.Discrete(5),  # NONE, UP, DOWN, LEFT, RIGHT
            }),
            "enemy_bullet": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.BOARD_SIZE[0], shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.BOARD_SIZE[1], shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.BULLET_SIZE[0], shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.BULLET_SIZE[1], shape=(), dtype=jnp.int32),
                "direction": spaces.Discrete(5),  # NONE, UP, DOWN, LEFT, RIGHT
            }),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=10, shape=(), dtype=jnp.int32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def image_space(self) -> spaces.Box:
        """Returns the image space of the game."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: WizardOfWorState) -> chex.Array:
        """Checks if the game is over."""
        return jnp.array(state.game_over, dtype=jnp.bool_)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState):
        """Calculates all rewards based on the previous and current state."""
        if self.reward_funcs is None:
            return jnp.zeros(1)
        return jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: WizardOfWorState, all_rewards: chex.Array = None) -> WizardOfWorInfo:
        """Returns additional information about the game state."""
        return WizardOfWorInfo(all_rewards=all_rewards)

    def _get_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState) -> chex.Array:
        """Calculates the reward based on the previous and current state."""
        return state.score - previous_state.score

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
            return ~jnp.equal(action, self.consts.NONE)

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
                    jnp.logical_or(jnp.equal(action, self.consts.UP), jnp.equal(action, self.consts.UPFIRE)),
                    lambda: EntityPosition(
                        x=player.x,
                        y=player.y - self.consts.STEP_SIZE,
                        width=player.width,
                        height=player.height,
                        direction=self.consts.UP
                    ),
                    lambda: jax.lax.cond(
                        jnp.logical_or(jnp.equal(action, self.consts.DOWN), jnp.equal(action, self.consts.DOWNFIRE)),
                        lambda: EntityPosition(
                            x=player.x,
                            y=player.y + self.consts.STEP_SIZE,
                            width=player.width,
                            height=player.height,
                            direction=self.consts.DOWN
                        ),
                        lambda: jax.lax.cond(
                            jnp.logical_or(jnp.equal(action, self.consts.LEFT),
                                           jnp.equal(action, self.consts.LEFTFIRE)),
                            lambda: EntityPosition(
                                x=player.x - self.consts.STEP_SIZE,
                                y=player.y,
                                width=player.width,
                                height=player.height,
                                direction=self.consts.LEFT
                            ),
                            lambda: jax.lax.cond(
                                jnp.logical_or(jnp.equal(action, self.consts.RIGHT),
                                               jnp.equal(action, self.consts.RIGHTFIRE)),
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
                        jnp.equal(action, self.consts.FIRE),
                        jnp.equal(action, self.consts.UPFIRE)
                    ),
                    jnp.logical_or(
                        jnp.equal(action, self.consts.DOWNFIRE),
                        jnp.logical_or(
                            jnp.equal(action, self.consts.LEFTFIRE),
                            jnp.equal(action, self.consts.RIGHTFIRE)
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
    def __init__(self, consts: WizardOfWorConstants = None):
        super().__init__()
        self.consts = consts or WizardOfWorConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_BURWOR,
            self.SPRITE_GARWOR,
            self.SPRITE_THORWOR,
            self.SPRITE_WORLUK,
            self.SPRITE_WIZARD,
            self.SPRITE_BULLET,
            self.SPRITE_ENEMY_BULLET,
            self.SCORE_DIGIT_SPRITES,
            self.SPRITE_WALL_HORIZONTAL,
            self.SPRITE_WALL_VERTICAL,
            self.SPRITE_RADAR_BLIP,
            self.SCORE_DIGIT_SPRITES
        ) = self.load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/background.npy"))
        player0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/player_0.npy"))
        player1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/player_1.npy"))
        player2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/player_2.npy"))
        player3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/player_3.npy"))
        player_death0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/death_0.npy"))
        player_death1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/death_1.npy"))
        player_death2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/death_2.npy"))
        burwor0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/burwor_0.npy"))
        burwor1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/burwor_1.npy"))
        burwor2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/burwor_2.npy"))
        burwor3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/burwor_3.npy"))
        burwor_death0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/death_0.npy"))
        burwor_death1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/death_1.npy"))
        burwor_death2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/death_2.npy"))
        burwor_bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/bullets/burwor.npy"))
        garwor0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/garwor/garwor_0.npy"))
        garwor1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/garwor/garwor_1.npy"))
        garwor2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/garwor/garwor_2.npy"))
        garwor3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/garwor/garwor_3.npy"))
        garwor_death0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/garwor/death_0.npy"))
        garwor_death1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/garwor/death_1.npy"))
        garwor_death2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/garwor/death_2.npy"))
        garwor_bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/bullets/garwor.npy"))
        thorwor0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/thorwor/thorwor_0.npy"))
        thorwor1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/thorwor/thorwor_1.npy"))
        thorwor2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/thorwor/thorwor_2.npy"))
        thorwor3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/thorwor/thorwor_3.npy"))
        thorwor_death0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/thorwor/death_0.npy"))
        thorwor_death1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/thorwor/death_1.npy"))
        thorwor_death2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/thorwor/death_2.npy"))
        thorwor_bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/bullets/thorwor.npy"))
        worluk0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/worluk/worluk_0.npy"))
        worluk1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/worluk/worluk_1.npy"))
        worluk2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/worluk/worluk_2.npy"))
        worluk3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/worluk/worluk_3.npy"))
        worluk_death0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/worluk/death_0.npy"))
        worluk_death1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/worluk/death_1.npy"))
        worluk_death2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/worluk/death_2.npy"))
        worluk_bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/bullets/worluk.npy"))
        wizard0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/wizard/wizard_0.npy"))
        wizard1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/wizard/wizard_1.npy"))
        wizard2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/wizard/wizard_2.npy"))
        wizard3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/wizard/wizard_3.npy"))
        wizard_death0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/wizard/death_0.npy"))
        wizard_death1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/wizard/death_1.npy"))
        wizard_death2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/wizard/death_2.npy"))
        wizard_bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/bullets/wizard.npy"))
        bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/bullet.npy"))
        wall_horizontal = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/wall_horizontal.npy"))
        wall_vertical = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/wall_vertical.npy"))
        radar_blip_empty = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_empty.npy"))
        radar_blip_burwor = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_burwor.npy"))
        radar_blip_garwor = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_garwor.npy"))
        radar_blip_thorwor = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_thorwor.npy"))
        radar_blip_worluk = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_worluk.npy"))
        radar_blip_wizard = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_wizard.npy"))

        SPRITE_PLAYER = jnp.stack([player0, player1, player2, player3, player_death0, player_death1, player_death2],
                                  axis=0)
        SPRITE_BURWOR = jnp.stack([burwor0, burwor1, burwor2, burwor3, burwor_death0, burwor_death1, burwor_death2],
                                  axis=0)
        SPRITE_GARWOR = jnp.stack([garwor0, garwor1, garwor2, garwor3, garwor_death0, garwor_death1, garwor_death2],
                                  axis=0)
        SPRITE_THORWOR = jnp.stack(
            [thorwor0, thorwor1, thorwor2, thorwor3, thorwor_death0, thorwor_death1, thorwor_death2], axis=0)
        SPRITE_WORLUK = jnp.stack([worluk0, worluk1, worluk2, worluk3, worluk_death0, worluk_death1, worluk_death2],
                                  axis=0)
        SPRITE_WIZARD = jnp.stack([wizard0, wizard1, wizard2, wizard3, wizard_death0, wizard_death1, wizard_death2],
                                  axis=0)
        SPRITE_RADAR_BLIP = jnp.stack([radar_blip_empty, radar_blip_burwor, radar_blip_garwor,
                                       radar_blip_thorwor, radar_blip_worluk, radar_blip_wizard], axis=0)

        SPRITE_BG = jnp.expand_dims(bg, axis=0)
        SPRITE_BULLET = jnp.expand_dims(bullet, axis=0)
        SPRITE_ENEMY_BULLET = jnp.stack(
            [burwor_bullet, burwor_bullet, garwor_bullet, thorwor_bullet, worluk_bullet, wizard_bullet],
            axis=0)
        SPRITE_WALL_HORIZONTAL = jnp.expand_dims(wall_horizontal, axis=0)
        SPRITE_WALL_VERTICAL = jnp.expand_dims(wall_vertical, axis=0)

        SCORE_DIGIT_SPRITES = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/wizardofwor/digits/score_{}.npy"),
            num_chars=10,
        )

        return (
            SPRITE_BG,
            SPRITE_PLAYER,
            SPRITE_BURWOR,
            SPRITE_GARWOR,
            SPRITE_THORWOR,
            SPRITE_WORLUK,
            SPRITE_WIZARD,
            SPRITE_BULLET,
            SPRITE_ENEMY_BULLET,
            SCORE_DIGIT_SPRITES,
            SPRITE_WALL_HORIZONTAL,
            SPRITE_WALL_VERTICAL,
            SPRITE_RADAR_BLIP,
            SCORE_DIGIT_SPRITES
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: WizardOfWorState):
        # Raster initialisieren
        raster = jr.create_initial_frame(width=self.consts.WINDOW_WIDTH, height=self.consts.WINDOW_HEIGHT)
        raster = self._render_gameboard(raster=raster, state=state)
        raster = self._render_radar(raster=raster, state=state)
        raster = self._render_player(raster=raster, state=state)
        raster = self._render_enemies(raster=raster, state=state)
        raster = self._render_player_bullet(raster=raster, state=state)
        raster = self._render_enemy_bullet(raster=raster, state=state)
        raster = self._render_score(raster=raster, state=state)
        raster = self._render_lives(raster=raster, state=state)
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_gameboard(self, raster, state: WizardOfWorState):
        def _render_gameboard_background(raster):
            return jr.render_at(
                raster=raster,
                sprite_frame=jr.get_sprite_frame(self.SPRITE_BG, 0),
                x=self.consts.BOARD_POSITION[0],
                y=self.consts.BOARD_POSITION[1]
            )

        def _render_gameboard_walls(raster, state: WizardOfWorState):
            walls_horizontal, walls_vertical = self.consts.get_walls_for_gameboard(gameboard=state.gameboard)

            def _render_horizontal_wall(raster, x: int, y: int, is_wall: int):
                def _get_raster_x_for_horizontal_wall(x):
                    return self.consts.GAME_AREA_OFFSET[0] + (
                            x * (self.consts.WALL_THICKNESS + self.consts.TILE_SIZE[0]))

                def _get_raster_y_for_horizontal_wall(y):
                    return self.consts.GAME_AREA_OFFSET[1] + self.consts.TILE_SIZE[1] + (
                            y * (self.consts.WALL_THICKNESS + self.consts.TILE_SIZE[1]))

                return jax.lax.cond(
                    is_wall > 0,
                    lambda _: jr.render_at(
                        raster=raster,
                        sprite_frame=jr.get_sprite_frame(self.SPRITE_WALL_HORIZONTAL, 0),
                        x=_get_raster_x_for_horizontal_wall(x),
                        y=_get_raster_y_for_horizontal_wall(y)
                    ),
                    lambda _: raster,
                    operand=None
                )

            def _render_vertical_wall(raster, x, y, is_wall):
                def _get_raster_x_for_vertical_wall(x):
                    return self.consts.GAME_AREA_OFFSET[0] + self.consts.TILE_SIZE[0] + (
                            x * (self.consts.WALL_THICKNESS + self.consts.TILE_SIZE[0]))

                def _get_raster_y_for_vertical_wall(y):
                    return self.consts.GAME_AREA_OFFSET[1] + (
                            y * (self.consts.WALL_THICKNESS + self.consts.TILE_SIZE[1]))

                return jax.lax.cond(
                    is_wall > 0,
                    lambda _: jr.render_at(
                        raster=raster,
                        sprite_frame=jr.get_sprite_frame(self.SPRITE_WALL_VERTICAL, 0),
                        x=_get_raster_x_for_vertical_wall(x=x),
                        y=_get_raster_y_for_vertical_wall(y=y)
                    ),
                    lambda _: raster,
                    operand=None
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

                init = raster
                elems = (xs_f, ys_f, vals_f)
                raster_final, _ = jax.lax.scan(
                    f=body,
                    init=init,
                    xs=elems
                )
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

                init = raster
                elems = (xs_f, ys_f, vals_f)
                raster_final, _ = jax.lax.scan(
                    f=body,
                    init=init,
                    xs=elems
                )
                return raster_final

            new_raster = _render_horizontal_walls(
                raster=raster,
                grid_vals=walls_horizontal
            )
            new_raster = _render_vertical_walls(
                raster=new_raster,
                grid_vals=walls_vertical
            )
            return new_raster

        def _render_gameboard_teleporter(raster, state: WizardOfWorState):
            # if teleporter is not active render two walls at TELEPORTER_LEFT_POSITION and TELEPORTER_RIGHT_POSITION.
            return jax.lax.cond(
                state.teleporter,
                lambda _: raster,  # If teleporter is active, do not render walls
                lambda _: _render_both_teleporter_walls(raster),
                operand=None
            )

        def _render_both_teleporter_walls(raster):
            # Render the left teleporter wall
            raster = jr.render_at(
                raster=raster,
                sprite_frame=jr.get_sprite_frame(self.SPRITE_WALL_VERTICAL, 0),
                x=self.consts.GAME_AREA_OFFSET[0] + self.consts.TELEPORTER_LEFT_POSITION[0],
                y=self.consts.GAME_AREA_OFFSET[1] + self.consts.TELEPORTER_LEFT_POSITION[1]
            )
            # Render the right teleporter wall
            raster = jr.render_at(
                raster=raster,
                sprite_frame=jr.get_sprite_frame(self.SPRITE_WALL_VERTICAL, 0),
                x=self.consts.GAME_AREA_OFFSET[0] + self.consts.TELEPORTER_RIGHT_POSITION[0],
                y=self.consts.GAME_AREA_OFFSET[1] + self.consts.TELEPORTER_RIGHT_POSITION[1]
            )
            return raster

        new_raster = _render_gameboard_background(raster=raster)
        new_raster = _render_gameboard_walls(raster=new_raster, state=state)
        new_raster = _render_gameboard_teleporter(raster=new_raster, state=state)
        return new_raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_radar(self, raster, state: WizardOfWorState):
        # We calculate the radar blips based on the enemies' positions.
        # If a monster is fully on a tile, it will be rendered as a radar blip.
        # If the monster is between tiles, it will be rendered as the tile its back is facing.
        def _render_radar_blip(raster, x, y, direction, enemy_type, death_animation):
            radar_x = jax.lax.cond(
                direction == self.consts.LEFT,
                lambda _: jnp.ceil(x / (self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS)),
                lambda _: jax.lax.cond(
                    direction == self.consts.RIGHT,
                    lambda _: jnp.floor(x / (self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS)),
                    lambda _: jnp.floor(x / (self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS)),
                    operand=None
                ),
                operand=None
            )
            radar_y = jax.lax.cond(
                direction == self.consts.UP,
                lambda _: jnp.ceil(y / (self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS)),
                lambda _: jax.lax.cond(
                    direction == self.consts.DOWN,
                    lambda _: jnp.floor(y / (self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS)),
                    lambda _: jnp.floor(y / (self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS)),
                    operand=None
                ),
                operand=None
            )
            return jax.lax.cond(
                (enemy_type == self.consts.ENEMY_NONE) | (death_animation > 0),
                lambda _: raster,
                lambda _: jr.render_at(
                    raster=raster,
                    sprite_frame=jr.get_sprite_frame(self.SPRITE_RADAR_BLIP, enemy_type),
                    x=self.consts.RADAR_OFFSET[0] + radar_x * self.consts.RADAR_BLIP_SIZE[0],
                    y=self.consts.RADAR_OFFSET[1] + radar_y * self.consts.RADAR_BLIP_SIZE[1]
                ),
                operand=None
            )

        def body(carry, enemy):
            r = carry
            x, y, direction, enemy_type, death_animation, timer, last_seen = enemy
            # Calculate the radar blip position based on the enemy's position
            # If the monster is between tiles, it will be rendered as the tile opposite to the direction it is facing.
            # so we have to use direction to determine the radar blip position.
            # 0 <= radar_x < self.consts.BOARD_SIZE[0]
            # 0 <= radar_y < self.consts.BOARD_SIZE[1]

            # Render the radar blip at the calculated position
            r = _render_radar_blip(raster=r, x=x, y=y, direction=direction, enemy_type=enemy_type,
                                   death_animation=death_animation)
            return r, None

        raster_final, _ = jax.lax.scan(
            f=body,
            init=raster,
            xs=state.enemies
        )
        return raster_final

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemies(self, raster, state: WizardOfWorState):
        def _render_enemies(self, raster, state: WizardOfWorState):
            def body(carry, enemy):
                r = carry
                x, y, direction, enemy_type, death_animation, timer, last_seen = enemy
                r = jax.lax.cond(
                    enemy_type != self.consts.ENEMY_NONE,
                    lambda _: jax.lax.switch(
                        enemy_type,
                        [
                            lambda: r,  # ENEMY_NONE, nichts rendern
                            lambda: self._render_character(
                                r,
                                self.SPRITE_BURWOR,
                                EntityPosition(x=x, y=y, direction=direction, width=self.consts.ENEMY_SIZE[0],
                                               height=self.consts.ENEMY_SIZE[1]),
                                death_animation=death_animation
                            ),
                            lambda: jax.lax.cond(
                                jnp.logical_or(last_seen < self.consts.INVISIBILITY_TIMER_GARWOR, death_animation > 0),
                                lambda _: self._render_character(
                                    r,
                                    self.SPRITE_GARWOR,
                                    EntityPosition(x=x, y=y, direction=direction, width=self.consts.ENEMY_SIZE[0],
                                                   height=self.consts.ENEMY_SIZE[1]),
                                    death_animation=death_animation
                                ),
                                lambda _: r,
                                operand=None
                            ),
                            lambda: jax.lax.cond(
                                jnp.logical_or(last_seen < self.consts.INVISIBILITY_TIMER_THORWOR, death_animation > 0),
                                lambda _: self._render_character(
                                    r,
                                    self.SPRITE_THORWOR,
                                    EntityPosition(x=x, y=y, direction=direction, width=self.consts.ENEMY_SIZE[0],
                                                   height=self.consts.ENEMY_SIZE[1]),
                                    death_animation=death_animation
                                ),
                                lambda _: r,
                                operand=None
                            ),
                            lambda: self._render_character(
                                r,
                                self.SPRITE_WORLUK,
                                EntityPosition(x=x, y=y, direction=direction, width=self.consts.ENEMY_SIZE[0],
                                               height=self.consts.ENEMY_SIZE[1]),
                                death_animation=death_animation
                            ),
                            lambda: self._render_character(
                                r,
                                self.SPRITE_WIZARD,
                                EntityPosition(x=x, y=y, direction=direction, width=self.consts.ENEMY_SIZE[0],
                                               height=self.consts.ENEMY_SIZE[1]),
                                death_animation=death_animation
                            ),
                        ]
                    ),
                    lambda _: r,
                    operand=None
                )
                return r, None

            raster_final, _ = jax.lax.scan(
                f=body,
                init=raster,
                xs=state.enemies
            )
            return raster_final

        new_raster = _render_enemies(self, raster=raster, state=state)
        return new_raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_player_bullet(self, raster, state: WizardOfWorState):
        new_raster = jax.lax.cond(
            state.bullet.x >= 0,  # Check if the bullet is active (x >= 0)
            lambda _: jr.render_at(
                raster=raster,
                sprite_frame=jr.get_sprite_frame(self.SPRITE_BULLET, 0),
                x=self.consts.GAME_AREA_OFFSET[0] + state.bullet.x,
                y=self.consts.GAME_AREA_OFFSET[1] + state.bullet.y
            ),
            lambda _: raster,  # If the bullet is not active, return the raster unchanged
            operand=None
        )
        return new_raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemy_bullet(self, raster, state: WizardOfWorState):
        enemy_type = jax.lax.cond(
            state.idx_enemy_bullet_shot_by >= 0,  # Check if the index is valid
            lambda: state.enemies[state.idx_enemy_bullet_shot_by, 3],  # Get the enemy type
            lambda: self.consts.ENEMY_NONE  # Default to NONE if invalid
        )
        new_raster = jax.lax.cond(
            state.enemy_bullet.x >= 0,  # Check if the bullet is active (x >= 0)
            lambda _: jr.render_at(
                raster=raster,
                sprite_frame=jr.get_sprite_frame(self.SPRITE_ENEMY_BULLET, enemy_type),
                x=self.consts.GAME_AREA_OFFSET[0] + state.enemy_bullet.x,
                y=self.consts.GAME_AREA_OFFSET[1] + state.enemy_bullet.y
            ),
            lambda _: raster,  # If the bullet is not active, return the raster unchanged
            operand=None
        )
        return new_raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_player(self, raster, state: WizardOfWorState):
        new_raster = self._render_character(raster, self.SPRITE_PLAYER, state.player,
                                            death_animation=state.player_death_animation)
        return new_raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_score(self, raster, state: WizardOfWorState):
        score_digits = jr.int_to_digits(state.score, max_digits=5)

        new_raster = jr.render_label_selective(raster=raster, x=self.consts.SCORE_OFFSET[0],
                                               y=self.consts.SCORE_OFFSET[1],
                                               all_digits=score_digits, char_sprites=self.SCORE_DIGIT_SPRITES,
                                               start_index=0, num_to_render=5,
                                               spacing=self.consts.SCORE_DIGIT_SPACING)
        return new_raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lives(self, raster, state: WizardOfWorState):
        new_raster = raster

        def render_life(carry, i):
            r = carry
            r = jax.lax.cond(
                (state.lives - 1) > i,
                lambda _: self._render_character(
                    raster=r,
                    sprite=self.SPRITE_PLAYER,
                    entity=EntityPosition(
                        x=self.consts.LIVES_OFFSET[0] - (i * (self.consts.PLAYER_SIZE[0] + self.consts.LIVES_GAP)),
                        y=self.consts.LIVES_OFFSET[1],
                        width=self.consts.PLAYER_SIZE[0],
                        height=self.consts.PLAYER_SIZE[1],
                        direction=self.consts.LEFT
                    ),
                    death_animation=0),
                lambda _: r,
                operand=None
            )
            return r, None

        indices = jnp.arange(start=0, stop=self.consts.MAX_LIVES, dtype=jnp.int32)
        new_raster, _ = jax.lax.scan(render_life, new_raster, indices)
        return new_raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_character(self, raster, sprite, entity: EntityPosition, death_animation):
        """
        Renders a character sprite at the specified position and direction.
        :param raster: The raster to render on.
        :param sprite: The sprite to render.
        :param entity: The entity to render, containing x, y,width,height and direction.
        :return: The raster with the rendered character.
        """
        direction = entity.direction

        def render_death_animation(_):
            frame_index = frame_index = jax.lax.cond(
                death_animation < self.consts.DEATH_ANIMATION_STEPS[0],
                lambda _: 4,
                lambda _: jax.lax.cond(
                    death_animation > self.consts.DEATH_ANIMATION_STEPS[1],
                    lambda _: 6,
                    lambda _: 5,
                    operand=None
                ),
                operand=None
            )
            sprite_frame = jr.get_sprite_frame(sprite, frame_index)
            return jr.render_at(
                raster=raster,
                sprite_frame=sprite_frame,
                x=self.consts.GAME_AREA_OFFSET[0] + entity.x,
                y=self.consts.GAME_AREA_OFFSET[1] + entity.y
            )

        def render_normal(_):
            frame_offset = ((entity.x + entity.y + 1) // 2) % 2
            # if the y position is above 60 frame offset is 1. THIS IS A SPECIAL CASE FOR LIVES RENDERING
            frame_offset = jax.lax.cond(
                entity.y >= 60,
                lambda _: 1,
                lambda _: frame_offset,
                operand=None
            )
            frame_index = jax.lax.cond(
                (direction == self.consts.LEFT) | (direction == self.consts.RIGHT),
                lambda _: frame_offset,
                lambda _: 2 + frame_offset,
                operand=None
            )
            sprite_frame = jr.get_sprite_frame(sprite, frame_index)
            # Spezialfall: Worluk-Sprite nicht flippen
            is_worluk = jnp.all(sprite == self.SPRITE_WORLUK)
            return jax.lax.cond(
                is_worluk,
                lambda _: jr.render_at(
                    raster=raster,
                    sprite_frame=sprite_frame,
                    x=self.consts.GAME_AREA_OFFSET[0] + entity.x,
                    y=self.consts.GAME_AREA_OFFSET[1] + entity.y
                ),
                lambda _: jax.lax.cond(
                    direction == self.consts.RIGHT,
                    lambda _: jr.render_at(
                        raster=raster,
                        sprite_frame=sprite_frame,
                        x=self.consts.GAME_AREA_OFFSET[0] + entity.x,
                        y=self.consts.GAME_AREA_OFFSET[1] + entity.y,
                        flip_horizontal=True
                    ),
                    lambda _: jax.lax.cond(
                        direction == self.consts.UP,
                        lambda _: jr.render_at(
                            raster=raster,
                            sprite_frame=sprite_frame,
                            x=self.consts.GAME_AREA_OFFSET[0] + entity.x,
                            y=self.consts.GAME_AREA_OFFSET[1] + entity.y,
                            flip_vertical=True
                        ),
                        lambda _: jax.lax.cond(
                            direction == self.consts.NONE,
                            lambda _: raster,  # Nichts rendern, wenn Richtung NONE
                            lambda _: jr.render_at(
                                raster=raster,
                                sprite_frame=sprite_frame,
                                x=self.consts.GAME_AREA_OFFSET[0] + entity.x,
                                y=self.consts.GAME_AREA_OFFSET[1] + entity.y
                            ),
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )

        return jax.lax.cond(
            death_animation > 0,
            render_death_animation,
            render_normal,
            operand=None
        )
