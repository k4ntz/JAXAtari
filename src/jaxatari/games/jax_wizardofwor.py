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


# Wizard of Wor: Spawn Behavior by Level
# Core Spawn Rules
#
#     Each dungeon starts with 6 Burwors on the field, and there are never more than 6 enemies on-screen at once.
#
# Killing Burwors progressively “promotes” them to Garwors depending on the dungeon number: in dungeon 1 only the last Burwor becomes a Garwor; in dungeon 2 the last two; continuing until dungeon 6+, where every Burwor you kill is replaced by a Garwor.
#
# On every dungeon, each Garwor killed immediately spawns a Thorwor; killing a Thorwor does not spawn further replacements until the last one is destroyed.
#
# From dungeon 2 onward, once the last Thorwor is killed, a Worluk appears and tries to escape through a side exit; the level ends when it escapes or is shot, and shooting it makes the next dungeon a Double Score Dungeon.
# Level-by-Level Promotion Pattern
#
#     Dungeon 1: 6 Burwors begin on the map; when only 1 Burwor remains, killing it spawns 1 Garwor; each Garwor killed spawns 1 Thorwor; no Worluk after Thorwors on dungeon 1.
#
# Dungeon 2: The last 2 Burwors killed are replaced by Garwors; each Garwor killed spawns a Thorwor; after the last Thorwor, the Worluk appears.
#
# Dungeon 3: The last 3 Burwors killed are replaced by Garwors; each Garwor killed spawns a Thorwor; Worluk appears after the last Thorwor.
#
# Dungeon 4: The last 4 Burwors killed are replaced by Garwors; each Garwor killed spawns a Thorwor; Worluk appears after the last Thorwor.
#
# Dungeon 5: The last 5 Burwors killed are replaced by Garwors; each Garwor killed spawns a Thorwor; Worluk appears after the last Thorwor.
#
# Dungeon 6 and beyond: All 6 Burwors, when killed, are replaced by Garwors; each Garwor killed spawns a Thorwor; Worluk appears after the last Thorwor.
# Special Spawns and Set Dungeons
#
#     The Wizard of Wor may appear starting from dungeon 2, with a low random chance; he teleports in after or around the Worluk phase and vanishes after killing a player or being shot.
#
# “The Arena” is always the 4th dungeon, and “The Pit” occurs at dungeons 13, 19, 25, 31, and then every 6th dungeon thereafter; The Pit has no inner walls and follows the same enemy promotion/spawn rules but on an open map.
# Practical Flow Within a Dungeon
#
#     Start: 6 Burwors active. Depending on dungeon number, some of the final Burwors killed will be replaced by Garwors immediately upon death.
#
# Mid-phase: Each Garwor killed spawns a Thorwor in place; this continues until all Garwors have turned into Thorwors and are then cleared.
#
# End-phase: From dungeon 2 onward, the Worluk spawns after the last Thorwor is destroyed and races for a side exit; shooting it ends the level and grants Double Score on the next dungeon; letting it escape also ends the level without the bonus.

class EntityPosition(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    direction: int  # Richtung aus UP, DOWN, LEFT, RIGHT


class WizardOfWorConstants(NamedTuple):
    WINDOW_WIDTH: int = 160
    WINDOW_HEIGHT: int = 210

    DEATH_ANIMATION_STEPS = [10, 20]

    # Sprite sizes (Platzhalter)
    PLAYER_SIZE: Tuple[int, int] = (8, 8)
    ENEMY_SIZE: Tuple[int, int] = (8, 8)
    BULLET_SIZE: Tuple[int, int] = (2, 2)
    TILE_SIZE: Tuple[int, int] = (8, 8)
    RADAR_BLIP_SIZE: Tuple[int, int] = (2, 2)
    WALL_THICKNESS: int = 2
    RADAR_BLIP_GAP: int = 0

    # Richtungen
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

    MAX_ENEMIES: int = 6  # Maximum number of enemies on the game board
    MAX_LEVEL: int = 1
    MAX_LIVES: int = 3

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

    TELEPORTER_LEFT_POSITION: Tuple[int, int] = (-2, 20)  # Position des linken Teleporters
    TELEPORTER_RIGHT_POSITION: Tuple[int, int] = (108, 20)  # Position des rechten Teleporters

    NO_ENEMY_POSITIONS = jnp.zeros((MAX_ENEMIES, 5), dtype=jnp.int32)  # No enemies

    PLAYER_SPAWN_POSITION: Tuple[int, int, int] = (100, 50, LEFT)  # Startposition der Spielfigur

    STEP_SIZE: int = 1  # Step size for player movement

    BOARD_POSITION: Tuple[int, int] = (16, 64)
    GAME_AREA_OFFSET: Tuple[int, int] = (
        BOARD_POSITION[0] + WALL_THICKNESS + TILE_SIZE[0], BOARD_POSITION[1] + WALL_THICKNESS)
    LIVES_OFFSET: Tuple[int, int] = (100, 60)  # Offset for lives display
    LIVES_GAP: int = 5  # Gap between lives icons
    RADAR_OFFSET: Tuple[int, int] = (BOARD_POSITION[0] + 53, BOARD_POSITION[1] + 72)  # Offset for radar display

    # IMPORTANT: About the coordinates
    # The board goes from 0,0 (top-left) to 110,60 (bottom-right)
    BOARD_SIZE: Tuple[int, int] = (110, 60)  # Size of the game board in tiles

    SCORE_DIGIT_SPACING: int = 8
    SCORE_OFFSET: Tuple[int, int] = (BOARD_POSITION[0] + 80, BOARD_POSITION[1] - 16)  # Offset for score display

    @partial(jax.jit, static_argnums=(0,))
    def _get_wall_position(self, x: int, y: int, horizontal: bool) -> EntityPosition:
        """Placeholder: Returns the position of a wall based on its coordinates.
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
        """    Placeholder: Returns the walls for the specified gameboard.
        :param gameboard: The gameboard for which the walls should be retrieved.
        :return: A tuple with the horizontal and vertical walls.
        """
        return jax.lax.cond(
            gameboard == 1,
            lambda _: (
                WizardOfWorConstants.GAMEBOARD_1_WALLS_HORIZONTAL, WizardOfWorConstants.GAMEBOARD_1_WALLS_VERTICAL),
            lambda _: (jnp.zeros((5, 11), dtype=jnp.int32), jnp.zeros((6, 10), dtype=jnp.int32)),
            operand=None
        )  # Hier können weitere Gameboards hinzugefügt werden


class WizardOfWorObservation(NamedTuple):
    player: EntityPosition
    enemies: chex.Array
    bullet: EntityPosition
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
                 score: chex.Array = None,
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
        state = WizardOfWorState(
            player=EntityPosition(
                x=self.consts.PLAYER_SPAWN_POSITION[0],
                y=self.consts.PLAYER_SPAWN_POSITION[1],
                width=self.consts.PLAYER_SIZE[0],
                height=self.consts.PLAYER_SIZE[1],
                direction=self.consts.PLAYER_SPAWN_POSITION[2]
            ),
            player_death_animation=self.consts.DEATH_ANIMATION_STEPS[1]+1,
            enemies=jnp.zeros(
                (self.consts.MAX_ENEMIES, 5),  # [x, y, direction, type]
                dtype=jnp.int32
            ),
            gameboard=1,
            bullet=EntityPosition(
                x=87,
                y=54,
                width=self.consts.BULLET_SIZE[0],
                height=self.consts.BULLET_SIZE[1],
                direction=self.consts.UP
            ),
            enemy_bullet=EntityPosition(
                x=80,
                y=54,
                width=self.consts.BULLET_SIZE[0],
                height=self.consts.BULLET_SIZE[1],
                direction=self.consts.UP
            ),
            score=jnp.array(0),
            lives=3,
            doubled=False,
            frame_counter=0,
            rng_key=jax.random.PRNGKey(1),  # Initialisiere den RNG
            level=0,
            game_over=False,
            teleporter=False
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: WizardOfWorState, action: chex.Array) -> Tuple[
        WizardOfWorObservation, WizardOfWorState, chex.Array, chex.Array, WizardOfWorInfo]:
        new_state = previous_state = state
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
        done = self._get_done(state=new_state)
        env_reward = self._get_reward(previous_state=previous_state, state=new_state)
        all_rewards = self._get_all_reward(previous_state=previous_state, state=new_state)
        info = self._get_info(state=new_state, all_rewards=all_rewards)
        observation = self._get_observation(state=new_state)

        return observation, new_state, env_reward, done, info

    def render(self, state: WizardOfWorState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)  # Platzhalter

    @partial(jax.jit, static_argnums=(0,))
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WINDOW_WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.WINDOW_HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WINDOW_WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.WINDOW_HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "enemies": spaces.Box(low=0, high=self.consts.WINDOW_WIDTH, shape=(None, 4), dtype=jnp.int32),
            "bullets": spaces.Box(low=0, high=self.consts.WINDOW_WIDTH, shape=(None, 4), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=10, shape=(), dtype=jnp.int32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: WizardOfWorState) -> chex.Array:
        return jnp.array(False)  # später implementieren

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        return jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: WizardOfWorState, all_rewards: chex.Array = None) -> WizardOfWorInfo:
        return WizardOfWorInfo(all_rewards=all_rewards)

    def _get_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState) -> chex.Array:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: WizardOfWorState) -> WizardOfWorObservation:
        player_entity = EntityPosition(
            x=state.player.x,
            y=state.player.y,
            width=state.player.width,
            height=state.player.height,
            direction=state.player.direction
        )
        enemies = state.enemies  # Placeholder for enemy positions
        bullet_entity = EntityPosition(
            x=state.bullet.x,
            y=state.bullet.y,
            width=state.bullet.width,
            height=state.bullet.height,
            direction=state.bullet.direction
        )
        return WizardOfWorObservation(
            player=player_entity,
            enemies=enemies,
            bullet=bullet_entity,
            score=state.score,
            lives=jnp.array(state.lives)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _is_enemy_dead(self, enemy) -> bool:
        x, y, direction, type, death_animation = enemy
        return (death_animation > self.consts.DEATH_ANIMATION_STEPS[1]) | (type == self.consts.ENEMY_NONE)

    @partial(jax.jit, static_argnums=(0,))
    def _get_gameboard_for_level(self, level: int) -> int:
        return jax.lax.cond(
            level == 1,
            lambda: 1,
            lambda: 1
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_start_enemies(self, rng_key) -> chex.Array:
        # This is always MAX_ENEMIES BURWOR on random tiles
        # return an array of 6 burwors at random positions.
        # x can be i * (TILE_SIZE[0] + WALL_THICKNESS)
        # y can be j * (TILE_SIZE[1] + WALL_THICKNESS)
        # so generate the enemies with x from 0 to 10 and y from 0 to 5 and then after multiply it.
        # this ensures the random generater doesnt spawn them between tiles
        # direction is randomly chosen from UP, DOWN, LEFT, RIGHT
        def _generate_single_enemy(rng_key) -> chex.Array:
            x = jax.random.randint(rng_key, shape=(), minval=0, maxval=11) * (
                        self.consts.TILE_SIZE[0] + self.consts.WALL_THICKNESS)
            y = jax.random.randint(rng_key, shape=(), minval=0, maxval=6) * (
                        self.consts.TILE_SIZE[1] + self.consts.WALL_THICKNESS)
            direction = jax.random.choice(rng_key, jnp.array(
                [self.consts.UP, self.consts.DOWN, self.consts.LEFT, self.consts.RIGHT]))
            return jnp.array([x, y, direction, self.consts.ENEMY_BURWOR, 0])

        return jax.vmap(_generate_single_enemy)(jax.random.split(rng_key, self.consts.MAX_ENEMIES))

    @partial(jax.jit, static_argnums=(0,))
    def _step_level_change(self, state):
        # We only do level 1 for this MVP
        # if no enemies are left, we increase the level
        # dont assume that this is  if state.enemies == self.consts.NO_ENEMY_POSITIONS.
        # go through the enemies and check if they are all dead
        all_enemies_dead = jnp.all(jax.vmap(self._is_enemy_dead)(state.enemies))
        new_rng_key, rng_key = jax.random.split(state.rng_key)
        state = update_state(state=state, rng_key=new_rng_key)
        return jax.lax.cond(
            all_enemies_dead,
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
                lives=state.lives + 1,
                enemies=self._get_start_enemies(rng_key)),
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
                    x=checked_new_position.x + checked_new_position.width // 2 - self.consts.BULLET_SIZE[0] // 2,
                    y=checked_new_position.y + checked_new_position.height // 2 - self.consts.BULLET_SIZE[1] // 2,
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
                player_death_animation= self.consts.DEATH_ANIMATION_STEPS[1] + 1,
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
    def _positions_equal(self, pos1: EntityPosition, pos2: EntityPosition) -> bool:
        """Check if two positions are equal."""
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
        """Check if the position is valid.
        :param position: The position to check.
        :return: True if the position is valid, False otherwise.
        """
        # check both walls and boundaries using _check_boundaries and _check_walls
        boundary_position = self._check_boundaries(old_position=old_position, new_position=new_position)
        return self._check_walls(state, old_position=old_position, new_position=boundary_position)

    @partial(jax.jit, static_argnums=(0,))
    def _check_boundaries(self, old_position: EntityPosition, new_position: EntityPosition) -> EntityPosition:
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
        """Prüft, ob sich zwei Bounding-Boxen überlappen (Kantenberührung zählt nicht als Kollision)."""
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
    def _step_bullet_movement(self, state):
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
                x=-1,
                y=-1,
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
        # map over all enemies and update their positions based on their direction

        def move_alive_enemy(enemy: chex.Array) -> chex.Array:
            x, y, direction, enemy_type, death_animation = enemy

            is_burwor = enemy_type == self.consts.ENEMY_BURWOR
            is_garwor = enemy_type == self.consts.ENEMY_GARWOR
            is_thorwor = enemy_type == self.consts.ENEMY_THORWOR
            is_worluk = enemy_type == self.consts.ENEMY_WORLUK
            is_wizard = enemy_type == self.consts.ENEMY_WIZARD

            return enemy

        def move_enemy(enemy: chex.Array) -> chex.Array:
            x, y, direction, enemy_type, death_animation = enemy

            is_none = enemy_type == self.consts.ENEMY_NONE
            is_dying = death_animation > 0
            dying_enemy = jnp.array([x, y, direction, enemy_type, death_animation + 1])

            result = jax.lax.cond(
                is_none,
                lambda _: enemy,
                lambda _: jax.lax.cond(
                    is_dying,
                    lambda _: dying_enemy,
                    lambda _: move_alive_enemy(enemy),
                    operand=None
                ),
                operand=None
            )
            return result

        new_enemies = jax.vmap(move_enemy)(state.enemies)
        return update_state(
            state=state,
            enemies=new_enemies
        )

    # In this function we handle the enemy level progression and the "cleanup" of dead enemies.
    # when we "clean up" an enemy, we check if:
    # if it is one of the n last burwors on the board. n is the level were in up to 6.
    #   - in this case we spawn a garwor on the board.
    # if it is a garwor, we spawn a thorwor.
    # if it is the last thorwor (so no other burwor, garwor or thorwor is on the board), we spawn a worluk.
    # if it is a worluk, we maybe (50% chance) spawn a wizard.
    # if none of these conditions are met, we just remove the enemy.
    # A enemy is considered dead if its death_animation > DEATH_ANIMATION_STEPS[1] and its type is not NONE.
    # A enemy is always spawned on a tile. so that means x between 0 and 10 and y between 0 and 5; and then multiplied by (TILE_SIZE[0] + WALL_THICKNESS) or (TILE_SIZE[1] + WALL_THICKNESS).

    def _step_enemy_level_progression(self, state):
        """
        Führt die Feind-Progression und das Aufräumen toter Feinde durch.
        Jede Regel ist als innere Funktion implementiert.
        """
        consts = self.consts
        enemies = state.enemies
        level = jnp.minimum(state.level + 1, 6)  # Level für Burwor->Garwor-Promotion, max 6

        def get_random_tile_position(rng_key):
            # Generiert eine zufällige Position auf einem Tile
            x_idx = jax.random.randint(rng_key, shape=(), minval=0, maxval=11)
            y_idx = jax.random.randint(rng_key, shape=(), minval=0, maxval=6)
            x = x_idx * (consts.TILE_SIZE[0] + consts.WALL_THICKNESS)
            y = y_idx * (consts.TILE_SIZE[1] + consts.WALL_THICKNESS)
            direction = jax.random.choice(rng_key, jnp.array([consts.UP, consts.DOWN, consts.LEFT, consts.RIGHT]))
            return jnp.array([x, y, direction, consts.ENEMY_NONE, 0])

        def is_dead(enemy):
            return (enemy[4] > consts.DEATH_ANIMATION_STEPS[1]) & (enemy[3] != consts.ENEMY_NONE)

        def promote_burwor_to_garwor(enemies, idx):
            # Letzte n Burwors werden zu Garwors
            # Wenn weniger als level Burwors übrig sind gebe true zurück
            alive_burwors = jnp.sum(enemies[:, 3] == consts.ENEMY_BURWOR)
            return alive_burwors <= level

        def spawn_garwor(enemy, rng_key):
            # Spawnt einen Garwor an einer zufälligen Position
            pos = get_random_tile_position(rng_key)
            return jnp.array([pos[0], pos[1], enemy[2], consts.ENEMY_GARWOR, 0])

        def spawn_thorwor(enemy, rng_key):
            # Spawnt einen Thorwor an einer zufälligen Position
            pos = get_random_tile_position(rng_key)
            return jnp.array([pos[0], pos[1], enemy[2], consts.ENEMY_THORWOR, 0])

        def spawn_worluk(enemies, rng_key):
            # Spawnt einen Worluk, wenn keine Burwors, Garwors oder Thorwors mehr leben
            alive_types = jnp.array([consts.ENEMY_BURWOR, consts.ENEMY_GARWOR, consts.ENEMY_THORWOR])
            any_alive = jnp.any(jnp.isin(enemies[:, 3], alive_types))
            if not any_alive:
                pos = get_random_tile_position(rng_key)
                return jnp.array([pos[0], pos[1], consts.RIGHT, consts.ENEMY_WORLUK, 0])
            else:
                return None

        def spawn_wizard(rng_key):
            # 50% Chance, einen Wizard zu spawnen
            return jax.random.bernoulli(rng_key, 0.5)

        def cleanup_enemy(enemy, idx, rng_key):
            # Hauptlogik für das Aufräumen und Progression
            def handle_dead():
                return jax.lax.cond(
                    (enemy[3] == consts.ENEMY_BURWOR) & promote_burwor_to_garwor(enemies, idx),
                    lambda: spawn_garwor(enemy, rng_key),
                    lambda: jax.lax.cond(
                        enemy[3] == consts.ENEMY_GARWOR,
                        lambda: spawn_thorwor(enemy, rng_key),
                        lambda: jax.lax.cond(
                            (enemy[3] == consts.ENEMY_THORWOR) & (~jnp.any(jnp.isin(enemies[:, 3], jnp.array(
                                [consts.ENEMY_BURWOR, consts.ENEMY_GARWOR, consts.ENEMY_THORWOR])))),
                            lambda: get_random_tile_position(rng_key).at[2].set(consts.ENEMY_WORLUK),
                            lambda: jax.lax.cond(
                                (enemy[3] == consts.ENEMY_WORLUK) & spawn_wizard(rng_key),
                                lambda: get_random_tile_position(rng_key).at[2].set(consts.ENEMY_WIZARD),
                                lambda: jnp.array([0, 0, 0, consts.ENEMY_NONE, 0])
                            )
                        )
                    )
                )

            return jax.lax.cond(
                is_dead(enemy),
                handle_dead,
                lambda: enemy
            )

        rng_keys = jax.random.split(state.rng_key, consts.MAX_ENEMIES)
        new_enemies = jax.vmap(cleanup_enemy, in_axes=(0, 0, 0))(enemies, jnp.arange(consts.MAX_ENEMIES), rng_keys)
        return update_state(state=state, enemies=new_enemies)

    @partial(jax.jit, static_argnums=(0,))
    def _step_collision_detection(self, state):
        # We have 3 types of collisions:
        # 1. Player vs Enemy
        #    - If a player collides with an enemy, the player dies (death_animation is set to 1)
        # 2. Player vs Enemy Bullet
        #    - If a player collides with an enemy bullet, the player dies (death_animation is set to 1) and
        #      the bullet is removed (direction is set to NONE)
        # 3. Enemy vs Player Bullet
        #    - If an enemy collides with a player bullet, the enemy dies (death_animation is set to 1)
        #      and the bullet is removed (direction is set to NONE)

        def check_player_enemy_collision(player: EntityPosition, enemy: chex.Array) -> bool:
            x, y, direction, enemy_type, death_animation = enemy
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
                bullet.direction == self.consts.NONE,
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
                x=-1,
                y=-1,
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
            x, y, direction, enemy_type, death_animation = enemy

            def handle_collision():
                return jnp.array([x, y, direction, enemy_type, 1])

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
                x=-1,
                y=-1,
                width=new_state.bullet.width,
                height=new_state.bullet.height,
                direction=self.consts.NONE
            ),
            lambda: new_state.bullet
        )

        # Update the state with new enemies and bullet
        new_state = update_state(
            state=new_state,
            enemies=new_enemies,
            bullet=new_bullet
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
        bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/bullet.npy"))
        enemy_bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemy_bullet.npy"))
        wall_horizontal = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/wall_horizontal.npy"))
        wall_vertical = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/wall_vertical.npy"))
        radar_blip_empty = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_empty.npy"))
        radar_blip_burwor = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_burwor.npy"))

        SPRITE_PLAYER = jnp.stack([player0, player1, player2, player3, player_death0, player_death1, player_death2],
                                  axis=0)
        SPRITE_BURWOR = jnp.stack([burwor0, burwor1, burwor2, burwor3, burwor_death0, burwor_death1, burwor_death2],
                                  axis=0)
        SPRITE_RADAR_BLIP = jnp.stack([radar_blip_empty, radar_blip_burwor], axis=0)

        SPRITE_BG = jnp.expand_dims(bg, axis=0)
        SPRITE_BULLET = jnp.expand_dims(bullet, axis=0)
        SPRITE_ENEMY_BULLET = jnp.expand_dims(enemy_bullet, axis=0)
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
            x, y, direction, enemy_type, death_animation = enemy
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
                x, y, direction, enemy_type, death_animation = enemy
                r = jax.lax.cond(
                    enemy_type != self.consts.ENEMY_NONE,
                    lambda _: self._render_character(
                        r,
                        self.SPRITE_BURWOR,  # Placeholder for enemy sprite, can be extended for other enemy types
                        EntityPosition(x=x, y=y, direction=direction, width=self.consts.ENEMY_SIZE[0],
                                       height=self.consts.ENEMY_SIZE[1]),
                        death_animation=death_animation
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
        new_raster = jax.lax.cond(
            state.enemy_bullet.x >= 0,  # Check if the bullet is active (x >= 0)
            lambda _: jr.render_at(
                raster=raster,
                sprite_frame=jr.get_sprite_frame(self.SPRITE_ENEMY_BULLET, 0),
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
            return jax.lax.cond(
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
            )

        return jax.lax.cond(
            death_animation > 0,
            render_death_animation,
            render_normal,
            operand=None
        )
