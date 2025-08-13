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


class WizardOfWorConstants(NamedTuple):
    WINDOW_WIDTH: int = 160
    WINDOW_HEIGHT: int = 210

    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER_COLOR: Tuple[int, int, int] = (255, 255, 0)
    ENEMY_COLOR: Tuple[int, int, int] = (255, 0, 0)
    BULLET_COLOR: Tuple[int, int, int] = (255, 255, 255)
    WALL_COLOR: Tuple[int, int, int] = (0, 0, 255)

    # Sprite sizes (Platzhalter)
    PLAYER_SIZE: Tuple[int, int] = (8, 8)
    ENEMY_SIZE: Tuple[int, int] = (8, 8)
    BULLET_SIZE: Tuple[int, int] = (2, 2)
    TILE_SIZE: Tuple[int, int] = (8, 8)
    WALL_THICKNESS: int = 2

    # Richtungen
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

    MAX_ENEMIES: int = 10  # Maximum number of enemies on the game board
    MAX_LEVEL: int = 1

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

    LEVEL_1_ENEMY_POSITIONS = jnp.concatenate([
        jnp.array([[0, 0, RIGHT, ENEMY_BURWOR]], dtype=jnp.int32),
        jnp.array([[40, 20, LEFT, ENEMY_BURWOR]], dtype=jnp.int32),
        jnp.array([[10, 40, UP, ENEMY_BURWOR]], dtype=jnp.int32),
        jnp.array([[60, 30, DOWN, ENEMY_BURWOR]], dtype=jnp.int32),
        jnp.tile(jnp.array([[0, 0, 0, 0]], dtype=jnp.int32), (MAX_ENEMIES - 4, 1))
    ], axis=0)

    NO_ENEMY_POSITIONS = jnp.zeros((MAX_ENEMIES, 4), dtype=jnp.int32)  # No enemies


    PLAYER_SPAWN_POSITION: Tuple[int, int,int] = (100,50,LEFT)  # Startposition der Spielfigur

    STEP_SIZE: int = 1  # Step size for player movement

    BOARD_POSITION: Tuple[int, int] = (16, 64)
    GAME_AREA_OFFSET: Tuple[int, int] = (
    BOARD_POSITION[0] + WALL_THICKNESS + TILE_SIZE[0], BOARD_POSITION[1] + WALL_THICKNESS)

    # IMPORTANT: About the coordinates
    # The board goes from 0,0 (top-left) to 110,60 (bottom-right)

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

    @staticmethod
    def get_enemy_positions_for_level(level: int) -> jnp.ndarray:
        """Placeholder: Returns the enemy positions for the specified level.
        :param level: The level for which the enemy positions should be retrieved.
        :return: An array with the enemy positions.
        """
        return jax.lax.cond(
            level == 1,
            lambda _: WizardOfWorConstants.LEVEL_1_ENEMY_POSITIONS,
            lambda _: WizardOfWorConstants.NO_ENEMY_POSITIONS,
            operand=None
        )

class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    direction: int  # Richtung aus UP, DOWN, LEFT, RIGHT


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
    enemies: chex.Array # Array of EntityPosition with length WizardOfWorConstants.MAX_ENEMIES
    gameboard: int
    bullet: EntityPosition
    score: chex.Array
    lives: int
    doubled: bool  # Flag to indicate if the player has the double score power-up. This is only relevant for WORLUK and WIZARD enemies.
    animation_counter: int # Counter for animations. This may not be needed since animation are tied to board position.
    rng_key: chex.PRNGKey  # Random key for JAX operations
    level: int  # The current level of the game, used for level progression
    game_over: bool


def update_state(state: WizardOfWorState, player: EntityPosition = None, enemies: chex.Array = None,
                 gameboard: int = None, bullet: EntityPosition = None, score: chex.Array = None,
                 lives: int = None, doubled: bool = None,animation_counter: int = None,rng_key: chex.PRNGKey = None,
                 level: int = None, game_over: bool = None) -> WizardOfWorState:
    """
    Updates the state of the game. Only this method should be used to mutate the State object.
    Parameters not passed will be taken from the current state.
    :param state: The current state of the game.
    :param player: New position of the player character.
    :param enemies: New positions of the enemies.
    :param gameboard: New gameboard.
    :param bullet: New position of the shot.
    :param score: New score.
    :param lives: New number of lives.
    :param doubled: Flag indicating whether the player has the double score power-up.
    :param animation_counter: Counter for animations, e.g. player walking animation.
    :param rng_key: Random key for JAX operations.
    :param level: The current level of the game.
    :return: A new state of the game with the updated values.
    """
    return WizardOfWorState(
        player=player if player is not None else state.player,
        enemies=enemies if enemies is not None else state.enemies,
        gameboard=gameboard if gameboard is not None else state.gameboard,
        bullet=bullet if bullet is not None else state.bullet,
        score=score if score is not None else state.score,
        lives=lives if lives is not None else state.lives,
        doubled=doubled if doubled is not None else state.doubled,
        animation_counter=animation_counter if animation_counter is not None else state.animation_counter,
        rng_key=rng_key if rng_key is not None else state.rng_key,
        level=level if level is not None else state.level,
        game_over=game_over if game_over is not None else state.game_over
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
                x=jnp.array(self.consts.PLAYER_SPAWN_POSITION[0]),
                y=jnp.array(self.consts.PLAYER_SPAWN_POSITION[1]),
                direction=self.consts.PLAYER_SPAWN_POSITION[2]
            ),
            enemies= jnp.zeros(
                (self.consts.MAX_ENEMIES, 4),  # [x, y, direction, type]
                dtype=jnp.int32
            ),
            gameboard=1,
            bullet=EntityPosition(
                x=jnp.array(-1),
                y=jnp.array(-1),
                direction=self.consts.UP
            ),
            score=jnp.array(0),
            lives=3,
            doubled=False,
            animation_counter=0,
            rng_key=jax.random.PRNGKey(0),  # Initialisiere den RNG
            level=0,
            game_over=False
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: WizardOfWorState, action: chex.Array) -> Tuple[
        WizardOfWorObservation, WizardOfWorState, chex.Array, chex.Array, WizardOfWorInfo]:
        new_state = previous_state = state
        new_state = self._step_level_change(state=new_state)
        new_state = self._step_respawn(state=new_state, action=action)
        new_state = self._step_player_movement(state=new_state, action=action)
        new_state = self._step_bullet_movement(state=new_state)
        new_state = self._step_enemy_movement(state=new_state)
        new_state = self._step_collision_detection(state=new_state)
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

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: WizardOfWorState, state: WizardOfWorState) -> chex.Array:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: WizardOfWorState) -> WizardOfWorObservation:
        player_entity = EntityPosition(
            x=state.player.x,
            y=state.player.y,
            direction=state.player.direction
        )
        enemies = state.enemies  # Placeholder for enemy positions
        bullet_entity = EntityPosition(
            x=state.bullet.x,
            y=state.bullet.y,
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
    def _step_level_change(self, state):
        # Wenn alle Gegner besiegt sind, Level erhöhen und Gegner respawnen. Falls das nächste Level größer als MAX_LEVEL ist, Spiel beenden.
        return jax.lax.cond(
            jnp.all(state.enemies[:, 3] == self.consts.ENEMY_NONE),
            lambda _: jax.lax.cond(
                (state.level + 1) > self.consts.MAX_LEVEL,
                lambda _: update_state(
                    state=state,
                    game_over=True
                ),
                lambda _: update_state(
                    state=state,
                    enemies=self.consts.get_enemy_positions_for_level(level=state.level + 1),
                    level=state.level + 1,
                ),
                operand=None
            ),
            lambda _: state,
            operand=None
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step_respawn(self, state, action):
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _step_player_movement(self, state, action):
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _step_bullet_movement(self, state):
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _step_enemy_movement(self, state):
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _step_collision_detection(self, state):
        return state


class WizardOfWorRenderer(JAXGameRenderer):
    def __init__(self, consts: WizardOfWorConstants = None):
        super().__init__()
        self.consts = consts or WizardOfWorConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_BURWOR,
            self.SPRITE_BULLET,
            self.SCORE_DIGIT_SPRITES,
            self.SPRITE_WALL_HORIZONTAL,
            self.SPRITE_WALL_VERTICAL,
            self.SPRITE_RADAR_BLIP,
        ) = self.load_sprites()

    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/background.npy"))
        player0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/player_0.npy"))
        player1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/player_1.npy"))
        player2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/player_2.npy"))
        player3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/player/player_3.npy"))
        burwor0 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/burwor_0.npy"))
        burwor1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/burwor_1.npy"))
        burwor2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/burwor_2.npy"))
        burwor3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/enemies/burwor/burwor_3.npy"))
        bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/bullet.npy"))
        wall_horizontal = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/wall_horizontal.npy"))
        wall_vertical = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/wall_vertical.npy"))
        radar_blip_empty = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_empty.npy"))
        radar_blip_burwor = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wizardofwor/radar/radar_burwor.npy"))

        SPRITE_PLAYER = jnp.stack([player0, player1, player2, player3], axis=0)
        SPRITE_BURWOR = jnp.stack([burwor0, burwor1, burwor2, burwor3], axis=0)
        SPRITE_RADAR_BLIP = jnp.stack([radar_blip_empty, radar_blip_burwor], axis=0)

        SPRITE_BG = jnp.expand_dims(bg, axis=0)
        SPRITE_BULLET = jnp.expand_dims(bullet, axis=0)
        SPRITE_WALL_HORIZONTAL = jnp.expand_dims(wall_horizontal, axis=0)
        SPRITE_WALL_VERTICAL = jnp.expand_dims(wall_vertical, axis=0)

        SCORE_DIGIT_SPRITES = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/pong/player_score_{}.npy"),
            num_chars=10,
        )

        return (
            SPRITE_BG,
            SPRITE_PLAYER,
            SPRITE_BURWOR,
            SPRITE_BULLET,
            SCORE_DIGIT_SPRITES,
            SPRITE_WALL_HORIZONTAL,
            SPRITE_WALL_VERTICAL,
            SPRITE_RADAR_BLIP,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: WizardOfWorState):
        # Raster initialisieren
        raster = jr.create_initial_frame(width=self.consts.WINDOW_WIDTH, height=self.consts.WINDOW_HEIGHT)
        raster = self._render_gameboard(raster=raster, state=state)
        raster = self._render_radar(raster=raster, state=state)
        raster = self._render_enemies(raster=raster, state=state)
        raster = self._render_bullet(raster=raster, state=state)
        raster = self._render_player(raster=raster, state=state)
        raster = self._render_score(raster=raster, state=state)
        raster = self._render_lives(raster=raster, state=state)
        return raster

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
                # grid_vals is shape [H, W] or [H, W, 3]; produce x,y indices matching it
                H, W = grid_vals.shape[:2]
                xs = jnp.repeat(jnp.arange(H)[:, None], W, axis=1)
                ys = jnp.repeat(jnp.arange(W)[None, :], H, axis=0)

                # flatten to iterate
                xs_f = xs.ravel()
                ys_f = ys.ravel()
                vals_f = grid_vals.reshape(-1, *grid_vals.shape[2:])  # [-1] or [-1,3]

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
                # grid_vals is shape [H, W] or [H, W, 3]; produce x,y indices matching it
                H, W = grid_vals.shape[:2]
                xs = jnp.repeat(jnp.arange(H)[:, None], W, axis=1)
                ys = jnp.repeat(jnp.arange(W)[None, :], H, axis=0)

                # flatten to iterate
                xs_f = xs.ravel()
                ys_f = ys.ravel()
                vals_f = grid_vals.reshape(-1, *grid_vals.shape[2:])

                # [-1] or [-1,3]
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

        new_raster = _render_gameboard_background(raster=raster)
        new_raster = _render_gameboard_walls(raster=new_raster, state=state)
        return new_raster

    def _render_radar(self, raster, state: WizardOfWorState):
        return raster

    def _render_enemies(self, raster, state: WizardOfWorState):
        def _render_enemies(self, raster, state: WizardOfWorState):
            def body(carry, enemy):
                r = carry
                x, y, direction, enemy_type = enemy
                r = jax.lax.cond(
                    enemy_type != self.consts.ENEMY_NONE,
                    lambda _: self._render_character(
                        r,
                        self.SPRITE_BURWOR, # Placeholder for enemy sprite, can be extended for other enemy types
                        EntityPosition(x=x, y=y, direction=direction)
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

    def _render_bullet(self, raster, state: WizardOfWorState):
        return raster

    def _render_player(self, raster, state: WizardOfWorState):
        new_raster = self._render_character(raster,self.SPRITE_PLAYER,state.player)
        return new_raster

    def _render_score(self, raster, state: WizardOfWorState):
        return raster

    def _render_lives(self, raster, state: WizardOfWorState):
        return raster

    def _render_character(self, raster, sprite, entity: EntityPosition):
        """
        Renders a character sprite at the specified position and direction.
        :param raster: The raster to render on.
        :param sprite: The sprite to render.
        :param entity: The entity to render, containing x, y, and direction.
        :return: The raster with the rendered character.
        """
        direction = entity.direction
        frame_offset = (entity.x+entity.y) % 2
        frame_index = jax.lax.cond(
            (direction == self.consts.LEFT) | (direction == self.consts.RIGHT),
            lambda _: frame_offset,
            lambda _: 2 + frame_offset,
            operand=None
        )
        sprite_frame = jr.get_sprite_frame(sprite,frame_index)
        return jax.lax.cond(
            direction == self.consts.RIGHT,
            lambda _: jr.render_at(
                raster=raster,
                sprite_frame=sprite_frame,
                x=self.consts.GAME_AREA_OFFSET[0]+entity.x,
                y=self.consts.GAME_AREA_OFFSET[1]+entity.y,
                flip_horizontal=True
            ),
            lambda _: jax.lax.cond(
                direction == self.consts.UP,
                lambda _: jr.render_at(
                    raster=raster,
                    sprite_frame=sprite_frame,
                    x=self.consts.GAME_AREA_OFFSET[0]+entity.x,
                    y=self.consts.GAME_AREA_OFFSET[1]+entity.y,
                    flip_vertical=True
                ),
                lambda _: jr.render_at(
                    raster=raster,
                    sprite_frame=sprite_frame,
                    x=self.consts.GAME_AREA_OFFSET[0]+entity.x,
                    y=self.consts.GAME_AREA_OFFSET[1]+entity.y
                ),
                operand=None
            ),
            operand=None
        )
