import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


class AsterixConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    player_width: int = 8
    player_height: int = 8
    num_stages: int = 8
    stage_spacing: int = 16 # ursprünglich 16
    stage_borders: List[int] = None
    top_border: int = 23 # oberer Rand des Spielfelds
    bottom_border: int = 8 * stage_spacing + top_border
    cooldown_frames: int = 8 # Cooldown frames for lane changes
    num_lives: int = 3 # Anzahl der Leben


    stage_borders = [
        top_border, # TOP
        1 * stage_spacing + top_border,  # Stage 1
        2 * stage_spacing + top_border,  # Stage 2
        3 * stage_spacing + top_border,  # Stage 3
        4 * stage_spacing + top_border,  # Stage 4
        5 * stage_spacing + top_border,  # Stage 5
        6 * stage_spacing + top_border,  # Stage 6
        7 * stage_spacing + top_border,  # Stage 7
        8 * stage_spacing + top_border,  # BOTTOM
    ]


class AsterixState(NamedTuple):
    """Represents the current state of the game"""
    player_x: chex.Array
    player_y: chex.Array
    score: chex.Array
    lives: chex.Array
    game_over: chex.Array
    stage_cooldown: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class AsterixObservation(NamedTuple):
    player: EntityPosition
    score: jnp.ndarray


class AsterixInfo(NamedTuple):
    all_rewards: jnp.ndarray


class JaxAsterix(JaxEnvironment[AsterixState, AsterixObservation, AsterixInfo, AsterixConstants]):
    def __init__(self, consts: AsterixConstants = None, reward_funcs: list[callable] = None):
        if consts is None:
            consts = AsterixConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.state = self.reset()
        self.renderer = AsterixRenderer()

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[AsterixObservation, AsterixState]:
        """Initialize a new game state"""
        stage_borders = jnp.array(self.consts.stage_borders, dtype=jnp.int32)
        player_x = self.consts.screen_width // 2
        player_y = (stage_borders[-2] + stage_borders[-1]) // 2

        state = AsterixState(
            player_x =jnp.array(player_x, dtype=jnp.int32),
            player_y=jnp.array(player_y, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32), # Start with 0 points
            lives=jnp.array(self.consts.num_lives, dtype=jnp.int32),  # 3 Leben
            game_over=jnp.array(False, dtype=jnp.bool_),
            stage_cooldown = jnp.array(self.consts.cooldown_frames, dtype=jnp.int32), # Cooldown initial 0
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AsterixState, action: int) -> tuple[
        AsterixObservation, AsterixState, float, bool, AsterixInfo]:
        """Take a step in the game given an action"""
        player_height = self.consts.player_height

        cooldown_frames = self.consts.cooldown_frames  # Cooldown frames for lane changes
        can_switch_stage = state.stage_cooldown <= 0

        stage_borders = jnp.array(self.consts.stage_borders, dtype=jnp.int32)
        num_stage = stage_borders.shape[0]

        stage_diffs = jnp.abs(stage_borders - state.player_y)
        current_stage = jnp.argmin(stage_diffs)

        new_stage = jnp.where(
            can_switch_stage & (action == Action.UP),
            jnp.maximum(current_stage - 1, 0),
            jnp.where(
                can_switch_stage & (action == Action.DOWN),
                jnp.minimum(current_stage + 1, num_stage - 2),
                current_stage
            )
        )
        new_y = ((stage_borders[new_stage] + stage_borders[new_stage + 1]) // 2) - (player_height // 2)

        new_cooldown = jnp.where(
            can_switch_stage & ((action == Action.UP) | (action == Action.DOWN)),
            cooldown_frames,
            jnp.maximum(state.stage_cooldown - 1, 0)
        )

        dy = jnp.where(action == Action.UP, -1.0, jnp.where(action == Action.DOWN, 1.0, 0.0))
        dx = jnp.where(action == Action.LEFT, -1.0, jnp.where(action == Action.RIGHT, 1.0, 0.0))

        stage_left_x = (self.consts.screen_width - self.renderer.sprites['STAGE'][0].shape[1]) // 2
        stage_right_x = stage_left_x + self.renderer.sprites['STAGE'][0].shape[1]

        new_x = jnp.clip(
            state.player_x + dx.astype(jnp.int32),
            stage_left_x,
            stage_right_x - self.consts.player_width,
        ).astype(jnp.int32)

        new_score = state.score

        new_lives = state.lives # TODO füge ein jnp.where hinzu um leben zu verlieren; dafür ist noch eine Kollisionserkennung notwendig

        # Check game over (optional: could be based on time or score limit)
        game_over = jnp.where(
            new_lives <= 0,
            jnp.array(True),
            state.game_over,
        )

        new_state = AsterixState(
            player_x=new_x,
            player_y=new_y,
            lives=state.lives,
            score=new_score,
            game_over=game_over,
            stage_cooldown=new_cooldown,
        )
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state, all_rewards)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AsterixState):
        # create chicken
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.player_width, dtype=jnp.int32),
            height=jnp.array(self.consts.player_height, dtype=jnp.int32),
        )


        return AsterixObservation(player=player, score=state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AsterixState, all_rewards: chex.Array = None) -> AsterixInfo:
        return AsterixInfo(all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsterixState, state: AsterixState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: AsterixState, state: AsterixState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AsterixState) -> bool:
        return state.game_over

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Asterix.
        Actions are:
        0: NOOP
        1: UP
        2: RIGHTS
        3: LEFT
        4: DOWN
        """
        return spaces.Discrete(5)

    def observation_space(self) -> spaces.Dict: # TODO kann entfernt werden? wird nicht verwendet / benötigt
        """Returns the observation space for Asterix.
        The observation contains:
        - player: EntityPosition (x, y, width, height)
        - score: int (0-99)
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            #"car": spaces.Box(low=0, high=160, shape=(10, 4), dtype=jnp.int32),
            #"score": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box: # TODO kann entfernt werden? wird nicht verwendet / benötigt
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

    def obs_to_flat_array(self, obs: AsterixObservation) -> jnp.ndarray: # TODO kann entfernt werden? wird nicht verwendet / benötigt
        """Convert observation to a flat array."""
        # Flatten chicken position and dimensions
        chicken_flat = jnp.concatenate([
            obs.player.x.reshape(-1),
            obs.player.y.reshape(-1),
            obs.player.width.reshape(-1),
            obs.player.height.reshape(-1)
        ])

        # Flatten car positions and dimensions
        #cars_flat = obs.car.reshape(-1)

        # Flatten score
        score_flat = obs.score.reshape(-1)

        # Concatenate all components
        return jnp.concatenate([chicken_flat, score_flat]).astype(jnp.int32) #TODO add cars_flat back an zweiter stelle when implemented


class AsterixRenderer(JAXGameRenderer):
    def __init__(self, consts: AsterixConstants = None):
        super().__init__()
        self.consts = consts or AsterixConstants()
        self.sprites, self.offsets = self._load_sprites()

    def _load_sprites(self):
        """Load all sprites required for Asterix rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(MODULE_DIR, "sprites/asterix/")

        sprites: Dict[str, Any] = {}
        offsets: Dict[str, Any] = {}

        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(sprite_path, f'{name}.npy')
            frame = jr.loadFrame(path)
            return frame.astype(jnp.uint8)

        sprite_names = [
            'ASTERIX', 'OBELIX', 'STAGE', 'TOP', 'BOTTOM',
        ]

        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                sprites[name] = loaded_sprite

        # pad the player sprites since they are used interchangably
        player_sprites, player_offsets = jr.pad_to_match([
            sprites['ASTERIX'], sprites['ASTERIX'] # first: player_hit, second: player_idle
        ])
        sprites['ASTERIX'] = player_sprites[0] # player_hit sprite
        sprites['ASTERIX'] = player_sprites[1] # player_idle sprite
        offsets['ASTERIX'] = player_offsets[0] # player_hit sprite offset
        offsets['ASTERIX'] = player_offsets[1] # player_idle sprite offset

        # --- Load Digit Sprites ---
        digit_path = os.path.join(sprite_path, 'DIGIT_{}.npy')
        digits = jr.load_and_pad_digits(digit_path, num_chars=10)
        sprites['digit'] = digits

        for key in sprites.keys():
            if isinstance(sprites[key], (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
            else:
                sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites, offsets

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """Render the game state to a raster image."""
        # --- Schwarzes Raster als Background rendern ---
        raster = jnp.zeros((self.consts.screen_height, self.consts.screen_width, 3), dtype=jnp.uint8)

        # --- Stages rendern ---
        stage_sprite = jr.get_sprite_frame(self.sprites['STAGE'], 0)
        stage_height = stage_sprite.shape[0]
        stage_x = (self.consts.screen_width - stage_sprite.shape[1]) // 2 # Center the stage horizontally

        for stage_y in self.consts.stage_borders:
            # oberste und unterste stage nicht rendern
            if stage_y == self.consts.stage_borders[0] or stage_y == self.consts.stage_borders[-1]:
                continue
            raster = jr.render_at(
                raster,
                stage_x,
                stage_y,  # Y-Position: Lane-Grenze
                stage_sprite
            )

        # --- Top und Bottom rendern ---
        top_sprite = jr.get_sprite_frame(self.sprites['TOP'], 0)
        bottom_sprite = jr.get_sprite_frame(self.sprites['BOTTOM'], 0)
        top_x = (self.consts.screen_width - top_sprite.shape[1]) // 2  # Center the top sprite horizontally
        # top_y = top_sprite.shape[0] // 2
        top_y = self.consts.top_border - self.consts.stage_spacing + stage_height
        bottom_x = (self.consts.screen_width - bottom_sprite.shape[1]) // 2  # Center the bottom sprite horizontally
        bottom_y = self.consts.stage_borders[-1]
        raster = jr.render_at(
            raster,
            top_x,
            top_y,
            top_sprite
        )
        raster = jr.render_at(
            raster,
            bottom_x,
            bottom_y,
            bottom_sprite
        )


        # --- Player rendern ---
        player_sprite = jr.get_sprite_frame(self.sprites['ASTERIX'], 0)
        player_hit_sprite = jr.get_sprite_frame(self.sprites['ASTERIX'], 0)
        player_sprite_offset = self.offsets['ASTERIX']
        player_hit_sprite_offset = self.offsets['ASTERIX']

        raster = jr.render_at(
            raster,
            state.player_x,
            state.player_y,
            player_sprite,
            flip_offset=player_sprite_offset
        )


        # ----------- SCORE -------------
        # Define score positions and spacing
        player_score_rightmost_digit_x = 49  # X position for the START of the player's rightmost digit (or single digit)
        max_score_digits = 6

        # Get digit sprites
        digit_sprites = self.sprites.get('digit', None)

        # Define the function to render scores if sprites are available
        def render_scores(raster_to_update):
            player_score_digits_indices = jr.int_to_digits(state.score, max_digits=max_score_digits)
            num_digits = jnp.maximum(1, jnp.floor(jnp.log10(jnp.maximum(state.score, 1))) + 1).astype(jnp.int32)
            digit_height = digit_sprites[0].shape[0]
            score_spacing = 8
            center_x = (self.consts.screen_width - (num_digits * score_spacing)) // 2
            score_y = self.consts.screen_height - digit_height - 5
            raster_updated = jr.render_label_selective(
                raster_to_update,
                center_x,
                score_y,
                player_score_digits_indices,
                digit_sprites[0],
                0,
                num_digits,
                spacing=score_spacing
            )
            return raster_updated

        # Render scores conditionally
        raster = jax.lax.cond(
            digit_sprites is not None,
            render_scores,
            lambda r: r,
            raster
        )

        # ----------- LIVES -------------
        num_lives = self.consts.num_lives
        life_sprite = jr.get_sprite_frame(self.sprites['ASTERIX'], 0)
        life_width = life_sprite.shape[1]
        life_height = life_sprite.shape[0]
        lives_spacing = 8  # Abstand zwischen den Leben
        total_lives_width = num_lives * life_width + (num_lives - 1) * lives_spacing
        lives_start_x = (self.consts.screen_width - total_lives_width) // 2
        lives_y = bottom_y + bottom_sprite.shape[0] + 3  # 3 Pixel unter Bottom

        def render_lives(raster_to_update):
            for i in range(num_lives-1):
                x = lives_start_x + i * (life_width + lives_spacing)
                raster_to_update = jr.render_at(
                    raster_to_update,
                    x,
                    lives_y,
                    life_sprite
                )
            return raster_to_update

        raster = jax.lax.cond(
            num_lives > 0,
            render_lives,
            lambda r: r,
            raster
        )


        return raster