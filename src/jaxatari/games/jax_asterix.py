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


def min_delay(level, base_min=30, spawn_accel=2, min_delay_clamp=20, max_delay_clamp=120):
    return jnp.clip(base_min - level * spawn_accel, min_delay_clamp, max_delay_clamp)


def max_delay(level, base_max=60, spawn_accel=2, min_delay_clamp=20, max_delay_clamp=120):
    return jnp.clip(base_max - level * spawn_accel, min_delay_clamp, max_delay_clamp)


def spawn_enemy(rng, level, platformY, screen_width, enemy_width, base_speed=1.0, speed_factor=0.1, lyre_height=8):
    rng_side, rng_platform = jax.random.split(rng)
    num_platforms = len(platformY) - 1
    platform = jax.random.randint(rng_platform, (), 0, num_platforms)
    y_center = (platformY[platform] + platformY[platform + 1]) // 2
    y = y_center - (lyre_height // 2)  # Korrigiert: Sprite mittig platzieren
    x = jax.lax.select(jax.random.bernoulli(rng_side), screen_width + enemy_width, -enemy_width)
    speed = base_speed + level * speed_factor
    vx = speed * jax.lax.select(x > 0, -1.0, 1.0)
    return Enemy(x, y, vx, True)


class AsterixConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    player_width: int = 8
    player_height: int = 8
    num_stages: int = 8
    stage_spacing: int = 16 # ursprünglich 16
    stage_positions: List[int] = None
    top_border: int = 23 # oberer Rand des Spielfelds
    bottom_border: int = 8 * stage_spacing + top_border
    cooldown_frames: int = 8 # Cooldown frames for lane changes
    num_lives: int = 3 # Anzahl der Leben


    stage_positions = [
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

class Enemy(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    alive: jnp.ndarray


class AsterixState(NamedTuple):
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
    spawn_timer: jnp.ndarray
    rng: jax.random.PRNGKey



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
        stage_borders = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        player_x = self.consts.screen_width // 2
        player_y = (stage_borders[-2] + stage_borders[-1]) // 2

        if key is None:
            key = jax.random.PRNGKey(0)
        platformY = jnp.array(self.consts.stage_positions[1:-1], dtype=jnp.int32)
        max_enemies = 32
        spawn_rng, timer_rng, state_rng = jax.random.split(key, 3)
        spawn_timer = jax.random.randint(timer_rng, (), min_delay(1), max_delay(1) + 1)
        enemies = Enemy(
            x=jnp.full((max_enemies,), -9999.0),
            y=jnp.full((max_enemies,), -9999.0),
            vx=jnp.zeros((max_enemies,)),
            alive=jnp.zeros((max_enemies,), dtype=bool)
        )

        state = AsterixState(
            player_x =jnp.array(player_x, dtype=jnp.int32),
            player_y=jnp.array(player_y, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32), # Start with 0 points
            lives=jnp.array(self.consts.num_lives, dtype=jnp.int32),  # 3 Leben
            game_over=jnp.array(False, dtype=jnp.bool_),
            stage_cooldown = jnp.array(self.consts.cooldown_frames, dtype=jnp.int32), # Cooldown initial 0
            bonus_life_stage=jnp.array(0, dtype=jnp.int32),  # Stage for bonus life
            player_direction=jnp.array(1, dtype=jnp.int32),  # Initial direction (1=links)
            enemies=enemies,
            spawn_timer=spawn_timer,
            rng=state_rng
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AsterixState, action: int) -> tuple[
        AsterixObservation, AsterixState, float, bool, AsterixInfo]:
        """Take a step in the game given an action"""
        player_height = self.consts.player_height

        cooldown_frames = self.consts.cooldown_frames  # Cooldown frames for lane changes
        can_switch_stage = state.stage_cooldown <= 0

        stage_borders = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
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

        stage_left_x = (self.consts.screen_width - self.renderer.sprites['STAGE'][0].shape[1]) // 2
        stage_right_x = stage_left_x + self.renderer.sprites['STAGE'][0].shape[1]

        dy = jnp.where(action == Action.UP, -1.0, jnp.where(action == Action.DOWN, 1.0, 0.0))
        dx = jnp.where(action == Action.LEFT, -1.0, jnp.where(action == Action.RIGHT, 1.0, 0.0))

        new_player_x = jnp.clip(
            state.player_x + dx.astype(jnp.int32),
            stage_left_x,
            stage_right_x - self.consts.player_width,
        ).astype(jnp.int32)

        new_player_direction = jnp.where(
            dx < 0, 1,  # links
            jnp.where(dx > 0, 2, state.player_direction)  # rechts oder idle: behalte alte Richtung
        )

        new_score = state.score

        bonus_thresholds = jnp.array([10_000, 30_000, 50_000, 80_000, 110_000], dtype=jnp.int32)
        bonus_interval = 40_000

        # Berechne, wie viele Bonusleben der Score verdient
        def calc_bonus_stage(score):
            # Zähle, wie viele Schwellen überschritten wurden
            below = jnp.sum(score >= bonus_thresholds)
            # Danach alle 40.000
            above = jnp.maximum(score - 110_000, 0) // bonus_interval
            return below + above

        new_bonus_stage = calc_bonus_stage(new_score)
        bonus_lives_gained = new_bonus_stage - state.bonus_life_stage
        new_lives = state.lives + bonus_lives_gained # TODO füge ein jnp.where hinzu um leben zu verlieren; dafür ist noch eine Kollisionserkennung notwendig


        # Check game over
        game_over = jnp.where(
            new_lives <= 0,
            jnp.array(True),
            state.game_over,
        )

        platformY = jnp.array(self.consts.stage_positions[1:-1], dtype=jnp.int32)
        enemy_width = 8
        screen_width = self.consts.screen_width
        level = 1  # oder aus Score ableiten

        # --- Feind-Spawn- und Update-Logik ---
        rng, rng_spawn, rng_delay = jax.random.split(state.rng, 3)
        spawn_timer = state.spawn_timer - 1


        def spawn_fn(args):
            enemies, rng_spawn, level = args
            new_enemy = spawn_enemy(rng_spawn, level, platformY, screen_width, enemy_width)
            # Prüfe, ob auf dieser Ebene schon ein Lyre aktiv ist
            already_exists = jnp.any((enemies.y == new_enemy.y) & enemies.alive)

            # Nur spawnen, wenn noch keiner auf dieser Ebene ist
            def do_spawn():
                idx = jnp.argmax(~enemies.alive)
                return enemies._replace(
                    x=enemies.x.at[idx].set(new_enemy.x),
                    y=enemies.y.at[idx].set(new_enemy.y),
                    vx=enemies.vx.at[idx].set(new_enemy.vx),
                    alive=enemies.alive.at[idx].set(True)
                )

            return jax.lax.cond(already_exists, lambda: enemies, do_spawn)

        should_spawn = spawn_timer <= 0
        enemies = jax.lax.cond(
            should_spawn,
            spawn_fn,
            lambda args: args[0],
            (state.enemies, rng_spawn, level)
        )

        def new_timer_fn(_):
            minD = min_delay(level)
            maxD = max_delay(level)
            return jax.random.randint(rng_delay, (), minD, maxD + 1)

        spawn_timer = jax.lax.cond(
            should_spawn,
            new_timer_fn,
            lambda _: spawn_timer,
            operand=None
        )

        # Feinde bewegen und entfernen
        new_enemy_x = enemies.x + enemies.vx
        alive = (new_enemy_x >= -enemy_width) & (new_enemy_x <= screen_width + enemy_width) & enemies.alive
        enemies = enemies._replace(x=new_enemy_x, alive=alive)

        # --- Kollisionserkennung Spieler <-> Gegner ---
        def check_collision(player_x, player_y, player_w, player_h, enemy_x, enemy_y, enemy_w, enemy_h):
            return (
                    (player_x < enemy_x + enemy_w) &
                    (player_x + player_w > enemy_x) &
                    (player_y < enemy_y + enemy_h) &
                    (player_y + player_h > enemy_y)
            )

        player_w = self.consts.player_width
        player_h = self.consts.player_height
        enemy_w = 8  # ggf. anpassen
        enemy_h = 8  # ggf. anpassen

        collisions = check_collision(
            new_player_x, new_y, player_w, player_h,
            enemies.x, enemies.y, enemy_w, enemy_h
        ) & enemies.alive

        any_collision = jnp.any(collisions)

        # Leben abziehen und Gegner deaktivieren, falls Kollision
        new_lives = jnp.where(any_collision, new_lives - 1, new_lives)
        game_over = jnp.where(new_lives <= 0, True, game_over)
        enemies = enemies._replace(
            alive=jnp.where(collisions, False, enemies.alive)
        )

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
            spawn_timer=spawn_timer,
            rng=rng,  # Update the RNG for the next step
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
            'ASTERIX_LEFT', 'ASTERIX_RIGHT', 'ASTERIX_LEFT_HIT', 'ASTERIX_RIGHT_HIT','OBELIX', 'STAGE', 'TOP', 'BOTTOM', 'LYRE_LEFT', 'LYRE_RIGHT',
        ]

        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                sprites[name] = loaded_sprite

        # pad the player sprites since they are used interchangably
        player_sprites, player_offsets = jr.pad_to_match([
            sprites['ASTERIX_LEFT_HIT'], sprites['ASTERIX_LEFT'] # first: player_hit, second: player_idle
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
        # ----------- RASTER INITIALIZATION -------------
        raster = jnp.zeros((self.consts.screen_height, self.consts.screen_width, 3), dtype=jnp.uint8)

        # ----------- STAGE -------------
        stage_sprite = jr.get_sprite_frame(self.sprites['STAGE'], 0)
        stage_height = stage_sprite.shape[0]
        stage_x = (self.consts.screen_width - stage_sprite.shape[1]) // 2 # Center the stage horizontally

        for stage_y in self.consts.stage_positions:
            # oberste und unterste stage nicht rendern
            if stage_y == self.consts.stage_positions[0] or stage_y == self.consts.stage_positions[-1]:
                continue
            raster = jr.render_at(
                raster,
                stage_x,
                stage_y,  # Y-Position: Lane-Grenze
                stage_sprite
            )

        # ----------- TOP AND BOTTOM -------------
        top_sprite = jr.get_sprite_frame(self.sprites['TOP'], 0)
        bottom_sprite = jr.get_sprite_frame(self.sprites['BOTTOM'], 0)
        top_x = (self.consts.screen_width - top_sprite.shape[1]) // 2  # Center the top sprite horizontally
        # top_y = top_sprite.shape[0] // 2
        top_y = self.consts.top_border - self.consts.stage_spacing + stage_height
        bottom_x = (self.consts.screen_width - bottom_sprite.shape[1]) // 2  # Center the bottom sprite horizontally
        bottom_y = self.consts.stage_positions[-1]
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


        # ----------- PLAYER -------------
        player_sprite = jr.get_sprite_frame(self.sprites['ASTERIX'], 0)
        player_hit_sprite = jr.get_sprite_frame(self.sprites['ASTERIX'], 0)
        player_sprite_offset = self.offsets['ASTERIX']
        player_hit_sprite_offset = self.offsets['ASTERIX']

        direction = state.player_direction
        player_sprite = jax.lax.switch(
            direction - 1,  # 1=links, 2=rechts → 0/1 für switch
            [
                lambda _: jr.get_sprite_frame(self.sprites['ASTERIX_LEFT'], 0),  # 1: links
                lambda _: jr.get_sprite_frame(self.sprites['ASTERIX_RIGHT'], 0),  # 2: rechts
            ],
            None  # Dummy-Argument, wird von den Lambdas ignoriert
        )

        raster = jr.render_at(
            raster,
            state.player_x,
            state.player_y,
            player_sprite,
            flip_offset=player_sprite_offset
        )

        # ----------- LYRES -------------
        lyre_left_sprite = jr.get_sprite_frame(self.sprites['LYRE_LEFT'], 0)
        lyre_right_sprite = jr.get_sprite_frame(self.sprites['LYRE_RIGHT'], 0)

        # ----------- LYRES (Feinde) -------------
        def render_lyres(raster_to_update):
            def render_single_lyre(i, raster_inner):
                is_alive = state.enemies.alive[i]
                x = state.enemies.x[i]
                y = state.enemies.y[i]
                vx = state.enemies.vx[i]
                lyre_sprite = jax.lax.select(
                    vx < 0,
                    lyre_left_sprite,
                    lyre_right_sprite
                )
                # Nur rendern, wenn alive
                raster_inner = jax.lax.cond(
                    is_alive,
                    lambda r: jr.render_at(r, x.astype(jnp.int32), y.astype(jnp.int32), lyre_sprite),
                    lambda r: r,
                    raster_inner
                )
                return raster_inner

            raster_out = raster_to_update
            for i in range(state.enemies.x.shape[0]):
                raster_out = render_single_lyre(i, raster_out)
            return raster_out

        raster = render_lyres(raster)



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
            score_x = ((self.consts.screen_width - (num_digits * score_spacing)) // 2) + 20
            score_y = bottom_y + bottom_sprite.shape[0] + jr.get_sprite_frame(self.sprites['ASTERIX'], 0).shape[0] + 6
            raster_updated = jr.render_label_selective(
                raster_to_update,
                score_x,
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
        num_lives = jnp.maximum(state.lives, 0).astype(jnp.int32) - 1
        life_sprite = jax.lax.switch(
            state.player_direction - 1,
            [
                lambda _: jr.get_sprite_frame(self.sprites['ASTERIX_LEFT'], 0),  # links
                lambda _: jr.get_sprite_frame(self.sprites['ASTERIX_RIGHT'], 0),  # rechts
            ],
            None
        )
        life_width = life_sprite.shape[1]
        life_height = life_sprite.shape[0]
        lives_spacing = 8  # Abstand zwischen den Leben
        total_lives_width = num_lives * life_width + (num_lives - 1) * lives_spacing
        lives_start_x = (self.consts.screen_width - total_lives_width) // 2
        lives_y = bottom_y + bottom_sprite.shape[0] + 3  # 3 Pixel unter Bottom

        def render_life(i, raster_to_update):
            x = lives_start_x + i * (life_width + lives_spacing)
            return jr.render_at(
                raster_to_update,
                x,
                lives_y,
                life_sprite
            )

        def render_lives(raster_to_update):
            def body_fun(i, r):
                return render_life(i, r)

            return jax.lax.fori_loop(0, num_lives, body_fun, raster_to_update)

        raster = jax.lax.cond(
            num_lives > 0,
            render_lives,
            lambda r: r,
            raster
        )


        return raster