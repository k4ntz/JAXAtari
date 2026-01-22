import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from functools import partial
from typing import NamedTuple, Tuple, Optional
import os

import chex
import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class HangmanConstants(NamedTuple):
    # Dimensions
    WIDTH: int = 160
    HEIGHT: int = 210

    # Game Logic
    ALPHABET_SIZE: int = 26
    SEPARATOR_IDX: int = 26
    CYCLE_SIZE: int = 27 # 26 letters + divider
    PAD_TOKEN: int = 26
    L_MAX: int = 6 # only use words with max 6 letter, because 7 underscores don't fit in
    MAX_MISSES: int = 11
    STEP_PENALTY: float = 0.0

    # Background Color
    BG_COLOR: Tuple[int, int, int] = (167, 26, 26)

    # Layout - Underscores & Letters
    UND_W: int = 20
    UND_H: int = 8
    UND_GAP: int = 4
    UND_Y: int = 181
    UND_START_X: int = 9
    LETTER_DIST: int = 8  # Distance between underscore top and letter bottom

    # Layout - Hangman
    HANGMAN_X: int = 17
    HANGMAN_Y: int = 37

    # Layout - Score
    SCORE_Y: int = 7
    SCORE_P_X: int = 33
    SCORE_C_X: int = 112
    SCORE_DIGITS: int = 1

    # Layout - Preview Letters
    PREVIEW_X: int = 108
    PREVIEW_Y: int = 72

CONSTANTS = HangmanConstants()


class HangmanConstants(NamedTuple):
    MAX_MISSES: int = MAX_MISSES
    STEP_PENALTY: float = STEP_PENALTY
    DIFFICULTY_MODE: str = DIFFICULTY_MODE
    TIMER_SECONDS: int = TIMER_SECONDS
    STEPS_PER_SECOND: int = STEPS_PER_SECOND


class HangmanState(NamedTuple):
    key: chex.Array
    word: chex.Array
    length: chex.Array
    mask: chex.Array
    guessed: chex.Array
    misses: chex.Array
    lives: chex.Array
    cursor_idx: chex.Array
    done: chex.Array
    reward: chex.Array
    step_counter: chex.Array
    score: chex.Array
    round_no: chex.Array
    time_left_steps: chex.Array
    cpu_score: chex.Array
    timer_max_steps: chex.Array
    last_commit: chex.Array


class HangmanObservation(NamedTuple):
    revealed: chex.Array
    mask: chex.Array
    guessed: chex.Array
    misses: chex.Array
    lives: chex.Array
    cursor_idx: chex.Array


class HangmanInfo(NamedTuple):
    time: chex.Array


# helpers functions
@jax.jit
def _sample_word(key: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    key, sub = jrandom.split(key)
    idx = jrandom.randint(sub, shape=(), minval=0, maxval=N_WORDS, dtype=jnp.int32)
    return key, WORDS_ENC[idx], WORDS_LEN[idx]


@jax.jit
def _compute_revealed(word: chex.Array, mask: chex.Array) -> chex.Array:
    return jnp.where(mask.astype(bool), word, CONSTANTS.PAD_TOKEN)


def _action_delta_cursor(action: chex.Array) -> chex.Array:
    # Actions that move cursor up: UP, UPRIGHT, UPLEFT, UPFIRE, UPRIGHTFIRE, UPLEFTFIRE
    up_like = jnp.logical_or(
        jnp.logical_or(
            jnp.logical_or(action == Action.UP, action == Action.UPRIGHT),
            jnp.logical_or(action == Action.UPLEFT, action == Action.UPFIRE)
        ),
        jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE)
    )
    # Actions that move cursor down: DOWN, DOWNRIGHT, DOWNLEFT, DOWNFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE
    down_like = jnp.logical_or(
        jnp.logical_or(
            jnp.logical_or(action == Action.DOWN, action == Action.DOWNRIGHT),
            jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNFIRE)
        ),
        jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE)
    )
    return jnp.where(up_like, -1, jnp.where(down_like, 1, 0)).astype(jnp.int32)


def _action_commit(action: chex.Array) -> chex.Array:
    # All FIRE actions commit: FIRE, UPFIRE, DOWNFIRE, RIGHTFIRE, LEFTFIRE, 
    # UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE
    return jnp.logical_or(
        jnp.logical_or(
            jnp.logical_or(action == Action.FIRE, action == Action.UPFIRE),
            jnp.logical_or(action == Action.DOWNFIRE, action == Action.RIGHTFIRE)
        ),
        jnp.logical_or(
            jnp.logical_or(action == Action.LEFTFIRE, action == Action.UPRIGHTFIRE),
            jnp.logical_or(action == Action.UPLEFTFIRE, 
                jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE))
        )
    )


@jax.jit
def _advance_cursor_skip_guessed(cursor: chex.Array,
                                 delta: chex.Array,
                                 guessed: chex.Array) -> chex.Array:
    """Move one step in delta direction. Cycle size is 27 (letters + divider)."""
    step = jnp.where(delta > 0, 1, jnp.where(delta < 0, -1, 0)).astype(jnp.int32)

    cur0 = jnp.where(step == 0, cursor, (cursor + step) % CONSTANTS.CYCLE_SIZE)

    def cond_fun(carry):
        cur, n = carry
        is_guessed = jnp.logical_and(cur < CONSTANTS.ALPHABET_SIZE, guessed[cur] == 1)

        need_move = jnp.logical_and(step != 0, is_guessed)
        return jnp.logical_and(n < CONSTANTS.CYCLE_SIZE, need_move)

    def body_fun(carry):
        cur, n = carry
        return ((cur + step) % CONSTANTS.CYCLE_SIZE, n + 1)

    cur, _ = lax.while_loop(cond_fun, body_fun, (cur0, jnp.int32(0)))
    return cur


# environment
class JaxHangman(JaxEnvironment[HangmanState, HangmanObservation, HangmanInfo, Action]):
    def __init__(self, reward_funcs: Optional[list] = None, *,
                 max_misses: int = 11,
                 step_penalty: float = 0.0,
                 difficulty_mode: str = "B",
                 timer_seconds: int = 20,
                 steps_per_second: int = 30):
        super().__init__()

        # Init Constants
        self.consts = HangmanConstants(
            MAX_MISSES=int(max_misses),
            STEP_PENALTY=float(step_penalty)
        )

        self.renderer = HangmanRenderer(self.consts)
        self.max_misses = self.consts.MAX_MISSES
        self.step_penalty = self.consts.STEP_PENALTY

    def __init__(self, consts: HangmanConstants = None):
        consts = consts or HangmanConstants()
        super().__init__(consts)
        self.renderer = HangmanRenderer()
        self.consts = consts

        self.action_set = [Action.NOOP, Action.FIRE, Action.UP, Action.DOWN, Action.UPFIRE, Action.DOWNFIRE]
        # obs size depends on L_MAX
        self.obs_size = self.consts.L_MAX + self.consts.L_MAX + self.consts.ALPHABET_SIZE + 3
        self.reward_funcs = tuple(reward_funcs) if reward_funcs is not None else None

        # Compute derived values from constants
        self.timed = 1 if str(consts.DIFFICULTY_MODE).upper() == "A" else 0
        self.timer_steps = int(consts.TIMER_SECONDS * consts.STEPS_PER_SECOND)

        self._rng_key = jrandom.PRNGKey(0)

    def seed(self, seed: int | chex.Array) -> chex.Array:
        if isinstance(seed, int):
            key = jrandom.PRNGKey(int(seed))
        else:
            arr = jnp.asarray(seed)
            key = arr if arr.shape == (2,) else jrandom.PRNGKey(int(jnp.uint32(jnp.sum(arr))))
        self._rng_key = key
        return key

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[HangmanObservation, HangmanState]:
        if key is None:
            key = self._rng_key

        key, word, length = _sample_word(key)

        self._rng_key = key
        # init round timer
        time0 = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)
        tmax = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)

        state = HangmanState(
            key=key,
            word=word,
            length=length,
            mask=jnp.zeros((self.consts.L_MAX,), dtype=jnp.int32),
            guessed=jnp.zeros((self.consts.ALPHABET_SIZE,), dtype=jnp.int32),
            misses=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_MISSES, dtype=jnp.int32),
            cursor_idx=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            reward=jnp.array(0.0, dtype=jnp.float32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            round_no=jnp.array(1, dtype=jnp.int32),
            time_left_steps=time0,
            timer_max_steps=tmax,
            cpu_score=jnp.array(0, dtype=jnp.int32),
            last_commit=jnp.array(0, dtype=jnp.int32),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HangmanState, action: chex.Array) -> Tuple[
        HangmanObservation, HangmanState, float, bool, HangmanInfo]:
        # Calculate raw button state
        commit = _action_commit(action).astype(jnp.int32)

        # Action only valid if rising edge AND cursor is not on the letter separator (26)
        is_valid_target = state.cursor_idx != self.consts.SEPARATOR_IDX
        do_action = jnp.logical_and(
            jnp.logical_and(commit, jnp.logical_not(state.last_commit)),
            is_valid_target
        )

        delta = _action_delta_cursor(action)
        delta = jnp.where(state.step_counter % 6 == 0, delta, 0)

        def _get_new_round_state(s: HangmanState, step_reward: chex.Array, score_delta: chex.Array,
                                 cpu_delta: chex.Array, current_commit_val: chex.Array) -> HangmanState:
            key, word, length = _sample_word(s.key)
            time0 = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)
            tmax = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)

            return HangmanState(
                key=key, word=word, length=length,
                mask=jnp.zeros((self.consts.L_MAX,), dtype=jnp.int32),
                guessed=jnp.zeros((self.consts.ALPHABET_SIZE,), dtype=jnp.int32),
                misses=jnp.array(0, dtype=jnp.int32),
                lives=jnp.array(self.consts.MAX_MISSES, dtype=jnp.int32),
                cursor_idx=jnp.array(0, dtype=jnp.int32),
                done=jnp.array(False),
                reward=step_reward,
                step_counter=s.step_counter + 1,
                score=(s.score + score_delta).astype(jnp.int32),
                round_no=(s.round_no + 1).astype(jnp.int32),
                time_left_steps=time0,
                timer_max_steps=tmax,
                cpu_score=(s.cpu_score + cpu_delta).astype(jnp.int32),
                last_commit=current_commit_val,
            )

        def _continue_round(s: HangmanState) -> HangmanState:
            cursor = _advance_cursor_skip_guessed(s.cursor_idx, delta, s.guessed)

            # Timer logic
            t0 = s.time_left_steps
            t1 = jnp.where(self.timed == 1, jnp.maximum(t0 - 1, 0), t0)
            timed_out = jnp.logical_and(self.timed == 1, t1 == 0)

            idx = jnp.arange(self.consts.L_MAX, dtype=jnp.int32)
            within = idx < s.length

            def on_commit(s2: HangmanState) -> HangmanState:
                already = s2.guessed[cursor] == 1
                guessed = s2.guessed.at[cursor].set(1)

                pos_hits = (s2.word == cursor).astype(jnp.int32) * within.astype(jnp.int32)
                any_hit = jnp.any(pos_hits == 1)
                mask = jnp.where(pos_hits.astype(bool), 1, s2.mask)

                wrong = jnp.logical_and(jnp.logical_not(any_hit), jnp.logical_not(already))
                misses = s2.misses + wrong.astype(jnp.int32)
                lives = s2.lives - wrong.astype(jnp.int32)

                n_revealed = jnp.sum(jnp.where(within, mask, 0))
                all_revealed = (n_revealed == s2.length)
                lost = misses >= self.consts.MAX_MISSES

                mask_final = jnp.where(lost, jnp.where(within, 1, mask), mask)
                step_reward = jnp.where(all_revealed, 1.0, jnp.where(lost, -1.0, self.step_penalty)).astype(jnp.float32)

                round_ended = jnp.logical_or(all_revealed, lost)

                base = HangmanState(
                    key=s2.key, word=s2.word, length=s2.length,
                    mask=mask_final, guessed=guessed, misses=misses, lives=lives,
                    cursor_idx=cursor, done=jnp.array(False), reward=step_reward,
                    step_counter=s2.step_counter + 1, score=s2.score, round_no=s2.round_no,
                    time_left_steps=jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32),
                    cpu_score=s2.cpu_score, timer_max_steps=s2.timer_max_steps,
                    last_commit=commit,
                )

                return lax.cond(
                    round_ended,
                    lambda _: _get_new_round_state(s2, step_reward,
                                                   score_delta=all_revealed.astype(jnp.int32),
                                                   cpu_delta=lost.astype(jnp.int32),
                                                   current_commit_val=commit),  # Pass raw commit
                    lambda s_: s_,
                    base
                )

            def no_commit(s2: HangmanState) -> HangmanState:
                active = (self.timed == 1)
                t0 = s2.time_left_steps
                t1 = jnp.where(active, jnp.maximum(t0 - 1, 0), t0)
                timed_out = jnp.logical_and(active, t1 == 0)

                add_miss = jnp.where(timed_out, 1, 0).astype(jnp.int32)
                misses = s2.misses + add_miss
                lives = s2.lives - add_miss
                lost = misses >= self.consts.MAX_MISSES

                idx = jnp.arange(self.consts.L_MAX, dtype=jnp.int32)
                within = idx < s2.length
                mask_final = jnp.where(lost, jnp.where(within, 1, s2.mask), s2.mask)

                t_next = jnp.where(timed_out, jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32),
                                   t1)
                step_reward = jnp.where(lost, -1.0, self.step_penalty).astype(jnp.float32)

                base = HangmanState(
                    key=s2.key, word=s2.word, length=s2.length,
                    mask=mask_final, guessed=s2.guessed, misses=misses, lives=lives,
                    cursor_idx=cursor, done=jnp.array(False), reward=step_reward,
                    step_counter=s2.step_counter + 1, score=s2.score, round_no=s2.round_no,
                    time_left_steps=t_next, cpu_score=s2.cpu_score, timer_max_steps=s2.timer_max_steps,
                    last_commit=commit,  # Store raw commit state
                )

                return lax.cond(
                    lost,
                    lambda _: _get_new_round_state(s2, step_reward,
                                                   score_delta=jnp.array(0, dtype=jnp.int32),
                                                   cpu_delta=jnp.array(1, dtype=jnp.int32),
                                                   current_commit_val=commit),  # Pass raw commit
                    lambda s_: s_,
                    base
                )

            return lax.cond(do_action, on_commit, no_commit, s)

        def _freeze(_s):
            return HangmanState(
                key=_s.key, word=_s.word, length=_s.length,
                mask=_s.mask, guessed=_s.guessed, misses=_s.misses, lives=_s.lives,
                cursor_idx=_s.cursor_idx,
                done=_s.done,
                reward=jnp.array(0.0, dtype=jnp.float32),
                step_counter=_s.step_counter,
                score=_s.score, round_no=_s.round_no,
                time_left_steps=_s.time_left_steps,
                cpu_score=_s.cpu_score,
                timer_max_steps=_s.timer_max_steps,
                last_commit=commit,
            )

        next_state = _continue_round(state)

        done = self._get_done(next_state)
        env_reward = self._get_env_reward(state, next_state)

        obs = self._get_observation(next_state)
        info = self._get_info(next_state)
        return obs, next_state, env_reward, done, info

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "revealed": spaces.Box(low=0, high=self.consts.PAD_TOKEN, shape=(self.consts.L_MAX,), dtype=jnp.int32),
            "mask": spaces.Box(low=0, high=1, shape=(self.consts.L_MAX,), dtype=jnp.int32),
            "guessed": spaces.Box(low=0, high=1, shape=(self.consts.ALPHABET_SIZE,), dtype=jnp.int32),
            "misses": spaces.Box(low=0, high=self.consts.MAX_MISSES, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.MAX_MISSES, shape=(), dtype=jnp.int32),
            "cursor_idx": spaces.Box(low=0, high=self.consts.ALPHABET_SIZE, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HangmanState) -> HangmanObservation:
        revealed = _compute_revealed(state.word, state.mask)
        return HangmanObservation(
            revealed=revealed, mask=state.mask, guessed=state.guessed,
            misses=state.misses, lives=state.lives, cursor_idx=state.cursor_idx,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: HangmanObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.revealed.flatten(),
            obs.mask.flatten(),
            obs.guessed.flatten(),
            obs.misses.reshape((1,)).astype(jnp.int32),
            obs.lives.reshape((1,)).astype(jnp.int32),
            obs.cursor_idx.reshape((1,)).astype(jnp.int32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: HangmanState, state: HangmanState):
        # single scalar float32
        return jnp.asarray(state.reward, dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: HangmanState, state: HangmanState):
        # return the reward
        return state.reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: HangmanState) -> bool:
        return state.done

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HangmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_info(self, state: HangmanState) -> HangmanInfo:
        return HangmanInfo(time=state.step_counter)


# render
class HangmanRenderer(JAXGameRenderer):
    def __init__(self, consts: HangmanConstants = CONSTANTS):
        super().__init__()
        self.consts = consts
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Create procedural background (Solid color only)
        background_sprite = self._create_background_sprite()

        # 2. Define asset config
        asset_config = self._get_asset_config(background_sprite)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(base_dir, "sprites", "hangman")

        # 3. Load assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        # Precompute the height of the letter sprites for correct positioning above underscores
        self.LETTER_HEIGHT = self.SHAPE_MASKS["letters"].shape[1]

    def _create_background_sprite(self) -> jnp.ndarray:
        """Creates an RGBA background solid color."""
        # Start with solid BG color
        bg_color_rgba = (*self.consts.BG_COLOR, 255)
        bg = jnp.tile(jnp.array(bg_color_rgba, dtype=jnp.uint8), (self.consts.HEIGHT, self.consts.WIDTH, 1))
        return bg

    def _get_asset_config(self, background_sprite: jnp.ndarray) -> list:
        letter_files = [os.path.join("letters", f"{chr(97 + i)}.npy") for i in range(26)]

        hangman_files = [os.path.join("hangman", f"hangman_{i}.npy") for i in range(1, 12)]

        return [
            {'name': 'background', 'type': 'background', 'data': background_sprite},

            # Group for letters A-Z (indices 0-25)
            {'name': 'letters', 'type': 'group', 'files': letter_files},

            # Group for hangman stages (indices 0-10 correspond to misses 1-11)
            {'name': 'hangman', 'type': 'group', 'files': hangman_files},

            {'name': 'underscore', 'type': 'single', 'file': 'underscore.npy'},
            {'name': 'divider', 'type': 'single', 'file': 'letter_divider.npy'},

            # Digits
            {'name': 'player_digits', 'type': 'digits', 'pattern': os.path.join('player_score', '{}.npy')},
            {'name': 'cpu_digits', 'type': 'digits', 'pattern': os.path.join('cpu_score', '{}.npy')},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # --- 1. Underscores ---
        underscore_mask = self.SHAPE_MASKS["underscore"]
        length = state.length

        def draw_underscore_fn(i, r):
            x = self.consts.UND_START_X + i * (self.consts.UND_W + self.consts.UND_GAP)
            return lax.cond(
                i < length,
                lambda rr: self.jr.render_at(rr, x, self.consts.UND_Y, underscore_mask),
                lambda rr: rr,
                r
            )

        raster = lax.fori_loop(0, self.consts.L_MAX, draw_underscore_fn, raster)

        # --- 2. Revealed Letters ---
        letter_w = self.SHAPE_MASKS["letters"].shape[2]
        x_inset = (self.consts.UND_W - letter_w) // 2

        # Y Calculation
        draw_y = self.consts.UND_Y - self.consts.LETTER_DIST - self.LETTER_HEIGHT

        def draw_revealed_fn(i, r):
            x = self.consts.UND_START_X + i * (self.consts.UND_W + self.consts.UND_GAP) + x_inset

            cond = jnp.logical_and(i < length, state.mask[i] == 1)
            idx = state.word[i]
            letter_mask = self.SHAPE_MASKS["letters"][idx]

            return lax.cond(
                cond,
                lambda rr: self.jr.render_at(rr, x, draw_y, letter_mask),
                lambda rr: rr,
                r
            )

        raster = lax.fori_loop(0, self.consts.L_MAX, draw_revealed_fn, raster)

        # --- 3. Preview Letter ---
        def draw_preview(r):
            # If cursor is at SEPARATOR_IDX (26), draw divider. Else draw letter.
            is_separator = state.cursor_idx == self.consts.SEPARATOR_IDX

            return lax.cond(
                is_separator,
                lambda rr: self.jr.render_at(rr, self.consts.PREVIEW_X, self.consts.PREVIEW_Y,
                                             self.SHAPE_MASKS["divider"]),
                lambda rr: self.jr.render_at(rr, self.consts.PREVIEW_X, self.consts.PREVIEW_Y,
                                             self.SHAPE_MASKS["letters"][state.cursor_idx]),
                r
            )

        raster = draw_preview(raster)

        # --- 4. Hangman Body ---
        misses = jnp.clip(state.misses, 0, 11)

        def draw_hangman(r):
            sprite_idx = misses - 1
            mask = self.SHAPE_MASKS["hangman"][sprite_idx]
            return self.jr.render_at(r, self.consts.HANGMAN_X, self.consts.HANGMAN_Y, mask)

        raster = lax.cond(misses > 0, draw_hangman, lambda r: r, raster)

        # --- 5. Scores ---
        p_digits = self.jr.int_to_digits(state.score, max_digits=self.consts.SCORE_DIGITS)
        raster = self.jr.render_label(raster, self.consts.SCORE_P_X, self.consts.SCORE_Y, p_digits,
                                      self.SHAPE_MASKS["player_digits"], spacing=12,
                                      max_digits=self.consts.SCORE_DIGITS)

        c_digits = self.jr.int_to_digits(state.cpu_score, max_digits=self.consts.SCORE_DIGITS)
        raster = self.jr.render_label(raster, self.consts.SCORE_C_X, self.consts.SCORE_Y, c_digits,
                                      self.SHAPE_MASKS["cpu_digits"], spacing=12, max_digits=self.consts.SCORE_DIGITS)

        # --- Final Palette Lookup ---
        return self.jr.render_from_palette(raster, self.PALETTE)




# --- Word List Processing ---

_RAW_WORDS: tuple[str, ...] = (
    "ABLE", "ABOUT", "ABOVE", "ABSORB", "ABUSE", "ACADEMY", "ACTION", "ACTOR", "ACUTE", "ADAPT", "ADDER", "ADDON",
    "ADEPT", "ADJUST", "ADMIRE", "ADOBE", "ADOPT", "ADORE", "ADULT", "AFFAIR", "AFFORD", "AFRAID", "AFRESH",
    "AGENT", "AGILE", "AGING", "AGREE", "AHEAD", "AIDED", "AIMED", "AIR", "AIRPORT", "AISLE", "ALARM", "ALBUM",
    "ALERT", "ALGAE", "ALIBI", "ALLEY", "ALLOY", "ALMOND", "ALMOST", "ALONE", "ALONG", "ALPHA", "ALTAR", "ALTER",
    "ALWAYS", "AMBER", "AMBLE", "AMEND", "AMONG", "AMOUNT", "AMPLE", "ANCHOR", "ANCIENT", "ANGLE", "ANGRY",
    "ANIMAL", "ANKLE", "ANNEX", "ANNUAL", "ANSWER", "ANT", "ANTELOPE", "ANTLER", "ANVIL", "AORTA", "APART",
    "APE", "APEX", "APPLE", "APPLY", "APRIL", "APRON", "ARCADE", "ARCH", "ARCTIC", "ARENA", "ARGUE", "ARISE",
    "ARM", "ARMOR", "ARMY", "AROMA", "ARROW", "ART", "ARTERY", "ARTIST", "ASCEND", "ASH", "ASIDE", "ASK",
    "ASLEEP", "ASPECT", "ASPEN", "ASSIST", "ASSORT", "ASSURE", "ASTRAL", "ATLAS", "ATOM", "ATOMIC", "ATONCE",
    "ATTACH", "ATTIC", "AUDIO", "AUGUST", "AUNT", "AURORA", "AUTHORS", "AUTO", "AUTUMN", "AUX", "AVENUE", "AVID",
    "AVOID", "AWAKE", "AWARE", "AWARD", "AWOKE", "AXIS", "AZURE", "BABY", "BACK", "BACON", "BADGER", "BAG",
    "BAKER", "BALANCE", "BALD", "BALLET", "BALLOON", "BALMY", "BAMBOO", "BANANA", "BAND", "BANDIT", "BANK",
    "BAR", "BARB", "BARBER", "BARE", "BARGE", "BARN", "BARRACK", "BARREL", "BASALT", "BASE", "BASIC", "BASIL",
    "BASKET", "BASS", "BATCH", "BATON", "BAT", "BATTLE", "BAY", "BEACH", "BEACON", "BEAD", "BEAK", "BEAM",
    "BEAN", "BEAR", "BEAVER", "BECOME", "BED", "BEE", "BEEF", "BEET", "BEETLE", "BEGAN", "BEGIN", "BEGUN",
    "BEING", "BELIEF", "BELL", "BELLY", "BELT", "BENCH", "BERRY", "BERTH", "BEST", "BET", "BETA", "BETTER",
    "BEYOND", "BIBLE", "BICYCLE", "BID", "BIG", "BIGGER", "BIKE", "BILL", "BIN", "BIND", "BIRD", "BIRTH",
    "BISON", "BISTRO", "BIT", "BITE", "BITTER", "BLACK", "BLADE", "BLAME", "BLANK", "BLAST", "BLAZE", "BLEAK",
    "BLEED", "BLEND", "BLESS", "BLIMP", "BLIND", "BLINK", "BLIP", "BLISS", "BLOOM", "BLOSSOM", "BLOW", "BLUE",
    "BLUFF", "BLUNT", "BLUR", "BOARD", "BOAT", "BODY", "BOIL", "BOLD", "BOLT", "BOMB", "BOND", "BONE", "BONUS",
    "BOOK", "BOOST", "BOOT", "BOOTH", "BOOTS", "BORDER", "BORE", "BORN", "BORROW", "BOSS", "BOTANY", "BOTH",
    "BOTTLE", "BOTTOM", "BOUNCE", "BOUND", "BOUT", "BOWL", "BOX", "BOXER", "BOY", "BRACE", "BRANCH", "BRAND",
    "BRASS", "BRAVE", "BRAVO", "BREAD", "BREAK", "BREAST", "BREATH", "BREEZE", "BREW", "BRICK", "BRIDE",
    "BRIDGE", "BRIEF", "BRIGHT", "BRIM", "BRING", "BRISK", "BRISTLE", "BROAD", "BROIL", "BROKE", "BRONZE",
    "BROOK", "BROOM", "BROWN", "BROWSE", "BRUSH", "BRUTE", "BUBBLE", "BUCK", "BUD", "BUDGET", "BUFF",
    "BUFFET", "BUG", "BUGGY", "BUILD", "BUILT", "BULB", "BULGE", "BULK", "BULL", "BULLET", "BUMPER", "BUNCH",
    "BUNDLE", "BUNNY", "BUNT", "BURDEN", "BURGER", "BURROW", "BURST", "BUS", "BUSH", "BUSY", "BUTCHER", "BUTTER",
    "BUTTON", "BUY", "BUZZ", "CAB", "CABIN", "CABLE", "CACHE", "CACTUS", "CAGE", "CAKE", "CALF", "CALL",
    "CALM", "CAMEL", "CAMP", "CANAL", "CANARY", "CANCEL", "CANDLE", "CANDY", "CANOE", "CANON", "CANOPY",
    "CANTEEN", "CANVAS", "CANYON", "CAP", "CAPABLE", "CAPE", "CAPTAIN", "CAR", "CARBON", "CARD", "CARE",
    "CARGO", "CARPET", "CARRIAGE", "CARRY", "CARROT", "CART", "CARVE", "CASE", "CASH", "CASINO", "CASK",
    "CAST", "CASTLE", "CASUAL", "CAT", "CATCH", "CATER", "CATTLE", "CAUSE", "CAVE", "CEASE", "CEDAR", "CEILING",
    "CELERY", "CELL", "CEMENT", "CENSUS", "CENTER", "CENTRE", "CHAIN", "CHAIR", "CHALK", "CHAMP", "CHANCE",
    "CHANGE", "CHAOS", "CHAPEL", "CHARM", "CHART", "CHASE", "CHASM", "CHEAP", "CHEAT", "CHEESE", "CHEF",
    "CHEMIST", "CHERRY", "CHEST", "CHEW", "CHICK", "CHIEF", "CHILD", "CHILI", "CHILL", "CHIME", "CHIN",
    "CHINA", "CHIP", "CHOCOL", "CHOICE", "CHOKE", "CHORD", "CHORE", "CHOSE", "CHOSEN", "CHUCK", "CHUNK",
    "CHURCH", "CIDER", "CINEMA", "CIPHER", "CIRCLE", "CIRCUIT", "CIRCUS", "CITY", "CIVIC", "CLAIM", "CLAM",
    "CLAMP", "CLAN", "CLASH", "CLASP", "CLASS", "CLAUSE", "CLAW", "CLAY", "CLEAN", "CLEAR", "CLERK", "CLICK",
    "CLIFF", "CLIMB", "CLINIC", "CLIP", "CLOCK", "CLOSE", "CLOSET", "CLOUD", "CLOUT", "CLOVE", "CLOWN", "CLUB",
    "CLUE", "CLUMP", "COACH", "COAL", "COAST", "COAT", "COAX", "COBALT", "COBRA", "COCOA", "COCONUT", "CODE",
    "CODER", "COFFEE", "COIL", "COIN", "CONE", "CONFIRM", "CONGA", "CONIC", "CONN", "CONTOUR", "CONTRA",
    "CONTROL", "CONVEY", "COOK", "COOKIE", "COOL", "COOP", "COPPER", "COPY", "CORAL", "CORD", "CORE",
    "CORK", "CORN", "CORNER", "CORRAL", "COST", "COTTON", "COUCH", "COUGAR", "COUNT", "COUNTRY", "COUPLE",
    "COURSE", "COURT", "COVER", "COW", "COWBOY", "COYOTE", "CRAB", "CRACK", "CRAFT", "CRANE", "CRASH",
    "CRATE", "CRAWL", "CRAYON", "CRAZE", "CRAZY", "CREAM", "CREATE", "CREDIT", "CREEK", "CREEP", "CREST",
    "CREW", "CRIB", "CRICKET", "CRIME", "CRISP", "CROOK", "CROP", "CROSS", "CROWD", "CROWN", "CRUDE",
    "CRUEL", "CRUISE", "CRUMB", "CRUSH", "CRUST", "CRY", "CUBE", "CUBIC", "CUBS", "CUFF", "CUISINE",
    "CULT", "CULTURE", "CUP", "CURB", "CURE", "CURL", "CURRENCY", "CURRY", "CURSE", "CURVE", "CUSHION",
    "CUSTOM", "CUT", "CYCLE", "CYCLER", "CYCLIC", "CYLINDER", "DAD", "DAILY", "DAIRY", "DAISY", "DAMAGE",
    "DANCE", "DANCER", "DANGER", "DARING", "DARK", "DARLING", "DART", "DATA", "DATE", "DAWN", "DAY",
    "DEAL", "DEAR", "DEBRIS", "DEBT", "DEBUG", "DEBUT", "DECENT", "DECIDE", "DECK", "DECOR", "DEED",
    "DEEP", "DEER", "DEFEND", "DEFER", "DEFINE", "DEGREE", "DELAY", "DELTA", "DELVE", "DEMAND", "DEMN",
    "DEMO", "DENT", "DENTAL", "DENTIST", "DENY", "DEPART", "DEPEND", "DEPLOY", "DEPOSIT", "DEPTH",
    "DEPUTY", "DERBY", "DESERT", "DESIGN", "DESK", "DESPITE", "DETAIL", "DETOUR", "DEUCE", "DEVICE", "DEVIL",
    "DEVOTE", "DIARY", "DICE", "DIE", "DIET", "DIG", "DIGEST", "DIGIT", "DILUTE", "DIM", "DINER", "DINGO",
    "DINGY", "DIODE", "DIP", "DIPPER", "DIRECT", "DIRT", "DIRTY", "DISCO", "DISH", "DISK", "DIVE",
    "DIVER", "DIVIDE", "DIVING", "DIZZY", "DOCK", "DOCTOR", "DODGE", "DOG", "DOGMA", "DOLL", "DOLPHIN",
    "DOMAIN", "DOME", "DOMINO", "DONE", "DONKEY", "DONOR", "DOOR", "DOSE", "DOT", "DOUBLE", "DOUGH",
    "DOVE", "DOWN", "DOZEN", "DRAB", "DRAFT", "DRAGON", "DRAIN", "DRAMA", "DRANK", "DRAPE", "DRAW",
    "DRAWN", "DREAD", "DREAM", "DRESS", "DRIED", "DRIFT", "DRILL", "DRINK", "DRIVE", "DRIVER", "DROID",
    "DROOP", "DROP", "DROVE", "DROWN", "DRUG", "DRUM", "DRY", "DUCK", "DUCT", "DUKE", "DULL", "DUMB",
    "DUNE", "DUNK", "DUO", "DUSK", "DUST", "DUTY", "DWELL", "DWINDLE", "EACH", "EAGER", "EAGLE", "EAR",
    "EARLY", "EARN", "EARTH", "EASE", "EASILY", "EAST", "EASY", "EAT", "EBONY", "ECHO", "ECLIPSE", "ECO",
    "EDGE", "EDIT", "EDITOR", "EEL", "EFFECT", "EFFORT", "EGG", "EIGHT", "EITHER", "ELBOW", "ELDER",
    "ELECT", "ELEGANT", "ELEMENT", "ELEPHANT", "ELEVATE", "ELITE", "ELK", "ELM", "ELSE", "EMBED", "EMBER",
    "EMBRACE", "EMIT", "EMOTION", "EMPLOY", "EMPTY", "EMU", "ENABLE", "ENACT", "END", "ENDURE", "ENEMY",
    "ENERGY", "ENJOY", "ENLIST", "ENORMOUS", "ENOUGH", "ENROLL", "ENSURE", "ENTER", "ENTIRE", "ENTRY",
    "ENVOY", "ENZYME", "EPOCH", "EQUAL", "EQUIP", "ERA", "ERASE", "ERECT", "ERODE", "ERROR", "ESCAPE",
    "ESSAY", "ESTATE", "ETC", "ETHIC", "ETHOS", "EURO", "EVADE", "EVEN", "EVENT", "EVER", "EVICT", "EVIL",
    "EVOLVE", "EXACT", "EXAM", "EXCEL", "EXHALE", "EXIST", "EXIT", "EXOTIC", "EXPAND", "EXPECT", "EXPENSE",
    "EXPERT", "EXPIRE", "EXPORT", "EXPOSE", "EXTEND", "EXTRA", "EYE", "FABRIC", "FACE", "FACT", "FACTOR",
    "FACTORY", "FADE", "FAIL", "FAIR", "FAITH", "FAKE", "FALL", "FALSE", "FAME", "FAMOUS", "FAN", "FANCY",
    "FANG", "FARM", "FARMER", "FARMS", "FAST", "FATAL", "FATE", "FAULT", "FAUNA", "FAVOR", "FAX", "FEAST",
    "FEED", "FEEL", "FELT", "FENCE", "FERN", "FERRY", "FERTILE", "FEST", "FETCH", "FEVER", "FEW", "FIBER",
    "FICTION", "FIELD", "FIERCE", "FIFTH", "FIFTY", "FIG", "FIGHT", "FIGURE", "FILE", "FILL", "FILM", "FILTER",
    "FINAL", "FINCH", "FIND", "FINE", "FINISH", "FIRE", "FIRM", "FIRST", "FISH", "FISHER", "FIST", "FIT",
    "FIVE", "FIX", "FIXED", "FLAG", "FLAKY", "FLAME", "FLANK", "FLASH", "FLAT", "FLAW", "FLEA", "FLEET",
    "FLESH", "FLICK", "FLIER", "FLIGHT", "FLING", "FLINT", "FLIP", "FLOAT", "FLOCK", "FLOOD", "FLOOR", "FLOUR",
    "FLOW", "FLOWER", "FLU", "FLUFF", "FLUID", "FLUTE", "FLUX", "FLY", "FOAM", "FOCUS", "FOG", "FOGGY",
    "FOOD", "FOOL", "FOOT", "FORCE", "FORD", "FOREST", "FORGET", "FORK", "FORM", "FORMAT", "FORT", "FORUM",
    "FOSSIL", "FOUND", "FOUR", "FOX", "FRAME", "FRANK", "FRAUD", "FRESH", "FRIAR", "FRIDAY", "FRIED",
    "FRIEND", "FRIES", "FROG", "FROM", "FRONT", "FROST", "FROZE", "FRUIT", "FUEL", "FULL", "FUN", "FUNCTION",
    "FUND", "FUNNY", "FUR", "FUSION", "FUTURE", "GADGET", "GALE", "GALLON", "GAMBLE", "GAME", "GAMER",
    "GAMMA", "GANG", "GARDEN", "GARLIC", "GASH", "GATE", "GATHER", "GAUGE", "GAUNT", "GAZE", "GEAR", "GECKO",
    "GEL", "GEM", "GENE", "GENERIC", "GENIUS", "GENRE", "GENTLE", "GENTLY", "GENUS", "GERM", "GET", "GHOST",
    "GIANT", "GIFT", "GIG", "GIGGLE", "GIRAFFE", "GIRL", "GIST", "GIVE", "GLACIER", "GLAD", "GLANCE", "GLASS",
    "GLAZE", "GLEAM", "GLIDE", "GLINT", "GLOBE", "GLOOM", "GLORY", "GLOVE", "GLOW", "GLUE", "GOAL", "GOAT",
    "GOBLIN", "GOD", "GODS", "GOING", "GOLD", "GOLF", "GONDOLA", "GONE", "GOOD", "GOOSE", "GOPHER", "GORILLA",
    "GOSPEL", "GOT", "GOURD", "GOWN", "GRACE", "GRADE", "GRAIN", "GRAND", "GRANT", "GRAPE", "GRAPH", "GRASP",
    "GRASS", "GRATE", "GRAVEL", "GRAVY", "GREAT", "GREED", "GREEN", "GREET", "GREY", "GRID", "GRIEF", "GRILL",
    "GRIN", "GRIND", "GRIP", "GRIT", "GROOM", "GROSS", "GROUP", "GROVE", "GROW", "GROWN", "GRUB", "GRUNT",
    "GUARD", "GUAVA", "GUESS", "GUEST", "GUIDE", "GUILD", "GUILT", "GUITAR", "GULL", "GULP", "GUM", "GUMBO",
    "GUST", "GUT", "GUY", "GYM", "GYPSY", "HABIT", "HACK", "HAIL", "HAIR", "HALF", "HALL", "HALO", "HALT",
    "HAM", "HAMLET", "HAND", "HANDLE", "HANG", "HARBOR", "HARD", "HARE", "HARM", "HARP", "HARSH", "HARVEST",
    "HAS", "HASH", "HASTE", "HAT", "HATCH", "HATE", "HAVE", "HAWK", "HAY", "HAZEL", "HAZY", "HEAD", "HEAL",
    "HEAP", "HEAR", "HEARD", "HEART", "HEAT", "HEAVY", "HEDGE", "HEEL", "HEFT", "HEIGHT", "HEIR", "HELD",
    "HELIX", "HELLO", "HELM", "HELMET", "HELP", "HEMP", "HEN", "HERB", "HERD", "HERON", "HERO", "HESSIAN",
    "HIDDEN", "HIDE", "HIGH", "HIKER", "HILL", "HILT", "HIND", "HINGE", "HINT", "HIP", "HIRE", "HIS", "HIT",
    "HIVE", "HOBBY", "HOG", "HOLD", "HOLE", "HOLIDAY", "HOLLY", "HOME", "HONEST", "HONEY", "HONOR", "HOOD",
    "HOOK", "HOOP", "HOP", "HOPPER", "HORN", "HORNET", "HORSE", "HOSE", "HOST", "HOT", "HOTEL", "HOUR", "HOUSE",
    "HOVER", "HUGE", "HUMAN", "HUMBLE", "HUMID", "HUMOR", "HUNGER", "HUNT", "HURRY", "HURT", "HUSK", "HUSKY",
    "HUT", "HYBRID", "HYDRA", "HYENA", "HYMNS", "ICE", "ICING", "ICON", "IDEA", "IDEAL", "IDENTITY", "IDLE",
    "IDOL", "IGLOO", "IGNITE", "IGUANA", "ILL", "IMAGE", "IMPACT", "IMPORT", "IN", "INCH", "INDEX", "INDIGO",
    "INDOOR", "INEPT", "INFER", "INFO", "INGOT", "INJECT", "INK", "INLAY", "INLET", "INN", "INNER", "INPUT",
    "INSIDE", "INSIGHT", "INSPIRE", "INSTANCE", "INTAKE", "INTENT", "INTERN", "INTO", "INVENT", "INVEST",
    "ION", "IRON", "ISLAND", "IVORY", "IVY", "JACK", "JACKET", "JAIL", "JAM", "JAR", "JASMINE", "JAW", "JAZZ",
    "JEANS", "JEEP", "JELLY", "JERKY", "JET", "JEWEL", "JIG", "JINX", "JOB", "JOG", "JOIN", "JOINT", "JOKE",
    "JOLLY", "JOURNAL", "JOY", "JUICE", "JUICY", "JULY", "JUMP", "JUMPER", "JUNE", "JUNGLE", "JUNIOR", "JUNK",
    "JUROR", "JUST", "KAYAK", "KEEN", "KEEP", "KELP", "KERNEL", "KETTLE", "KEY", "KEYBOARD", "KICK", "KID",
    "KIDNEY", "KILO", "KIND", "KING", "KIOSK", "KISS", "KIT", "KITE", "KITTEN", "KIWI", "KNEAD", "KNEE",
    "KNIFE", "KNIT", "KNOB", "KNOT", "KNOW", "KOALA", "LABEL", "LABOR", "LACE", "LACK", "LADDER", "LADLE",
    "LADY", "LAGOON", "LAKE", "LAMB", "LAMP", "LANCE", "LAND", "LANE", "LAP", "LAPTOP", "LARGE", "LARK",
    "LASER", "LAST", "LATCH", "LATE", "LAUGH", "LAUNCH", "LAVA", "LAWN", "LAWSUIT", "LAWYER", "LAYER", "LAZY",
    "LEAF", "LEAGUE", "LEAK", "LEAN", "LEARN", "LEASE", "LEASH", "LEAVE", "LED", "LEFT", "LEG", "LEGAL",
    "LEGEND", "LEMON", "LEND", "LENS", "LEOPARD", "LESS", "LETTUCE", "LEVEL", "LEVER", "LIAR", "LID", "LIE",
    "LIFE", "LIFT", "LIGHT", "LIKE", "LILAC", "LILY", "LIMB", "LIME", "LIMIT", "LINE", "LINK", "LION",
    "LIP", "LIQUID", "LIST", "LIT", "LITER", "LITTLE", "LIVE", "LIVER", "LIZARD", "LOAD", "LOAF", "LOAN",
    "LOBSTER", "LOCAL", "LOCK", "LODGE", "LOFT", "LOG", "LOOM", "LOON", "LOOP", "LOOSE", "LOOT", "LORD",
    "LORRY", "LOSE", "LOSS", "LOST", "LOTION", "LOTUS", "LOUD", "LOUNGE", "LOVE", "LOW", "LOYAL", "LUCK",
    "LUCID", "LUMP", "LUNCH", "LUNGE", "LUSH", "LUST", "LUTE", "LUXURY", "LYCHEE", "LYRIC", "MAGENTA", "MAGIC",
    "MAID", "MAIL", "MAJOR", "MAKER", "MALE", "MALL", "MAMMAL", "MAN", "MANGO", "MANTIS", "MAP", "MAPLE",
    "MARBLE", "MARCH", "MARE", "MARGIN", "MARINE", "MARK", "MARKET", "MARROW", "MARRY", "MARS", "MART",
    "MASK", "MASS", "MASTER", "MATCH", "MATE", "MATH", "MATRIX", "MAY", "MAYOR", "MAZE", "MEADOW", "MEAL",
    "MEAN", "MEASURE", "MEAT", "MECHANIC", "MEDAL", "MEDIA", "MEDIC", "MEET", "MELON", "MELT", "MEMBER",
    "MEMORY", "MEND", "MENU", "MERCY", "MERGE", "MERIT", "MERRY", "MESS", "METAL", "METER", "METRO", "MICRO",
    "MIDDLE", "MIGHT", "MILD", "MILE", "MILK", "MILL", "MIMIC", "MINCE", "MIND", "MINER", "MINI", "MINOR",
    "MINT", "MINUTE", "MIRROR", "MIRTH", "MIX", "MIXER", "MIXED", "MIXTURE", "MOBILE", "MODEL", "MODEM",
    "MODERN", "MODIFY", "MODULE", "MOIST", "MOLD", "MOLAR", "MOM", "MONITOR", "MONKEY", "MONTH", "MOOD",
    "MOON", "MOOR", "MOOSE", "MOP", "MORAL", "MORE", "MORNING", "MORPH", "MOSS", "MOST", "MOTH", "MOTHER",
    "MOTION", "MOTOR", "MOTTO", "MOUND", "MOUNT", "MOURN", "MOUSE", "MOUTH", "MOVE", "MOVIE", "MUFFIN",
    "MULE", "MUSE", "MUSIC", "MUSK", "MUSSEL", "MUST", "MUTE", "MUTTER", "MUTTON", "NACHO", "NAIL", "NAME",
    "NAPKIN", "NARROW", "NASTY", "NATION", "NATIVE", "NATURE", "NAVY", "NEAR", "NEAT", "NECK", "NEED", "NEON",
    "NERVE", "NEST", "NET", "NETWORK", "NEURAL", "NEATLY", "NEVER", "NEW", "NEWS", "NEXT", "NICE", "NICKEL",
    "NIECE", "NIGHT", "NINE", "NINJA", "NINTH", "NOBLE", "NOD", "NOISE", "NOISY", "NONE", "NOODLE", "NORTH",
    "NOSE", "NOTCH", "NOTE", "NOTIFY", "NOTION", "NOVEL", "NOVICE", "NOW", "NUANCE", "NUCLEAR", "NUDGE",
    "NULL", "NUMBER", "NUMB", "NURSE", "NUT", "NYLON", "OAK", "OAR", "OASIS", "OAT", "OATH", "OBEY", "OBJECT",
    "OBLIGE", "OCEAN", "OCTAVE", "OCTET", "ODD", "ODOR", "OFF", "OFFER", "OFFICE", "OFFSET", "OFTEN", "OIL",
    "OINK", "OLD", "OLIVE", "OMEGA", "OMELET", "OMEN", "OMICRON", "ONCE", "ONE", "ONION", "ONLINE", "ONLY",
    "ONTO", "ONYX", "OPEN", "OPERA", "OPINE", "OPTIC", "OPTION", "ORANGE", "ORBIT", "ORCHID", "ORDER",
    "ORE", "ORGAN", "ORIGIN", "ORION", "ORNATE", "ORPHAN", "OSPREY", "OSTRICH", "OTHER", "OTTER", "OUNCE",
    "OUR", "OUT", "OVAL", "OVEN", "OVER", "OWL", "OX", "OYSTER", "OZONE", "PACE", "PACK", "PAD", "PADDLE",
    "PAGE", "PAID", "PAIL", "PAIN", "PAINT", "PAIR", "PALACE", "PALE", "PALM", "PANDA", "PANEL", "PANTRY",
    "PANTS", "PAPA", "PAPER", "PARADE", "PARCEL", "PARDON", "PARENT", "PARK", "PARROT", "PART", "PARTY",
    "PASTA", "PASTE", "PASTOR", "PASTRY", "PATCH", "PATH", "PATIENT", "PATROL", "PATIO", "PATSY", "PAUSE",
    "PAY", "PEACE", "PEACH", "PEAR", "PEARL", "PEAS", "PECK", "PEDAL", "PEEL", "PEEP", "PEER", "PELICAN",
    "PEN", "PENALTY", "PENCIL", "PEND", "PENGUIN", "PENNY", "PEOPLE", "PEPPER", "PERCH", "PERFECT", "PERIL",
    "PERK", "PERMIT", "PERSON", "PEST", "PET", "PHASE", "PHONE", "PHOTO", "PHYSIC", "PICK", "PICKUP",
    "PICNIC", "PIE", "PIECE", "PIER", "PIGEON", "PIGMENT", "PILOT", "PIN", "PINCH", "PINE", "PING", "PINK",
    "PINT", "PIONEER", "PIPE", "PIPELINE", "PIRATE", "PISTOL", "PITCH", "PIVOT", "PIXEL", "PIZZA", "PLACE",
    "PLAID", "PLAIN", "PLAN", "PLANE", "PLANET", "PLANT", "PLASMA", "PLATE", "PLAY", "PLAYER", "PLEAD",
    "PLEASE", "PLEDGE", "PLENTY", "PLIERS", "PLIGHT", "PLOD", "PLOW", "PLUCK", "PLUG", "PLUMB", "PLUME",
    "PLUMP", "PLUM",
)

_raw_max = max(len(w) for w in _RAW_WORDS)
_raw_encoded = jnp.array([
    [ord(c) - 65 for c in w.upper()] + [CONSTANTS.PAD_TOKEN] * (_raw_max - len(w))
    for w in _RAW_WORDS
], dtype=jnp.int32)

_raw_lens = jnp.sum(_raw_encoded != CONSTANTS.PAD_TOKEN, axis=1)
_valid_mask = _raw_lens <= CONSTANTS.L_MAX
_filtered_enc = _raw_encoded[_valid_mask]

WORDS_ENC = _filtered_enc[:, :CONSTANTS.L_MAX]
WORDS_LEN = _raw_lens[_valid_mask]
N_WORDS = WORDS_ENC.shape[0]