import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from functools import partial
from typing import NamedTuple, Tuple, Optional

import chex
import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer

#constants
WIDTH, HEIGHT = 160, 210  
ALPHABET_SIZE = 26
PAD_TOKEN = 26
L_MAX = 8

MAX_MISSES = 11
STEP_PENALTY = 0.0

_WORDS = ["CAT", "TREE", "MOUSE", "ROBOT", "LASER", "JAX"]

def _encode_word(w: str) -> jnp.ndarray:
    arr = [ord(c) - 65 for c in w.upper()]
    arr = arr[:L_MAX]
    arr += [PAD_TOKEN] * (L_MAX - len(arr))
    return jnp.array(arr, dtype=jnp.int32)

WORDS_ENC = jnp.stack([_encode_word(w) for w in _WORDS], axis=0)
WORDS_LEN = jnp.array([min(len(w), L_MAX) for w in _WORDS], dtype=jnp.int32)
N_WORDS = WORDS_ENC.shape[0]

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

class HangmanObservation(NamedTuple):
    revealed: chex.Array      
    mask: chex.Array          
    guessed: chex.Array       
    misses: chex.Array        
    lives: chex.Array         
    cursor_idx: chex.Array    

class HangmanInfo(NamedTuple):
    time: chex.Array
    all_rewards: chex.Array

# helpers funcutions
@jax.jit
def _sample_word(key: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    key, sub = jrandom.split(key)
    idx = jrandom.randint(sub, shape=(), minval=0, maxval=N_WORDS, dtype=jnp.int32)
    return key, WORDS_ENC[idx], WORDS_LEN[idx]

@jax.jit
def _compute_revealed(word: chex.Array, mask: chex.Array) -> chex.Array:
    return jnp.where(mask.astype(bool), word, PAD_TOKEN)

def _action_delta_cursor(action: chex.Array) -> chex.Array:
    up_like = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
    down_like = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)
    return jnp.where(up_like, -1, jnp.where(down_like, 1, 0)).astype(jnp.int32)

def _action_commit(action: chex.Array) -> chex.Array:
    return jnp.logical_or(
        jnp.logical_or(action == Action.FIRE, action == Action.UPFIRE),
        action == Action.DOWNFIRE
    )

# render
class HangmanRenderer(AtraJaxisRenderer):
    def __init__(self):
        self.BG = jnp.array([144, 72, 17], dtype=jnp.uint8)  # reddish

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HangmanState) -> jnp.ndarray:
        return jnp.broadcast_to(self.BG, (WIDTH, HEIGHT, 3))


#environment

class JaxHangman(JaxEnvironment[HangmanState, HangmanObservation, HangmanInfo]):
    def __init__(self, reward_funcs: Optional[list] = None, *, max_misses: int = MAX_MISSES, step_penalty: float = STEP_PENALTY):
        super().__init__()
        self.renderer = HangmanRenderer()
        self.max_misses = int(max_misses)
        self.step_penalty = float(step_penalty)

        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.DOWN,
            Action.UPFIRE,
            Action.DOWNFIRE,
        ]
        self.obs_size = L_MAX + L_MAX + ALPHABET_SIZE + 3
        self.reward_funcs = tuple(reward_funcs) if reward_funcs is not None else None

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[HangmanObservation, HangmanState]:
        if key is None:
            key = jrandom.PRNGKey(0)
        key, word, length = _sample_word(key)
        state = HangmanState(
            key=key,
            word=word,
            length=length,
            mask=jnp.zeros((L_MAX,), dtype=jnp.int32),
            guessed=jnp.zeros((ALPHABET_SIZE,), dtype=jnp.int32),
            misses=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.max_misses, dtype=jnp.int32),
            cursor_idx=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            reward=jnp.array(0.0, dtype=jnp.float32),
            step_counter=jnp.array(0, dtype=jnp.int32),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HangmanState, action: chex.Array) -> Tuple[HangmanObservation, HangmanState, float, bool, HangmanInfo]:
        commit = _action_commit(action)
        delta = _action_delta_cursor(action)

        def _new_round(s: HangmanState) -> HangmanState:
            key, word, length = _sample_word(s.key)
            return HangmanState(
                key=key, word=word, length=length,
                mask=jnp.zeros((L_MAX,), dtype=jnp.int32),
                guessed=jnp.zeros((ALPHABET_SIZE,), dtype=jnp.int32),
                misses=jnp.array(0, dtype=jnp.int32),
                lives=jnp.array(self.max_misses, dtype=jnp.int32),
                cursor_idx=jnp.array(0, dtype=jnp.int32),
                done=jnp.array(False),
                reward=jnp.array(0.0, dtype=jnp.float32),
                step_counter=jnp.array(0, dtype=jnp.int32),
            )

        def _continue_round(s: HangmanState) -> HangmanState:
            cursor = (s.cursor_idx + delta) % ALPHABET_SIZE

            def on_commit(s2: HangmanState) -> HangmanState:
                already = s2.guessed[cursor] == 1
                guessed = s2.guessed.at[cursor].set(1)

                pos_hits = (s2.word == cursor).astype(jnp.int32)
                pos_hits = pos_hits * (jnp.arange(L_MAX, dtype=jnp.int32) < s2.length).astype(jnp.int32)

                any_hit = jnp.any(pos_hits == 1)
                mask = jnp.where(pos_hits.astype(bool), 1, s2.mask)

                wrong = jnp.logical_and(jnp.logical_not(any_hit), jnp.logical_not(already))
                misses = s2.misses + wrong.astype(jnp.int32)
                lives  = s2.lives  - wrong.astype(jnp.int32)

                idx = jnp.arange(L_MAX, dtype=jnp.int32)
                within = idx < s2.length  # True only for the active letters
                all_revealed = jnp.all(jnp.where(within, mask == 1, True))

                lost = misses >= self.max_misses

                step_reward = jnp.where(all_revealed, 1.0, jnp.where(lost, -1.0, self.step_penalty)).astype(jnp.float32)

                return HangmanState(
                    key=s2.key, word=s2.word, length=s2.length,
                    mask=mask, guessed=guessed, misses=misses, lives=lives,
                    cursor_idx=cursor,
                    done=jnp.logical_or(all_revealed, lost),
                    reward=step_reward,
                    step_counter=s2.step_counter + 1,
                )

            def no_commit(s2: HangmanState) -> HangmanState:
                return HangmanState(
                    key=s2.key, word=s2.word, length=s2.length,
                    mask=s2.mask, guessed=s2.guessed, misses=s2.misses, lives=s2.lives,
                    cursor_idx=cursor,
                    done=s2.done,
                    reward=jnp.array(self.step_penalty, dtype=jnp.float32),
                    step_counter=s2.step_counter + 1,
                )

            return lax.cond(commit, on_commit, no_commit, s)

        next_state = lax.cond(jnp.logical_and(state.done, commit), _new_round, _continue_round, state)

        done = self._get_done(next_state)
        env_reward = self._get_env_reward(state, next_state)
        all_rewards = self._get_all_reward(state, next_state)
        obs = self._get_observation(next_state)
        info = self._get_info(next_state, all_rewards)
        return obs, next_state, env_reward, done, info

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "revealed": spaces.Box(low=0, high=PAD_TOKEN, shape=(L_MAX,), dtype=jnp.int32),
            "mask":     spaces.Box(low=0, high=1,          shape=(L_MAX,), dtype=jnp.int32),
            "guessed":  spaces.Box(low=0, high=1,          shape=(ALPHABET_SIZE,), dtype=jnp.int32),
            "misses":   spaces.Box(low=0, high=self.max_misses, shape=(), dtype=jnp.int32),
            "lives":    spaces.Box(low=0, high=self.max_misses, shape=(), dtype=jnp.int32),
            "cursor_idx": spaces.Box(low=0, high=ALPHABET_SIZE-1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT, 3), dtype=jnp.uint8)

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
    def _get_env_reward(self, previous_state: HangmanState, state: HangmanState):
        return state.reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: HangmanState, state: HangmanState):
        if self.reward_funcs is None:
            return jnp.zeros(1, dtype=jnp.float32)
        rewards = jnp.array([rf(previous_state, state) for rf in self.reward_funcs], dtype=jnp.float32)
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: HangmanState) -> bool:
        return state.done

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HangmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: HangmanState, all_rewards: chex.Array) -> HangmanInfo:
        return HangmanInfo(time=state.step_counter, all_rewards=all_rewards)
