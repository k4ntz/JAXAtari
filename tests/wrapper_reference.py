"""Frozen scan/cond reference for wrapper compatibility tests.

Reset and step methods from the pre-optimization implementation. Constructors
and unchanged processing helpers are inherited; no git checkout is needed at
runtime. Keep these methods independent of the optimized control flow.
"""
import functools
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Tuple
import chex
import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtari.environment import JaxtariAction as Action
from jaxtari import wrappers as current
from jaxtari.wrappers import (
    AtariState, ObjectCentricState, PixelState, PixelAndObjectCentricState,
    preprocess_image,
)

class AtariWrapper(current.AtariWrapper):
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, AtariState]:
        # Split keys for all potential random operations
        env_key, wrapper_key, noop_key = jax.random.split(key, 3)
        obs, env_state = self._env.reset(env_key)
        step = jnp.array(0, dtype=jnp.int32)
        prev_action = jnp.array(0, dtype=jnp.int32)

        # ========== NOOP RESET ==========
        def perform_noop_reset(carry):
            # This function will be executed if self.noop_reset is True
            env_state, obs, step = carry
            # Generate the random number of no-op steps to take.
            num_noops = jax.random.randint(noop_key, shape=(), minval=0, maxval=self.noop_max + 1)

            def noop_body_fn(i, loop_carry):
                current_obs, current_env_state = loop_carry
                # We always compute the next step for static graph tracing...
                next_obs, next_env_state, reward, done, info = self._env.step(current_env_state, Action.NOOP)
                # Might be done from the no-op step, in which case we reset.
                next_obs, next_env_state = jax.lax.cond(
                    done,
                    lambda: self._env.reset(env_key),
                    lambda: (next_obs, next_env_state)
                )
                return next_obs, next_env_state

            # Loop for the static maximum number of no-ops.
            final_obs, final_env_state = jax.lax.fori_loop(0, num_noops, noop_body_fn, (obs, env_state))

            # Update the step counter by the dynamic number of no-ops performed.
            final_step = step + num_noops
            return final_env_state, final_obs, final_step

        # Use lax.cond to conditionally apply the whole no-op block based on the static self.noop_reset flag.
        env_state, obs, step = jax.lax.cond(
            self.noop_reset,
            lambda carry: perform_noop_reset(carry),
            lambda carry: carry,
            (env_state, obs, step)
        )

        # ========== FIRST FIRE ==========
        def perform_first_fire(carry):
            env_state, obs, step, _ = carry
            fire_obs, fire_env_state, _, _, _ = self._env.step(env_state, self.fire_action_index)
            return fire_env_state, fire_obs, step + 1, self.fire_action_index

        def identity_fire(carry):
            return carry

        # Conditionally apply the fire action based on the static self.first_fire flag.
        env_state, obs, step, prev_action = jax.lax.cond(
            self.first_fire,
            perform_first_fire,
            identity_fire,
            (env_state, obs, step, prev_action)
        )

        return obs, AtariState(env_state, wrapper_key, step, prev_action)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtariState, action: int) -> Tuple[chex.Array, AtariState, float, bool, bool, Dict[Any, Any]]:
        step_key, next_state_key = jax.random.split(state.key)

        use_sticky_action = jax.random.uniform(step_key, shape=()) < self.sticky_actions
        new_action = jnp.where(use_sticky_action, state.prev_action, action)
        obs, new_env_state, reward, env_done, infos = self._env.step(state.env_state, new_action)

        terminated = env_done
        if self.episodic_life:
            # If the player has lost a life, we consider the episode done
            if hasattr(state.env_state, "lives"):
                condition = jnp.logical_and(state.env_state.lives > 0, new_env_state.lives < state.env_state.lives)
                terminated = jnp.logical_or(terminated, condition)
            elif hasattr(state.env_state, "lives_lost"):
                terminated = jnp.logical_or(terminated, new_env_state.lives_lost > state.env_state.lives_lost)

        if hasattr(infos, '_asdict'):
            # It's a namedtuple or similar, convert to dict
            info_items = infos._asdict().items()
        elif is_dataclass(infos):
            # It's a dataclass, convert to dict
            info_items = asdict(infos).items()
        else:
            # It's already a dict
            info_items = infos.items()

        info_dict = {k: v for k, v in info_items}

        next_state = AtariState(new_env_state, next_state_key, state.step + 1, new_action)

        # store actual done - not affected by episodic life
        info_dict["env_done"] = env_done

        # store actual reward in info dict before clipping
        info_dict["env_reward"] = reward

        truncated = (state.step + 1 >= self.max_frames_per_episode)
        return obs, next_state, reward, terminated, truncated, info_dict

class ObjectCentricWrapper(current.ObjectCentricWrapper):
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, ObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        flat_obs = self._flatten_single_obs(obs)
        obs_stack = jnp.stack([flat_obs] * self.frame_stack_size)
        return obs_stack, ObjectCentricState(atari_state, obs_stack)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: ObjectCentricState,
        action: int,
    ) -> Tuple[chex.Array, ObjectCentricState, float, bool, bool, Dict[Any, Any]]:

        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, reward, terminated, truncated, info)

        (atari_state, _), (obs, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        flat_latest_obs = self._flatten_single_obs(latest_obs)
        obs_stack = jnp.concatenate([state.obs_stack[1:], jnp.expand_dims(flat_latest_obs, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)

        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        obs_stack, oc_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),  # reset if done, using the current key for proper random state advancement
            lambda: (obs_stack, ObjectCentricState(atari_state, obs_stack)),  # step if not done
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return obs_stack, oc_state, reward, terminated, truncated, info_dict

class PixelObsWrapper(current.PixelObsWrapper):
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, PixelState]:
        _, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)

        image_stack = jnp.stack([processed_image] * self.frame_stack_size)
        return image_stack, PixelState(atari_state, image_stack)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelState,
        action: int,
    ) -> Tuple[chex.Array, PixelState, float, bool, bool, Dict[Any, Any]]:

        def body_fn(carry, _):
            atari_state, action = carry
            _, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)
        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        image_stack, pixel_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),
            lambda: (image_stack, PixelState(atari_state, image_stack))
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return image_stack, pixel_state, reward, terminated, truncated, info_dict

class PixelAndObjectCentricWrapper(current.PixelAndObjectCentricWrapper):
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        flat_obs = self._flatten_single_obs(obs)
        obs_stack = jnp.stack([flat_obs] * self.frame_stack_size)

        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)
        image_stack = jnp.stack([processed_image] * self.frame_stack_size)

        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: int,
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState, float, bool, bool, Dict[Any, Any]]:
        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (obs, env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        flat_latest_obs = self._flatten_single_obs(latest_obs)
        obs_stack = jnp.concatenate([state.obs_stack[1:], jnp.expand_dims(flat_latest_obs, axis=0)], axis=0)

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)

        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        (image_stack, obs_stack), pixel_oc_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),
            lambda: ((image_stack, obs_stack), PixelAndObjectCentricState(atari_state, image_stack, obs_stack))
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return (image_stack, obs_stack), pixel_oc_state, reward, terminated, truncated, info_dict

class PixelAndObjectObsWrapper(current.PixelAndObjectObsWrapper):
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[Tuple[chex.Array, Any], PixelAndObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)
        image_stack = jnp.stack([processed_image] * self.frame_stack_size)

        obs_stack = jax.tree.map(
            lambda leaf: jnp.stack([leaf] * self.frame_stack_size),
            obs,
        )

        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: int,
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState, float, bool, bool, Dict[Any, Any]]:
        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (obs, env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        obs_stack = jax.tree.map(
            lambda stack_leaf, obs_leaf: jnp.concatenate([stack_leaf[1:], jnp.expand_dims(obs_leaf, axis=0)], axis=0),
            state.obs_stack,
            latest_obs,
        )

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)

        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        (image_stack, obs_stack), pixel_oc_state = jax.lax.cond(
            jnp.logical_or(infos["env_done"].any(), truncated),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),
            lambda: ((image_stack, obs_stack), PixelAndObjectCentricState(atari_state, image_stack, obs_stack)),
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), pixel_oc_state, reward, terminated, truncated, info_dict

