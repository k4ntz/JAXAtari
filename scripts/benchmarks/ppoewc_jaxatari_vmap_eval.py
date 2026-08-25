import jax
import jax.numpy as jnp

def make_eval_step_fns(env, network, actor, object_centric: bool, padding_width: int, capture_video: bool, max_steps: int = 10000):
    """Creates and JIT-compiles the evaluation loops for a specific environment."""
    
    @jax.jit
    def compiled_reset(keys):
        def single_reset(key):
            next_obs, state = env.reset(key)
            next_obs = next_obs.squeeze()[None, ...]
            if object_centric:
                next_obs = jnp.pad(next_obs, ((0, 0), (0, padding_width)))
            return next_obs, state
        return jax.vmap(single_reset)(keys)

    @jax.jit
    def compiled_rollout(network_params, actor_params, init_obs, init_env_states, init_keys):
        def get_action_and_value(obs, key):
            hidden = network.apply(network_params, obs)
            logits = actor.apply(actor_params, hidden)
            # sample action: Gumbel-softmax trick
            # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey, shape=logits.shape)
            action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=-1)
            return action, key

        def wrapped_step(state, action):
            next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
            done = jnp.logical_or(terminated, truncated)
            next_obs = next_obs.squeeze()[None, ...]
            if object_centric:
                next_obs = jnp.pad(next_obs, ((0, 0), (0, padding_width)))
            return next_obs, next_state, reward, done, info

        def step_fn(carry, unused):
            next_obs, env_state, keys = carry
            actions, keys = jax.vmap(get_action_and_value, in_axes=(0, 0))(next_obs, keys)
            next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, actions)

            if capture_video:
                first_states = jax.tree.map(lambda x: x[0], env_state)
            else:
                first_states = ()
            
            return (next_obs, env_state, keys), (first_states, done, reward, actions)

        _, (first_states, dones, rewards, actions) = jax.lax.scan(
            step_fn, (init_obs, init_env_states, init_keys), None, length=max_steps
        )
        return first_states, dones, rewards, actions

    return compiled_reset, compiled_rollout


def evaluate_cached(
    compiled_reset,
    compiled_rollout,
    network_params,
    actor_params,
    eval_episodes: int,
    capture_video: bool,
    seed: int = 1,
):
    """Runs evaluation using pre-compiled functions and pre-loaded parameters."""
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, eval_episodes)
    
    next_obs, env_states = compiled_reset(reset_keys)
    step_keys = jax.random.split(key, eval_episodes)
    
    first_states, dones, rewards, actions = compiled_rollout(
        network_params, actor_params, next_obs, env_states, step_keys
    )

    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1,0),(0,0)), constant_values=0)
    rewards = rewards * (1 - mask_after_first_done)
    
    episodic_returns = jnp.sum(rewards, axis=0)

    env_states_until_done = None
    if capture_video:
        is_done = bool(jnp.any(dones[:, 0]).item())
        end_idx = int(first_done[0].item()) if is_done else (dones.shape[0] - 1)
        env_states_until_done = jax.tree.map(lambda x: x[:end_idx + 1], first_states.atari_state.atari_state.env_state)

    return episodic_returns, env_states_until_done