import jaxatari
import jaxatari.wrappers as wrappers
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import functools
import jax.tree_util as tree

def run_jitted_benchmark(env, num_episodes: int):
    """A JAX-idiomatic benchmark function."""
    key = jax.random.PRNGKey(0)
    step_fn = env.step

    @functools.partial(jax.jit, static_argnums=1)
    def run_episodes(initial_state_and_key, num_steps):

        def _scan_body(carry, _):
            state, key = carry # Unpack state and key
            
            # Split the key for this step's action
            key, action_key = jax.random.split(key)
            action = env.action_space().sample(action_key)
            
            _, next_state, _, _, _ = step_fn(state, action)
            
            return (next_state, key), None # Pass the new state AND the new key to the next iteration

        # Pass the initial state and an initial key into the scan
        (final_state, final_key), _ = jax.lax.scan(
            _scan_body,
            initial_state_and_key,
            None,
            length=num_steps
        )
        return final_state

    initial_obs, initial_state = env.reset(key)
    
    # --- WARM-UP RUN ---
    final_state = run_episodes((initial_state, key), num_episodes)
    
    # --- FIX STARTS HERE ---
    # Generalize the blocking call to work for any state object (pytree).
    # We get all array "leaves" of the state object and block on the first one.
    leaves = tree.tree_leaves(final_state)
    if leaves:
        leaves[0].block_until_ready()
    # --- FIX ENDS HERE ---

    # --- TIMED RUN ---
    start_time = time.time()
    final_state = run_episodes((initial_state, key), num_episodes)
    
    # --- Apply the same fix for the timed run ---
    leaves = tree.tree_leaves(final_state)
    if leaves:
        leaves[0].block_until_ready()
    
    end_time = time.time()
    return (end_time - start_time) / num_episodes


def get_unwrapped_env_time(env_name: str, num_episodes: int = 10000) -> float:
    env = jaxatari.make(env_name)
    return run_jitted_benchmark(env, num_episodes)


def get_standard_wrapper_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env)
    env = wrappers.ObjectCentricWrapper(env)
    env = wrappers.FlattenObservationWrapper(env)
    return run_jitted_benchmark(env, num_episodes)


def get_rendered_wrapper_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env)
    env = wrappers.PixelAndObjectCentricWrapper(env)
    env = wrappers.FlattenObservationWrapper(env)
    return run_jitted_benchmark(env, num_episodes)

def get_logging_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env)
    env = wrappers.ObjectCentricWrapper(env)
    env = wrappers.FlattenObservationWrapper(env)
    env = wrappers.LogWrapper(env)
    return run_jitted_benchmark(env, num_episodes)

def get_atari_wrapper_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env)
    return run_jitted_benchmark(env, num_episodes)


def get_object_centric_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env)
    env = wrappers.ObjectCentricWrapper(env)
    return run_jitted_benchmark(env, num_episodes)


def get_flattened_object_centric_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env)
    env = wrappers.ObjectCentricWrapper(env)
    env = wrappers.FlattenObservationWrapper(env)
    return run_jitted_benchmark(env, num_episodes)


def get_pixel_object_centric_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env)
    env = wrappers.PixelAndObjectCentricWrapper(env)
    return run_jitted_benchmark(env, num_episodes)


def get_flattened_pixel_object_centric_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env)
    env = wrappers.PixelAndObjectCentricWrapper(env)
    env = wrappers.FlattenObservationWrapper(env)
    return run_jitted_benchmark(env, num_episodes)

def get_atari_wrapper_no_frameskip_time(env_name: str, num_episodes: int = 10000) -> float:
    base_env = jaxatari.make(env_name)
    env = wrappers.AtariWrapper(base_env, frame_skip=1)
    return run_jitted_benchmark(env, num_episodes)

def visualize_wrapper_performances(times: dict) -> None:
    labels = list(times.keys())
    values = list(times.values())
    plt.bar(labels, values)
    plt.ylabel('Time per episode (s)')
    plt.title('Wrapper Performance Comparison')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_episodes = 1000
    env_name = "pong"

    times = {}
    times["Unwrapped"] = get_unwrapped_env_time(env_name, num_episodes)
    times["AtariWrapper"] = get_atari_wrapper_time(env_name, num_episodes)
    times["Atari+ObjectCentric"] = get_object_centric_time(env_name, num_episodes)
    times["Atari+ObjectCentric+Flattened"] = get_flattened_object_centric_time(env_name, num_episodes)
    times["Atari+PixelObjectCentric"] = get_pixel_object_centric_time(env_name, num_episodes)
    times["Atari+PixelObjectCentric+Flattened"] = get_flattened_pixel_object_centric_time(env_name, num_episodes)
    times["Atari+ObjectCentric+Flattened+Logging"] = get_logging_time(env_name, num_episodes)
    times["AtariWrapper_no_frameskip"] = get_atari_wrapper_no_frameskip_time(env_name, num_episodes)

    for label, t in times.items():
        print(f"{label} time: {t}")

    visualize_wrapper_performances(times)

