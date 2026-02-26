import os
import csv
import time
import importlib
import signal
from contextlib import contextmanager
from itertools import combinations
from typing import Any, Dict, List, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import ale_py
from omegaconf import DictConfig, OmegaConf
import wandb

# import jaxatari
# from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper, PixelObsWrapper


#### NOTE
# To 
# GXM and envpool (without XLA) works with envpool==0.6.6 and jax+jaxlib==0.6.2, gxm==0.1.3
# uv pip install wandb "jax[cuda12]" hydra-core "gxm[envpool]" 

# For XLA envpool to work, we need to use envpool==0.6.6, but jax==0.3.13 and jaxlib==0.3.10 (directly from jax server) and numpy==1.24.4 and scipy==1.8.1
# uv pip install jax==0.3.13 jaxlib==0.3.10+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Since envpool async is faster than scanned xla (which is also async??), we can just keep using the higher jax versions

# performance of scanned XLA (8 envs, 10000 steps): 40k throughput (7.97s)
# performance of scanned XLA (128 envs, 10000 steps): 240k throughput (21.388s)
# performance of scanned XLA (256 envs, 10000 steps): 299k throughput (34.207s) 
# performance of scanned XLA (512 envs, 10000 steps): 361k throughput (56.5s)
#(that's already multiplied by 4 for frame skip) 
# performance of async envpool (256 envs, 10000 steps): 640k throughput (24.3s) [batch_size=128]
# performance of async envpool (256 envs, 10000 steps): ~800k throughput (24.3s) NUMA, [batch_size=128]
# GXM
# performance of gxm envpool (256 envs, 10000 steps): 193k throughput (52.3s) -> is it synchronous?, nope.. even worse :(
# Synced envpool
# performance of sync envpool (256 envs, 10000 steps): 398k throughput (24.7s)
# performance of sync envpool (512 envs, 10000 steps): 490k throughput (41s)
# With NUMA synced envpool (256 envs): 101k throughput (100s) -> lol

JAXATARI_BACKEND = "jaxatari"
ALE_BACKEND = "ale"
GXM_BACKEND = "gxm"
ENVPOOL_XLA_BACKEND = "envpool_xla"
ENVPOOL_ASYNC_BACKEND = "envpool_async"
ENVPOOL_SYNC_BACKEND = "envpool_sync"
JAX_MODE_OC = "oc"
JAX_MODE_PIXEL = "pixel"
PIXEL_OPT_RESIZED = "resized"
PIXEL_OPT_GRAYSCALE = "grayscale"
PIXEL_OPT_NATIVE = "native"
CPU_PLATFORM = "cpu"
GPU_PLATFORM = "gpu"


class BenchmarkTimeoutError(RuntimeError):
    pass


@contextmanager
def _timeout_guard(seconds: int, context: str):
    if seconds <= 0 or os.name == "nt":
        yield
        return

    def _handle_alarm(signum, frame):
        raise BenchmarkTimeoutError(f"{context} exceeded timeout of {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_with_timeout(function, timeout_seconds: int, context: str, **kwargs):
    with _timeout_guard(timeout_seconds, context):
        return function(**kwargs)


def _compile_and_time_rollout(rollout_fn, *args):
    compile_start = time.perf_counter()
    compiled = rollout_fn.lower(*args).compile()
    compile_s = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    output = compiled(*args)
    first_leaf = jax.tree_util.tree_leaves(output)[0]
    first_leaf.block_until_ready()
    run_s = time.perf_counter() - run_start

    return compile_s, run_s


def _build_given_action_scanned_rollout(
    env,
    num_envs: int,
    num_steps: int,
    given_action: int,
    jax_platform: str,
):
    vmapped_step = jax.vmap(env.step)
    action_batch = jnp.full((num_envs,), given_action, dtype=jnp.int32)

    def rollout(states):
        def body_fn(curr_states, _):
            _, next_states, _, _, _ = vmapped_step(curr_states, action_batch)
            return next_states, None

        final_states, _ = jax.lax.scan(body_fn, states, xs=None, length=num_steps)
        return final_states

    return jax.jit(rollout, backend=jax_platform)


def _normalize_option_set(options: List[str]) -> Tuple[str, ...]:
    if not options:
        return tuple()
    return tuple(sorted({str(x).lower() for x in options}))


def _expand_pixel_option_combinations(config: Dict[str, Any]) -> List[Tuple[str, ...]]:
    allowed = {PIXEL_OPT_RESIZED, PIXEL_OPT_GRAYSCALE, PIXEL_OPT_NATIVE}

    combo_config = config.get("JAXATARI_PIXEL_OPTION_COMBINATIONS", None)
    if combo_config is not None:
        normalized = []
        for combo in combo_config:
            option_set = _normalize_option_set(list(combo))
            unknown = set(option_set) - allowed
            if unknown:
                raise ValueError(f"Unknown pixel options in combination {combo}: {sorted(unknown)}")
            normalized.append(option_set)
    else:
        base_options = [str(x).lower() for x in config.get("JAXATARI_PIXEL_OPTIONS", [])]
        unknown = set(base_options) - allowed
        if unknown:
            raise ValueError(f"Unknown JAXATARI_PIXEL_OPTIONS: {sorted(unknown)}")

        if config.get("JAXATARI_PIXEL_TRY_ALL_COMBINATIONS", False):
            normalized = []
            for r in range(len(base_options) + 1):
                for combo in combinations(base_options, r):
                    normalized.append(_normalize_option_set(list(combo)))
        else:
            normalized = [_normalize_option_set(base_options)]

    deduped = []
    seen = set()
    for combo in normalized:
        if combo not in seen:
            seen.add(combo)
            deduped.append(combo)

    return deduped if deduped else [tuple()]


def _prepare_jaxatari_states(
    game_name: str,
    num_envs: int,
    seed: int,
    mode: str,
    pixel_options: Tuple[str, ...],
    atari_frame_skip: int,
    pixel_resize_shape: Tuple[int, int],
):
    env = jaxatari.make(game_name)
    env = AtariWrapper(env, frame_skip=atari_frame_skip)

    if mode == JAX_MODE_OC:
        env = ObjectCentricWrapper(env)
        env = FlattenObservationWrapper(env)
    elif mode == JAX_MODE_PIXEL:
        do_resize = PIXEL_OPT_RESIZED in pixel_options
        grayscale = PIXEL_OPT_GRAYSCALE in pixel_options
        use_native_downscaling = PIXEL_OPT_NATIVE in pixel_options
        env = PixelObsWrapper(
            env,
            do_pixel_resize=do_resize,
            pixel_resize_shape=pixel_resize_shape,
            grayscale=grayscale,
            use_native_downscaling=use_native_downscaling,
        )
    else:
        raise ValueError(f"Unknown JAXAtari mode '{mode}'. Expected one of [{JAX_MODE_OC}, {JAX_MODE_PIXEL}]")

    base_key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(base_key, num_envs)
    _, states = jax.vmap(env.reset)(reset_keys)
    return env, states


def _default_ale_env_id(game_name: str) -> str:
    return f"ALE/{game_name.capitalize()}-v5"


def _default_gxm_env_id(game_name: str) -> str:
    return f"{game_name.capitalize()}-v5"


def _normalize_gxm_env_id(gxm_env_id: str) -> str:
    return gxm_env_id if "/" in gxm_env_id else f"Envpool/{gxm_env_id}"


def _default_envpool_env_id(game_name: str) -> str:
    return f"{game_name.capitalize()}-v5"


def _run_ale_benchmark(
    ale_env_id: str,
    num_envs: int,
    num_steps: int,
    atari_frame_skip: int,
    seed: int,
    given_action: int,
) -> Dict[str, Any]:
    env_fns = [lambda: gym.make(ale_env_id) for _ in range(num_envs)]
    vector_env = gym.vector.AsyncVectorEnv(env_fns, context="spawn")
    vector_env.reset(seed=seed)

    actions = np.full((num_envs,), given_action, dtype=np.int32)
    # Equivalent to nested stepping for this benchmark because action is fixed.
    effective_num_steps = num_steps * atari_frame_skip
    start = time.perf_counter()
    for _ in range(effective_num_steps):
        _, _, _, _, _ = vector_env.step(actions)
    runtime_s = time.perf_counter() - start
    vector_env.close()

    total_env_steps = num_envs * effective_num_steps
    throughput = total_env_steps / runtime_s
    return {
        "compile_s": 0.0,
        "runtime_s": runtime_s,
        "total_env_steps": total_env_steps,
        "throughput_env_steps_per_sec": throughput,
    }


def _create_envpool_env(
    envpool_env_id: str,
    num_envs: int,
    atari_frame_skip: int,
    batch_size: int = None,
):
    try:
        envpool_module = importlib.import_module("envpool")
    except ImportError as error:
        raise ImportError(
            "EnvPool backend requested but 'envpool' is not installed. "
            "Install with: pip install envpool"
        ) from error
    env = envpool_module.make_gym(
        envpool_env_id,
        num_envs=num_envs,
        batch_size=batch_size if batch_size is not None else num_envs,
        frame_skip=atari_frame_skip,
    )
    return env, "envpool.make_gym"


def _run_envpool_xla_benchmark(
    envpool_env_id: str,
    num_envs: int,
    num_steps: int,
    atari_frame_skip: int,
    seed: int,
    given_action: int,
) -> Dict[str, Any]:
    env, envpool_impl = _create_envpool_env(
        envpool_env_id=envpool_env_id,
        num_envs=num_envs,
        atari_frame_skip=atari_frame_skip,
    )

    env.action_space.seed(seed)
    action_batch = jnp.full((num_envs,), given_action, dtype=jnp.int32)

    try:
        handle, _, _, step_env = env.xla()

        def envpool_rollout(initial_handle, action):
            def body_fn(curr_handle, _):
                next_handle, _ = step_env(curr_handle, action)
                return next_handle, None

            final_handle, _ = jax.lax.scan(body_fn, initial_handle, xs=None, length=num_steps)
            return final_handle

        rollout = jax.jit(envpool_rollout)
        compile_s, runtime_s = _compile_and_time_rollout(rollout, handle, action_batch)
    finally:
        env.close()

    total_env_steps = num_envs * num_steps * atari_frame_skip
    throughput = total_env_steps / runtime_s
    return {
        "compile_s": compile_s,
        "runtime_s": runtime_s,
        "total_env_steps": total_env_steps,
        "throughput_env_steps_per_sec": throughput,
        "envpool_impl": f"{envpool_impl}.xla",
    }


def _run_envpool_async_benchmark(
    envpool_env_id: str,
    num_envs: int,
    num_steps: int,
    atari_frame_skip: int,
    seed: int,
    given_action: int,
) -> Dict[str, Any]:
    env, envpool_impl = _create_envpool_env(
        envpool_env_id=envpool_env_id,
        num_envs=num_envs,
        atari_frame_skip=atari_frame_skip,
        batch_size=128
    )

    env.async_reset()
    env.action_space.seed(seed)
    # action_batch = np.full((num_envs,), given_action, dtype=np.int32)
    action_batch = np.full((128,), given_action, dtype=np.int32)

    try:
        actual_size = 0
        start = time.perf_counter()
        # for _ in range(num_steps):
        while actual_size < num_envs * num_steps:
            info = env.recv()[-1]
            actual_size += len(info["env_id"])
            env.send(action_batch, info["env_id"])
        runtime_s = time.perf_counter() - start
    finally:
        env.close()

    # total_env_steps = num_envs * num_steps * atari_frame_skip
    total_env_steps = actual_size * atari_frame_skip
    throughput = total_env_steps / runtime_s
    return {
        "compile_s": 0.0,
        "runtime_s": runtime_s,
        "total_env_steps": total_env_steps,
        "throughput_env_steps_per_sec": throughput,
        "envpool_impl": f"{envpool_impl}.async",
    }


def _run_envpool_sync_benchmark(
    envpool_env_id: str,
    num_envs: int,
    num_steps: int,
    atari_frame_skip: int,
    seed: int,
    given_action: int,
) -> Dict[str, Any]:
    env, envpool_impl = _create_envpool_env(
        envpool_env_id=envpool_env_id,
        num_envs=num_envs,
        atari_frame_skip=atari_frame_skip,
    )

    env.action_space.seed(seed)
    action_batch = np.full((num_envs,), given_action, dtype=np.int32)

    try:
        env.reset()
        start = time.perf_counter()
        for _ in range(num_steps):
            step_result = env.step(action_batch)
        runtime_s = time.perf_counter() - start
    finally:
        env.close()

    total_env_steps = num_envs * num_steps * atari_frame_skip
    throughput = total_env_steps / runtime_s
    return {
        "compile_s": 0.0,
        "runtime_s": runtime_s,
        "total_env_steps": total_env_steps,
        "throughput_env_steps_per_sec": throughput,
        "envpool_impl": f"{envpool_impl}.sync",
    }


def _create_gxm_env(
    gxm_env_id: str,
    atari_frame_skip: int,
):
    try:
        gxm_module = importlib.import_module("gxm")
    except ImportError as error:
        raise ImportError(
            "GXM backend requested but 'gxm' is not installed. "
            "Install with: pip install 'gxm[envpool]'"
        ) from error

    gxm_target_env = _normalize_gxm_env_id(gxm_env_id)
    creation_attempts = []

    def _try_add(description, factory):
        creation_attempts.append((description, factory))

    if hasattr(gxm_module, "make"):
        _try_add(
            "gxm.make",
            lambda: gxm_module.make(gxm_target_env, frame_skip=atari_frame_skip),
        )

    errors = []
    for description, factory in creation_attempts:
        try:
            return factory(), description
        except Exception as error:
            errors.append(f"{description}: {type(error).__name__}: {error}")

    raise RuntimeError(
        "Unable to construct a GXM/envpool environment. Attempted constructors:\n"
        + "\n".join(errors)
    )


def _run_gxm_benchmark(
    gxm_env_id: str,
    num_envs: int,
    num_steps: int,
    atari_frame_skip: int,
    seed: int,
    given_action: int,
    gxm_platform: str,
) -> Dict[str, Any]:
    env, gxm_impl = _create_gxm_env(
        gxm_env_id=gxm_env_id,
        atari_frame_skip=atari_frame_skip,
    )

    base_key = jax.random.PRNGKey(seed)
    init_keys = jax.random.split(base_key, num_envs)
    reset_keys = jax.random.split(jax.random.fold_in(base_key, 1), num_envs)
    action_batch = jnp.full((num_envs,), given_action, dtype=jnp.int32)
    vmapped_init = jax.vmap(env.init)
    vmapped_reset = jax.vmap(env.reset)
    vmapped_step = jax.vmap(env.step)

    env_states, timesteps = vmapped_init(init_keys)
    env_states, timesteps = vmapped_reset(reset_keys, env_states)
    def gxm_rollout(initial_states, initial_timesteps, rollout_key):
        # Equivalent to nested scans for this benchmark because action is fixed.
        def body_fn(carry, _):
            curr_states, curr_timesteps, curr_key = carry
            curr_key, step_key = jax.random.split(curr_key)
            step_keys = jax.random.split(step_key, num_envs)
            next_states, next_timesteps = vmapped_step(step_keys, curr_states, action_batch)
            return (next_states, next_timesteps, curr_key), None

        final_carry, _ = jax.lax.scan(
            body_fn,
            (initial_states, initial_timesteps, rollout_key),
            xs=None,
            length=num_steps,
        )
        return final_carry

    rollout = jax.jit(gxm_rollout, backend=gxm_platform)
    compile_s, runtime_s = _compile_and_time_rollout(rollout, env_states, timesteps, base_key)

    total_env_steps = num_envs * num_steps * atari_frame_skip
    throughput = total_env_steps / runtime_s
    return {
        "compile_s": compile_s,
        "runtime_s": runtime_s,
        "total_env_steps": total_env_steps,
        "throughput_env_steps_per_sec": throughput,
        "gxm_impl": gxm_impl,
    }


def _save_results_csv(results: List[Dict[str, Any]], csv_path: str) -> str:
    if not results:
        return ""

    abs_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(abs_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return abs_path


def _normalize_frame_skip_values(frame_skip_config: Any) -> List[int]:
    raw_values = frame_skip_config if isinstance(frame_skip_config, (list, tuple)) else [frame_skip_config]
    frame_skips = [int(value) for value in raw_values]
    if not frame_skips:
        raise ValueError("ATARI_FRAME_SKIP must contain at least one value")
    for frame_skip in frame_skips:
        if frame_skip < 1:
            raise ValueError(f"ATARI_FRAME_SKIP values must be >= 1, got {frame_skip}")
    return frame_skips


def run_throughput_benchmark(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    env_counts = [int(x) for x in config["ENV_COUNTS"]]
    num_steps = int(config["NUM_STEPS"])
    game_name = str(config["GAME_NAME"]).lower()
    backends = [str(x).lower() for x in config.get("BENCHMARK_BACKENDS", [JAXATARI_BACKEND])]
    jaxatari_modes = [str(x).lower() for x in config.get("JAXATARI_MODES", [JAX_MODE_OC, JAX_MODE_PIXEL])]
    jaxatari_platforms = [str(x).lower() for x in config.get("JAXATARI_PLATFORMS", [GPU_PLATFORM])]
    atari_frame_skips = _normalize_frame_skip_values(config.get("ATARI_FRAME_SKIP", 4))
    pixel_resize_shape = tuple(config.get("PIXEL_RESIZE_SHAPE", [84, 84]))
    pixel_option_combinations = _expand_pixel_option_combinations(config)
    ale_env_id = str(config.get("ALE_ENV_ID") or _default_ale_env_id(game_name))
    gxm_env_id = str(config.get("GXM_ENV_ID") or _default_gxm_env_id(game_name))
    envpool_env_id = str(config.get("ENVPOOL_ENV_ID") or _default_envpool_env_id(game_name))
    gxm_env_type = str(config.get("GXM_ENV_TYPE", "envpool"))
    gxm_platforms = [str(x).lower() for x in config.get("GXM_PLATFORMS", [GPU_PLATFORM])]
    base_seed = int(config["SEED"])
    given_action = int(config["GIVEN_ACTION"])
    save_results_csv = bool(config.get("SAVE_RESULTS_CSV", False))
    results_csv_path = str(config.get("RESULTS_CSV_PATH", "./scripts/benchmarks/outputs/throughput_results.csv"))
    ale_timeout_s = int(config.get("ALE_TIMEOUT_S", 3600))
    gxm_timeout_s = int(config.get("GXM_TIMEOUT_S", 3600))
    envpool_timeout_s = int(config.get("ENVPOOL_TIMEOUT_S", 3600))

    if JAXATARI_BACKEND in backends or GXM_BACKEND in backends:
        available_platforms = {device.platform for device in jax.devices()}

    if JAXATARI_BACKEND in backends:
        for platform in jaxatari_platforms:
            if platform not in (CPU_PLATFORM, GPU_PLATFORM):
                raise ValueError(f"Unknown JAX platform '{platform}'. Expected one of [{CPU_PLATFORM}, {GPU_PLATFORM}]")
            if platform not in available_platforms:
                raise ValueError(f"Requested JAX platform '{platform}' not available. Available: {sorted(available_platforms)}")

        for mode in jaxatari_modes:
            if mode not in (JAX_MODE_OC, JAX_MODE_PIXEL):
                raise ValueError(f"Unknown JAXAtari mode '{mode}'. Expected one of [{JAX_MODE_OC}, {JAX_MODE_PIXEL}]")

    if GXM_BACKEND in backends:
        for platform in gxm_platforms:
            if platform not in (CPU_PLATFORM, GPU_PLATFORM):
                raise ValueError(f"Unknown GXM platform '{platform}'. Expected one of [{CPU_PLATFORM}, {GPU_PLATFORM}]")
            if platform not in available_platforms:
                raise ValueError(f"Requested GXM platform '{platform}' not available. Available: {sorted(available_platforms)}")

    gym.register_envs(ale_py)

    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        mode=config["WANDB_MODE"],
        name=config.get("RUN_NAME", "throughput_analysis"),
        tags=list(config.get("TAGS", [])),
        notes=config.get("NOTES", ""),
        config=config,
    )

    results: List[Dict[str, Any]] = []
    run_step = 0

    for atari_frame_skip in atari_frame_skips:
        for backend in backends:
            if backend not in (
                JAXATARI_BACKEND,
                ALE_BACKEND,
                GXM_BACKEND,
                ENVPOOL_XLA_BACKEND,
                ENVPOOL_ASYNC_BACKEND,
                ENVPOOL_SYNC_BACKEND,
            ):
                raise ValueError(
                    "Unknown backend "
                    f"'{backend}'. Expected one of: [{JAXATARI_BACKEND}, {ALE_BACKEND}, {GXM_BACKEND}, "
                    f"{ENVPOOL_XLA_BACKEND}, {ENVPOOL_ASYNC_BACKEND}, {ENVPOOL_SYNC_BACKEND}]"
                )

            for num_envs in env_counts:
                if backend == JAXATARI_BACKEND:
                    for jax_mode in jaxatari_modes:
                        mode_pixel_combos = pixel_option_combinations if jax_mode == JAX_MODE_PIXEL else [tuple()]

                        for pixel_options in mode_pixel_combos:
                            pixel_options_str = "+".join(pixel_options)

                            for jax_platform in jaxatari_platforms:
                                env_seed = base_seed + run_step
                                env, states = _prepare_jaxatari_states(
                                    game_name=game_name,
                                    num_envs=num_envs,
                                    seed=env_seed,
                                    mode=jax_mode,
                                    pixel_options=pixel_options,
                                    atari_frame_skip=atari_frame_skip,
                                    pixel_resize_shape=pixel_resize_shape,
                                )
                                rollout = _build_given_action_scanned_rollout(
                                    env=env,
                                    num_envs=num_envs,
                                    num_steps=num_steps,
                                    given_action=given_action,
                                    jax_platform=jax_platform,
                                )
                                compile_s, runtime_s = _compile_and_time_rollout(rollout, states)
                                total_env_steps = num_envs * num_steps * atari_frame_skip
                                throughput = total_env_steps / runtime_s

                                row = {
                                    "backend": backend,
                                    "game_name": game_name,
                                    "ale_env_id": "",
                                    "gxm_env_id": "",
                                    "gxm_env_type": "",
                                    "gxm_impl": "",
                                    "jaxatari_mode": jax_mode,
                                    "pixel_options": pixel_options_str,
                                    "atari_frame_skip": atari_frame_skip,
                                    "jax_platform": jax_platform,
                                    "num_envs": num_envs,
                                    "num_steps": num_steps,
                                    "total_env_steps": total_env_steps,
                                    "given_action": given_action,
                                    "compile_s": compile_s,
                                    "runtime_s": runtime_s,
                                    "throughput_env_steps_per_sec": throughput,
                                }
                                results.append(row)

                                wandb.log(
                                    {
                                        "benchmark/backend": backend,
                                        "benchmark/game_name": game_name,
                                        "benchmark/ale_env_id": "",
                                        "benchmark/gxm_env_id": "",
                                        "benchmark/gxm_env_type": "",
                                        "benchmark/gxm_impl": "",
                                        "benchmark/jaxatari_mode": jax_mode,
                                        "benchmark/pixel_options": pixel_options_str,
                                        "benchmark/atari_frame_skip": atari_frame_skip,
                                        "benchmark/jax_platform": jax_platform,
                                        "benchmark/num_envs": num_envs,
                                        "benchmark/num_steps": num_steps,
                                        "benchmark/total_env_steps": total_env_steps,
                                        "benchmark/given_action": given_action,
                                        "benchmark/compile_s": compile_s,
                                        "benchmark/runtime_s": runtime_s,
                                        "benchmark/throughput_env_steps_per_sec": throughput,
                                    },
                                    step=run_step,
                                )

                                print(
                                    f"backend={backend:8s} | fs={atari_frame_skip:2d} | mode={jax_mode:5s} | "
                                    f"pixel={pixel_options_str or '-':15s} | platform={jax_platform:3s} | "
                                    f"envs={num_envs:4d} | steps={num_steps} | "
                                    f"compile={compile_s:.3f}s | run={runtime_s:.3f}s | "
                                    f"throughput={throughput:,.2f} env-steps/s"
                                )

                                run_step += 1
                    continue
                elif backend == ALE_BACKEND:
                    env_seed = base_seed + run_step
                    try:
                        ale_metrics = _run_with_timeout(
                            _run_ale_benchmark,
                            timeout_seconds=ale_timeout_s,
                            context=f"ALE benchmark (envs={num_envs}, frame_skip={atari_frame_skip})",
                            ale_env_id=ale_env_id,
                            num_envs=num_envs,
                            num_steps=num_steps,
                            atari_frame_skip=atari_frame_skip,
                            seed=env_seed,
                            given_action=given_action,
                        )
                    except Exception as error:
                        print(
                            f"backend={backend:8s} | fs={atari_frame_skip:2d} | env_id={ale_env_id:16s} | "
                            f"envs={num_envs:4d} | steps={num_steps} | ERROR={type(error).__name__}: {error}"
                        )
                        run_step += 1
                        continue
                    compile_s = ale_metrics["compile_s"]
                    runtime_s = ale_metrics["runtime_s"]
                    total_env_steps = ale_metrics["total_env_steps"]
                    throughput = ale_metrics["throughput_env_steps_per_sec"]
                    gxm_impl = ""
                    row = {
                        "backend": backend,
                        "game_name": game_name,
                        "ale_env_id": ale_env_id,
                        "gxm_env_id": "",
                        "gxm_env_type": "",
                        "gxm_impl": gxm_impl,
                        "jaxatari_mode": "",
                        "pixel_options": "",
                        "atari_frame_skip": atari_frame_skip,
                        "jax_platform": "",
                        "num_envs": num_envs,
                        "num_steps": num_steps,
                        "total_env_steps": total_env_steps,
                        "given_action": given_action,
                        "compile_s": compile_s,
                        "runtime_s": runtime_s,
                        "throughput_env_steps_per_sec": throughput,
                    }
                    results.append(row)

                    wandb.log(
                        {
                            "benchmark/backend": backend,
                            "benchmark/game_name": game_name,
                            "benchmark/ale_env_id": ale_env_id,
                            "benchmark/gxm_env_id": "",
                            "benchmark/gxm_env_type": "",
                            "benchmark/gxm_impl": gxm_impl,
                            "benchmark/jaxatari_mode": "",
                            "benchmark/pixel_options": "",
                            "benchmark/atari_frame_skip": atari_frame_skip,
                            "benchmark/jax_platform": "",
                            "benchmark/num_envs": num_envs,
                            "benchmark/num_steps": num_steps,
                            "benchmark/total_env_steps": total_env_steps,
                            "benchmark/given_action": given_action,
                            "benchmark/compile_s": compile_s,
                            "benchmark/runtime_s": runtime_s,
                            "benchmark/throughput_env_steps_per_sec": throughput,
                        },
                        step=run_step,
                    )

                    print(
                        f"backend={backend:8s} | fs={atari_frame_skip:2d} | env_id={ale_env_id:16s} | "
                        f"envs={num_envs:4d} | steps={num_steps} | run={runtime_s:.3f}s | "
                        f"throughput={throughput:,.2f} env-steps/s"
                    )

                    run_step += 1
                    continue
                elif backend == ENVPOOL_XLA_BACKEND:
                    env_seed = base_seed + run_step
                    try:
                        envpool_metrics = _run_with_timeout(
                            _run_envpool_xla_benchmark,
                            timeout_seconds=envpool_timeout_s,
                            context=(
                                f"EnvPool XLA benchmark (envs={num_envs}, frame_skip={atari_frame_skip})"
                            ),
                            envpool_env_id=envpool_env_id,
                            num_envs=num_envs,
                            num_steps=num_steps,
                            atari_frame_skip=atari_frame_skip,
                            seed=env_seed,
                            given_action=given_action,
                        )
                    except Exception as error:
                        print(
                            f"backend={backend:8s} | fs={atari_frame_skip:2d} | env_id={envpool_env_id:16s} | "
                            f"envs={num_envs:4d} | steps={num_steps} | ERROR={type(error).__name__}: {error}"
                        )
                        run_step += 1
                        continue

                    compile_s = envpool_metrics["compile_s"]
                    runtime_s = envpool_metrics["runtime_s"]
                    total_env_steps = envpool_metrics["total_env_steps"]
                    throughput = envpool_metrics["throughput_env_steps_per_sec"]
                    envpool_impl = envpool_metrics["envpool_impl"]
                    row = {
                        "backend": backend,
                        "game_name": game_name,
                        "ale_env_id": "",
                        "gxm_env_id": envpool_env_id,
                        "gxm_env_type": "envpool-standard",
                        "gxm_impl": envpool_impl,
                        "jaxatari_mode": "",
                        "pixel_options": "",
                        "atari_frame_skip": atari_frame_skip,
                        "jax_platform": "",
                        "num_envs": num_envs,
                        "num_steps": num_steps,
                        "total_env_steps": total_env_steps,
                        "given_action": given_action,
                        "compile_s": compile_s,
                        "runtime_s": runtime_s,
                        "throughput_env_steps_per_sec": throughput,
                    }
                    results.append(row)

                    wandb.log(
                        {
                            "benchmark/backend": backend,
                            "benchmark/game_name": game_name,
                            "benchmark/ale_env_id": "",
                            "benchmark/gxm_env_id": envpool_env_id,
                            "benchmark/gxm_env_type": "envpool-standard",
                            "benchmark/gxm_impl": envpool_impl,
                            "benchmark/jaxatari_mode": "",
                            "benchmark/pixel_options": "",
                            "benchmark/atari_frame_skip": atari_frame_skip,
                            "benchmark/jax_platform": "",
                            "benchmark/num_envs": num_envs,
                            "benchmark/num_steps": num_steps,
                            "benchmark/total_env_steps": total_env_steps,
                            "benchmark/given_action": given_action,
                            "benchmark/compile_s": compile_s,
                            "benchmark/runtime_s": runtime_s,
                            "benchmark/throughput_env_steps_per_sec": throughput,
                        },
                        step=run_step,
                    )

                    print(
                        f"backend={backend:8s} | fs={atari_frame_skip:2d} | impl={envpool_impl:16s} | "
                        f"envs={num_envs:4d} | steps={num_steps} | run={runtime_s:.3f}s | "
                        f"throughput={throughput:,.2f} env-steps/s"
                    )

                    run_step += 1
                    continue
                elif backend == ENVPOOL_ASYNC_BACKEND:
                    env_seed = base_seed + run_step
                    try:
                        envpool_metrics = _run_with_timeout(
                            _run_envpool_async_benchmark,
                            timeout_seconds=envpool_timeout_s,
                            context=(
                                f"EnvPool async benchmark (envs={num_envs}, frame_skip={atari_frame_skip})"
                            ),
                            envpool_env_id=envpool_env_id,
                            num_envs=num_envs,
                            num_steps=num_steps,
                            atari_frame_skip=atari_frame_skip,
                            seed=env_seed,
                            given_action=given_action,
                        )
                    except Exception as error:
                        print(
                            f"backend={backend:8s} | fs={atari_frame_skip:2d} | env_id={envpool_env_id:16s} | "
                            f"envs={num_envs:4d} | steps={num_steps} | ERROR={type(error).__name__}: {error}"
                        )
                        run_step += 1
                        continue

                    compile_s = envpool_metrics["compile_s"]
                    runtime_s = envpool_metrics["runtime_s"]
                    total_env_steps = envpool_metrics["total_env_steps"]
                    throughput = envpool_metrics["throughput_env_steps_per_sec"]
                    envpool_impl = envpool_metrics["envpool_impl"]
                    row = {
                        "backend": backend,
                        "game_name": game_name,
                        "ale_env_id": "",
                        "gxm_env_id": envpool_env_id,
                        "gxm_env_type": "envpool-standard",
                        "gxm_impl": envpool_impl,
                        "jaxatari_mode": "",
                        "pixel_options": "",
                        "atari_frame_skip": atari_frame_skip,
                        "jax_platform": "",
                        "num_envs": num_envs,
                        "num_steps": num_steps,
                        "total_env_steps": total_env_steps,
                        "given_action": given_action,
                        "compile_s": compile_s,
                        "runtime_s": runtime_s,
                        "throughput_env_steps_per_sec": throughput,
                    }
                    results.append(row)

                    wandb.log(
                        {
                            "benchmark/backend": backend,
                            "benchmark/game_name": game_name,
                            "benchmark/ale_env_id": "",
                            "benchmark/gxm_env_id": envpool_env_id,
                            "benchmark/gxm_env_type": "envpool-standard",
                            "benchmark/gxm_impl": envpool_impl,
                            "benchmark/jaxatari_mode": "",
                            "benchmark/pixel_options": "",
                            "benchmark/atari_frame_skip": atari_frame_skip,
                            "benchmark/jax_platform": "",
                            "benchmark/num_envs": num_envs,
                            "benchmark/num_steps": num_steps,
                            "benchmark/total_env_steps": total_env_steps,
                            "benchmark/given_action": given_action,
                            "benchmark/compile_s": compile_s,
                            "benchmark/runtime_s": runtime_s,
                            "benchmark/throughput_env_steps_per_sec": throughput,
                        },
                        step=run_step,
                    )

                    print(
                        f"backend={backend:8s} | fs={atari_frame_skip:2d} | impl={envpool_impl:16s} | "
                        f"envs={num_envs:4d} | steps={num_steps} | run={runtime_s:.3f}s | "
                        f"throughput={throughput:,.2f} env-steps/s"
                    )

                    run_step += 1
                    continue
                elif backend == ENVPOOL_SYNC_BACKEND:
                    env_seed = base_seed + run_step
                    try:
                        envpool_metrics = _run_with_timeout(
                            _run_envpool_sync_benchmark,
                            timeout_seconds=envpool_timeout_s,
                            context=(
                                f"EnvPool sync benchmark (envs={num_envs}, frame_skip={atari_frame_skip})"
                            ),
                            envpool_env_id=envpool_env_id,
                            num_envs=num_envs,
                            num_steps=num_steps,
                            atari_frame_skip=atari_frame_skip,
                            seed=env_seed,
                            given_action=given_action,
                        )
                    except Exception as error:
                        print(
                            f"backend={backend:8s} | fs={atari_frame_skip:2d} | env_id={envpool_env_id:16s} | "
                            f"envs={num_envs:4d} | steps={num_steps} | ERROR={type(error).__name__}: {error}"
                        )
                        run_step += 1
                        continue

                    compile_s = envpool_metrics["compile_s"]
                    runtime_s = envpool_metrics["runtime_s"]
                    total_env_steps = envpool_metrics["total_env_steps"]
                    throughput = envpool_metrics["throughput_env_steps_per_sec"]
                    envpool_impl = envpool_metrics["envpool_impl"]
                    row = {
                        "backend": backend,
                        "game_name": game_name,
                        "ale_env_id": "",
                        "gxm_env_id": envpool_env_id,
                        "gxm_env_type": "envpool-standard",
                        "gxm_impl": envpool_impl,
                        "jaxatari_mode": "",
                        "pixel_options": "",
                        "atari_frame_skip": atari_frame_skip,
                        "jax_platform": "",
                        "num_envs": num_envs,
                        "num_steps": num_steps,
                        "total_env_steps": total_env_steps,
                        "given_action": given_action,
                        "compile_s": compile_s,
                        "runtime_s": runtime_s,
                        "throughput_env_steps_per_sec": throughput,
                    }
                    results.append(row)

                    wandb.log(
                        {
                            "benchmark/backend": backend,
                            "benchmark/game_name": game_name,
                            "benchmark/ale_env_id": "",
                            "benchmark/gxm_env_id": envpool_env_id,
                            "benchmark/gxm_env_type": "envpool-standard",
                            "benchmark/gxm_impl": envpool_impl,
                            "benchmark/jaxatari_mode": "",
                            "benchmark/pixel_options": "",
                            "benchmark/atari_frame_skip": atari_frame_skip,
                            "benchmark/jax_platform": "",
                            "benchmark/num_envs": num_envs,
                            "benchmark/num_steps": num_steps,
                            "benchmark/total_env_steps": total_env_steps,
                            "benchmark/given_action": given_action,
                            "benchmark/compile_s": compile_s,
                            "benchmark/runtime_s": runtime_s,
                            "benchmark/throughput_env_steps_per_sec": throughput,
                        },
                        step=run_step,
                    )

                    print(
                        f"backend={backend:8s} | fs={atari_frame_skip:2d} | impl={envpool_impl:16s} | "
                        f"envs={num_envs:4d} | steps={num_steps} | run={runtime_s:.3f}s | "
                        f"throughput={throughput:,.2f} env-steps/s"
                    )

                    run_step += 1
                    continue
                else:
                    for gxm_platform in gxm_platforms:
                        env_seed = base_seed + run_step
                        try:
                            gxm_metrics = _run_with_timeout(
                                _run_gxm_benchmark,
                                timeout_seconds=gxm_timeout_s,
                                context=(
                                    f"GXM benchmark (platform={gxm_platform}, envs={num_envs}, "
                                    f"frame_skip={atari_frame_skip})"
                                ),
                                gxm_env_id=gxm_env_id,
                                num_envs=num_envs,
                                num_steps=num_steps,
                                atari_frame_skip=atari_frame_skip,
                                seed=env_seed,
                                given_action=given_action,
                                gxm_platform=gxm_platform,
                            )
                        except Exception as error:
                            print(
                                f"backend={backend:8s} | fs={atari_frame_skip:2d} | platform={gxm_platform:3s} | "
                                f"envs={num_envs:4d} | steps={num_steps} | ERROR={type(error).__name__}: {error}"
                            )
                            run_step += 1
                            continue
                        compile_s = gxm_metrics["compile_s"]
                        runtime_s = gxm_metrics["runtime_s"]
                        total_env_steps = gxm_metrics["total_env_steps"]
                        throughput = gxm_metrics["throughput_env_steps_per_sec"]
                        gxm_impl = gxm_metrics["gxm_impl"]

                        row = {
                            "backend": backend,
                            "game_name": game_name,
                            "ale_env_id": "",
                            "gxm_env_id": gxm_env_id,
                            "gxm_env_type": gxm_env_type,
                            "gxm_impl": gxm_impl,
                            "jaxatari_mode": "",
                            "pixel_options": "",
                            "atari_frame_skip": atari_frame_skip,
                            "jax_platform": gxm_platform,
                            "num_envs": num_envs,
                            "num_steps": num_steps,
                            "total_env_steps": total_env_steps,
                            "given_action": given_action,
                            "compile_s": compile_s,
                            "runtime_s": runtime_s,
                            "throughput_env_steps_per_sec": throughput,
                        }
                        results.append(row)

                        wandb.log(
                            {
                                "benchmark/backend": backend,
                                "benchmark/game_name": game_name,
                                "benchmark/ale_env_id": "",
                                "benchmark/gxm_env_id": gxm_env_id,
                                "benchmark/gxm_env_type": gxm_env_type,
                                "benchmark/gxm_impl": gxm_impl,
                                "benchmark/jaxatari_mode": "",
                                "benchmark/pixel_options": "",
                                "benchmark/atari_frame_skip": atari_frame_skip,
                                "benchmark/jax_platform": gxm_platform,
                                "benchmark/num_envs": num_envs,
                                "benchmark/num_steps": num_steps,
                                "benchmark/total_env_steps": total_env_steps,
                                "benchmark/given_action": given_action,
                                "benchmark/compile_s": compile_s,
                                "benchmark/runtime_s": runtime_s,
                                "benchmark/throughput_env_steps_per_sec": throughput,
                            },
                            step=run_step,
                        )

                        print(
                            f"backend={backend:8s} | fs={atari_frame_skip:2d} | impl={gxm_impl:16s} | "
                            f"platform={gxm_platform:3s} | envs={num_envs:4d} | steps={num_steps} | "
                            f"compile={compile_s:.3f}s | run={runtime_s:.3f}s | "
                            f"throughput={throughput:,.2f} env-steps/s"
                        )

                        run_step += 1
                    continue

    if run is not None:
        table = wandb.Table(
            columns=[
                "backend",
                "game_name",
                "ale_env_id",
                "gxm_env_id",
                "gxm_env_type",
                "gxm_impl",
                "jaxatari_mode",
                "pixel_options",
                "atari_frame_skip",
                "jax_platform",
                "num_envs",
                "num_steps",
                "total_env_steps",
                "given_action",
                "compile_s",
                "runtime_s",
                "throughput_env_steps_per_sec",
            ]
        )
        for row in results:
            table.add_data(
                row["backend"],
                row["game_name"],
                row["ale_env_id"],
                row["gxm_env_id"],
                row["gxm_env_type"],
                row["gxm_impl"],
                row["jaxatari_mode"],
                row["pixel_options"],
                row["atari_frame_skip"],
                row["jax_platform"],
                row["num_envs"],
                row["num_steps"],
                row["total_env_steps"],
                row["given_action"],
                row["compile_s"],
                row["runtime_s"],
                row["throughput_env_steps_per_sec"],
            )
        wandb.log({"benchmark/results_table": table})

        if save_results_csv:
            csv_output_path = _save_results_csv(results, results_csv_path)
            print(f"Saved results CSV: {csv_output_path}")

        run.finish()

    return results


@hydra.main(version_base=None, config_path="./config", config_name="throughput_analysis")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))
    config = OmegaConf.to_container(cfg, resolve=True)
    run_throughput_benchmark(config)


if __name__ == "__main__":
    main()
