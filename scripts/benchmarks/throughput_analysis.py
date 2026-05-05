import os
import csv
import time
import signal
import re
import math
from datetime import datetime, timezone
from contextlib import contextmanager
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, cast

import hydra
import numpy as np
import gymnasium as gym
import ale_py
from omegaconf import DictConfig, OmegaConf
# import wandb
from cleanrl_atari_wrapper import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


#### NOTE
# To 
# GXM and envpool (without XLA) works with envpool==0.6.6 and jax+jaxlib==0.6.2, gxm==0.1.3
# uv pip install wandb "jax[cuda12]" hydra-core "gxm[envpool]" 

# For XLA envpool to work, we need to use envpool==0.6.6, but jax==0.3.13 and jaxlib==0.3.10 (directly from jax server) and numpy==1.24.4 and scipy==1.8.1
# also needed for async(?), otherwise I get segfaults/double free etc
# uv pip install jax==0.3.13 jaxlib==0.3.10+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# uv pip install numpy==1.24.4 scipy==1.8.1 
# Since envpool async is faster than scanned xla (which is also async??), we can just keep using the higher jax versions

# performance of scanned XLA (8 envs, 10000 steps): 40k throughput (7.97s)
# performance of scanned XLA (128 envs, 10000 steps): 240k throughput (21.388s)
# performance of scanned XLA (256 envs, 10000 steps): 299k throughput (34.207s) 
# performance of scanned XLA (512 envs, 10000 steps): 361k throughput (56.5s)
#(that's already multiplied by 4 for frame skip) 
# performance of async envpool (256 envs, 10000 steps): 640k throughput (24.3s) [batch_size=128]
# performance of async envpool (256 envs, 10000 steps): ~800k throughput (24.3s) NUMA, [batch_size=128]
# GXM
# performance of gxm envpool (256 envs, 10000 steps): 193k throughput (52.3s) -> is it synchronous? yes, and weirdly implemented
# Synced envpool
# performance of sync envpool (256 envs, 10000 steps): 398k throughput (24.7s)
# performance of sync envpool (512 envs, 10000 steps): 490k throughput (41s)

JAXATARI_BACKEND = "jaxatari"
ALE_BACKEND_LEGACY = "ale"
GYM_ALE_BACKEND = "gym_ale"
ALE_VECTORENV_BACKEND = "ale_vectorenv"
JAX_MODE_OC = "oc"
JAX_MODE_PIXEL = "pixel"
PIXEL_OPT_RESIZED = "resized"
PIXEL_OPT_GRAYSCALE = "grayscale"
PIXEL_OPT_NATIVE = "native"
CPU_PLATFORM = "cpu"
GPU_PLATFORM = "gpu"

RESULT_COLUMNS = [
    "backend",
    "env_name",
    "jaxatari_mode",
    "pixel_options",
    "ale_vectorenv_async_stepping",
    "atari_frame_skip",
    "jax_platform",
    "num_envs",
    "num_steps",
    "total_env_steps",
    "compile_s",
    "runtime_s",
    "throughput_env_steps_per_sec",
    "status",
    "error",
]


class BenchmarkTimeoutError(RuntimeError):
    pass


_JAX_MODULE = None
_JNP_MODULE = None
_JAXATARI_MODULE = None
_ATARI_WRAPPER = None
_OBJECT_CENTRIC_WRAPPER = None
_FLATTEN_OBSERVATION_WRAPPER = None
_PIXEL_OBS_WRAPPER = None


def _get_jax_modules():
    global _JAX_MODULE, _JNP_MODULE
    if _JAX_MODULE is None or _JNP_MODULE is None:
        import jax
        import jax.numpy as jnp

        _JAX_MODULE = jax
        _JNP_MODULE = jnp
    return _JAX_MODULE, _JNP_MODULE


def _get_jaxatari_modules():
    global _JAXATARI_MODULE
    global _ATARI_WRAPPER, _OBJECT_CENTRIC_WRAPPER, _FLATTEN_OBSERVATION_WRAPPER, _PIXEL_OBS_WRAPPER
    if (
        _JAXATARI_MODULE is None
        or _ATARI_WRAPPER is None
        or _OBJECT_CENTRIC_WRAPPER is None
        or _FLATTEN_OBSERVATION_WRAPPER is None
        or _PIXEL_OBS_WRAPPER is None
    ):
        import jaxatari
        from jaxatari.wrappers import (
            AtariWrapper,
            FlattenObservationWrapper,
            ObjectCentricWrapper,
            PixelObsWrapper,
        )

        _JAXATARI_MODULE = jaxatari
        _ATARI_WRAPPER = AtariWrapper
        _OBJECT_CENTRIC_WRAPPER = ObjectCentricWrapper
        _FLATTEN_OBSERVATION_WRAPPER = FlattenObservationWrapper
        _PIXEL_OBS_WRAPPER = PixelObsWrapper

    return (
        _JAXATARI_MODULE,
        _ATARI_WRAPPER,
        _OBJECT_CENTRIC_WRAPPER,
        _FLATTEN_OBSERVATION_WRAPPER,
        _PIXEL_OBS_WRAPPER,
    )


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
    jax, _ = _get_jax_modules()
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
    jax, jnp = _get_jax_modules()
    vmapped_step = jax.vmap(env.step)
    action_batch = jnp.full((num_envs,), given_action, dtype=jnp.int32)

    def rollout(states):
        def body_fn(curr_states, _):
            _, next_states, _, _, _, _ = vmapped_step(curr_states, action_batch)
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
    jax, _ = _get_jax_modules()
    (
        jaxatari,
        AtariWrapper,
        ObjectCentricWrapper,
        FlattenObservationWrapper,
        PixelObsWrapper,
    ) = _get_jaxatari_modules()

    env = jaxatari.make(game_name)
    env = AtariWrapper(env)

    if mode == JAX_MODE_OC:
        env = ObjectCentricWrapper(env, frame_skip=atari_frame_skip)
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
            frame_skip=atari_frame_skip,
        )
    else:
        raise ValueError(f"Unknown JAXAtari mode '{mode}'. Expected one of [{JAX_MODE_OC}, {JAX_MODE_PIXEL}]")

    base_key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(base_key, num_envs)
    _, states = jax.vmap(env.reset)(reset_keys)
    return env, states


def _to_ale_gym_env_id(env_name: str) -> str:
    normalized = str(env_name).strip()
    if normalized.startswith("ALE/"):
        return normalized

    tokenized = _tokenize_game_name(normalized)
    game_id = _to_camel_case(tokenized)
    return f"ALE/{game_id}-v5"


def _strip_ale_prefix_suffix(game_name: str) -> str:
    normalized = str(game_name).strip()
    if normalized.startswith("ALE/"):
        normalized = normalized[4:]
    if normalized.lower().endswith("-v5"):
        normalized = normalized[:-3]
    return normalized


def _tokenize_game_name(game_name: str) -> List[str]:
    base_name = _strip_ale_prefix_suffix(game_name)
    if not base_name:
        raise ValueError(f"Invalid ALE env name: '{game_name}'")

    if re.search(r"[-_\s]", base_name):
        tokens = [token for token in re.split(r"[-_\s]+", base_name) if token]
        if tokens:
            return tokens

    # Split CamelCase and acronym-ish names (e.g., MontezumaRevenge, MsPacman).
    camel_tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", base_name)
    if camel_tokens:
        return camel_tokens

    return [base_name]


def _to_camel_case(tokens: List[str]) -> str:
    return "".join(token[:1].upper() + token[1:].lower() for token in tokens)


def _make_gym_ale_env(
    gym_env_id: str,
    seed: int,
    atari_frame_skip: int,
):
    def thunk():
        env = gym.make(gym_env_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=atari_frame_skip)
        env = EpisodicLifeEnv(env)
        action_meanings = getattr(env.unwrapped, "get_action_meanings", lambda: [])()
        if "FIRE" in action_meanings:
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        grayscale_wrapper = getattr(gym.wrappers, "GrayScaleObservation", None)
        if grayscale_wrapper is None:
            grayscale_wrapper = getattr(gym.wrappers, "GrayscaleObservation")
        env = grayscale_wrapper(env)
        framestack_wrapper = getattr(gym.wrappers, "FrameStack", None)
        if framestack_wrapper is None:
            framestack_wrapper = getattr(gym.wrappers, "FrameStackObservation")
        env = framestack_wrapper(env, 4)
        env.action_space.seed(seed)
        return env

    return thunk


def _run_gym_ale_benchmark(
    env_name: str,
    num_envs: int,
    num_steps: int,
    atari_frame_skip: int,
    seed: int,
    given_action: int,
) -> Dict[str, Any]:
    gym_env_id = _to_ale_gym_env_id(env_name)
    envs = gym.vector.SyncVectorEnv(
        [_make_gym_ale_env(gym_env_id, seed + i, atari_frame_skip) for i in range(num_envs)]
    )
    try:
        envs.reset(seed=seed)

        actions = np.full((num_envs,), given_action, dtype=np.int32)
        start = time.perf_counter()
        for _ in range(num_steps):
            _, _, _, _, _ = envs.step(actions)
        runtime_s = time.perf_counter() - start
    finally:
        envs.close()

    total_env_steps = num_envs * num_steps * atari_frame_skip
    throughput = total_env_steps / runtime_s
    return {
        "compile_s": 0.0,
        "runtime_s": runtime_s,
        "total_env_steps": total_env_steps,
        "throughput_env_steps_per_sec": throughput,
    }


def _run_ale_vectorenv_benchmark(
    env_name: str,
    num_envs: int,
    num_steps: int,
    atari_frame_skip: int,
    seed: int,
    given_action: int,
    async_stepping: bool,
    batch_size: int,
    thread_affinity_offset: Optional[int],
) -> Dict[str, Any]:
    from ale_py.vector_env import AtariVectorEnv

    if thread_affinity_offset is None:
        envs = AtariVectorEnv(
            game=env_name,
            num_envs=num_envs,
            batch_size=batch_size,
            frameskip=atari_frame_skip,
            stack_num=4,
            grayscale=True,
            img_height=84,
            img_width=84,
        )
    else:
        envs = AtariVectorEnv(
            game=env_name,
            num_envs=num_envs,
            batch_size=batch_size,
            frameskip=atari_frame_skip,
            stack_num=4,
            grayscale=True,
            img_height=84,
            img_width=84,
            thread_affinity_offset=int(thread_affinity_offset),
        )

    try:
        envs.reset(seed=seed)

        if async_stepping:
            actions = np.full((batch_size,), given_action, dtype=np.int32)
            actual_steps = 0
            start = time.perf_counter()
            while actual_steps < num_envs * num_steps:
                envs.send(actions)
                recv_result = envs.recv()
                infos = recv_result[-1] if recv_result else {}
                if isinstance(infos, dict) and "env_id" in infos:
                    actual_steps += len(infos["env_id"])
                else:
                    actual_steps += batch_size
            runtime_s = time.perf_counter() - start
            total_env_steps = actual_steps * atari_frame_skip
        else:
            actions = np.full((batch_size,), given_action, dtype=np.int32)
            actual_steps = 0
            start = time.perf_counter()
            target_steps = num_envs * num_steps
            while actual_steps < target_steps:
                envs.step(actions)
                actual_steps += batch_size
            runtime_s = time.perf_counter() - start
            total_env_steps = actual_steps * atari_frame_skip
    finally:
        envs.close()

    throughput = total_env_steps / runtime_s
    return {
        "compile_s": 0.0,
        "runtime_s": runtime_s,
        "total_env_steps": total_env_steps,
        "throughput_env_steps_per_sec": throughput,
    }


def _open_results_csv_writer(csv_path: str, fieldnames: List[str]):
    abs_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    csv_file = open(abs_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    return abs_path, csv_file, writer


def _append_timestamp_to_csv_path(csv_path: str) -> str:
    root, ext = os.path.splitext(csv_path)
    if not ext:
        ext = ".csv"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{root}_{timestamp}{ext}"


def _write_results_csv_row(csv_file, csv_writer, row: Dict[str, Any]):
    csv_writer.writerow(row)
    csv_file.flush()


def _build_result_row(
    backend: str,
    env_name: str,
    atari_frame_skip: int,
    num_envs: int,
    num_steps: int,
    *,
    jaxatari_mode: str = "",
    pixel_options: str = "",
    ale_vectorenv_async_stepping: str = "",
    jax_platform: str = "",
    compile_s: float = float("nan"),
    runtime_s: float = float("nan"),
    total_env_steps: float = float("nan"),
    throughput_env_steps_per_sec: float = float("nan"),
    status: str = "ok",
    error: str = "",
) -> Dict[str, Any]:
    return {
        "backend": backend,
        "env_name": env_name,
        "jaxatari_mode": jaxatari_mode,
        "pixel_options": pixel_options,
        "ale_vectorenv_async_stepping": ale_vectorenv_async_stepping,
        "atari_frame_skip": atari_frame_skip,
        "jax_platform": jax_platform,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "total_env_steps": total_env_steps,
        "compile_s": compile_s,
        "runtime_s": runtime_s,
        "throughput_env_steps_per_sec": throughput_env_steps_per_sec,
        "status": status,
        "error": error,
    }


def _normalize_frame_skip_values(frame_skip_config: Any) -> List[int]:
    raw_values = frame_skip_config if isinstance(frame_skip_config, (list, tuple)) else [frame_skip_config]
    frame_skips = [int(value) for value in raw_values]
    if not frame_skips:
        raise ValueError("ATARI_FRAME_SKIP must contain at least one value")
    for frame_skip in frame_skips:
        if frame_skip < 1:
            raise ValueError(f"ATARI_FRAME_SKIP values must be >= 1, got {frame_skip}")
    return frame_skips


def _to_str_list(config_value: Any) -> List[str]:
    if config_value is None:
        return []
    if isinstance(config_value, (list, tuple)):
        return [str(value) for value in config_value]
    return [str(config_value)]


def _normalize_bool_list(config_value: Any, default: bool) -> List[bool]:
    raw_values = config_value if isinstance(config_value, (list, tuple)) else [config_value]
    normalized: List[bool] = []
    for value in raw_values:
        if value is None:
            normalized.append(default)
            continue
        if isinstance(value, bool):
            normalized.append(value)
            continue
        if isinstance(value, (int, float)):
            normalized.append(bool(value))
            continue

        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            normalized.append(True)
            continue
        if text in {"0", "false", "f", "no", "n", "off"}:
            normalized.append(False)
            continue
        raise ValueError(f"Cannot parse boolean value: {value}")

    if not normalized:
        normalized = [default]

    deduped: List[bool] = []
    seen = set()
    for value in normalized:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _resolve_benchmark_targets(config: Dict[str, Any]) -> List[Dict[str, str]]:
    game_names = _to_str_list(
        config.get("GAME_NAMES")
        or config.get("ENV_NAMES")
        or config.get("GAME_NAME")
        or config.get("ENV_NAME")
    )
    if not game_names:
        raise ValueError("Provide at least one env name via GAME_NAME/GAME_NAMES (or ENV_NAME/ENV_NAMES)")

    return [
        {
            "game_name_input": game_name,
            "game_name_jaxatari": "".join(token.lower() for token in _tokenize_game_name(game_name)),
            "game_name_ale_vectorenv": "_".join(token.lower() for token in _tokenize_game_name(game_name)),
            "game_name_ale_standard": _to_camel_case(_tokenize_game_name(game_name)),
        }
        for game_name in game_names
    ]


def _default_ale_vectorenv_batch_size(num_envs: int) -> int:
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    # Default to full batch when no preferred power-of-two value fits.
    return num_envs


def run_throughput_benchmark(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    env_counts = [int(x) for x in config["ENV_COUNTS"]]
    num_steps = int(config["NUM_STEPS"])
    benchmark_targets = _resolve_benchmark_targets(config)
    requested_backends = [str(x).lower() for x in config.get("BENCHMARK_BACKENDS", [JAXATARI_BACKEND])]
    backends = [GYM_ALE_BACKEND if backend == ALE_BACKEND_LEGACY else backend for backend in requested_backends]
    jaxatari_modes = [str(x).lower() for x in config.get("JAXATARI_MODES", [JAX_MODE_OC, JAX_MODE_PIXEL])]
    jaxatari_platforms = [str(x).lower() for x in config.get("JAXATARI_PLATFORMS", [GPU_PLATFORM])]
    atari_frame_skips = _normalize_frame_skip_values(config.get("ATARI_FRAME_SKIP", 4))
    pixel_resize_shape = tuple(config.get("PIXEL_RESIZE_SHAPE", [84, 84]))
    pixel_option_combinations = _expand_pixel_option_combinations(config)
    base_seed = int(config["SEED"])
    given_action = int(config["GIVEN_ACTION"])
    save_results_csv = bool(config.get("SAVE_RESULTS_CSV", False))
    results_csv_path = str(config.get("RESULTS_CSV_PATH", "./scripts/benchmarks/outputs/throughput_results.csv"))
    gym_ale_timeout_s = int(config.get("GYM_ALE_TIMEOUT_S", config.get("ALE_TIMEOUT_S", 3600)))
    ale_vectorenv_async_values = _normalize_bool_list(config.get("ALE_VECTORENV_ASYNC_STEPPING", False), default=False)
    ale_vectorenv_batch_size_override = config.get("ALE_VECTORENV_BATCH_SIZE", None)
    ale_vectorenv_thread_affinity_offset = config.get("ALE_VECTORENV_THREAD_AFFINITY_OFFSET", None)

    if JAXATARI_BACKEND in backends:
        jax, _ = _get_jax_modules()
        device_list = jax.devices() + jax.devices(backend="cpu")
        available_platforms = {device.platform for device in device_list}

    if JAXATARI_BACKEND in backends:
        for platform in jaxatari_platforms:
            if platform not in (CPU_PLATFORM, GPU_PLATFORM):
                raise ValueError(f"Unknown JAX platform '{platform}'. Expected one of [{CPU_PLATFORM}, {GPU_PLATFORM}]")
            if platform not in available_platforms:
                raise ValueError(f"Requested JAX platform '{platform}' not available. Available: {sorted(available_platforms)}")

        for mode in jaxatari_modes:
            if mode not in (JAX_MODE_OC, JAX_MODE_PIXEL):
                raise ValueError(f"Unknown JAXAtari mode '{mode}'. Expected one of [{JAX_MODE_OC}, {JAX_MODE_PIXEL}]")

    gym.register_envs(ale_py)

    run = None
    # run = wandb.init(
    #     entity=config["ENTITY"],
    #     project=config["PROJECT"],
    #     mode=config["WANDB_MODE"],
    #     name=config.get("RUN_NAME", "throughput_analysis"),
    #     tags=list(config.get("TAGS", [])),
    #     notes=config.get("NOTES", ""),
    #     config=config,
    # )

    results: List[Dict[str, Any]] = []
    run_step = 0
    csv_output_path = ""
    csv_file = None
    csv_writer = None
    if save_results_csv:
        csv_output_path = _append_timestamp_to_csv_path(results_csv_path)
        print("Writing results to CSV at:", csv_output_path)
        csv_output_path, csv_file, csv_writer = _open_results_csv_writer(csv_output_path, RESULT_COLUMNS)

    def _record_result_row(row: Dict[str, Any]):
        nonlocal run_step
        results.append(row)
        if csv_file is not None and csv_writer is not None:
            _write_results_csv_row(csv_file, csv_writer, row)

        if run is not None:
            wandb_payload = {f"benchmark/{column}": row[column] for column in RESULT_COLUMNS}
            wandb.log(wandb_payload, step=run_step)

        status = str(row.get("status", "ok"))
        details = []
        if row.get("jaxatari_mode"):
            details.append(f"mode={row['jaxatari_mode']}")
        if row.get("pixel_options"):
            details.append(f"pixel={row['pixel_options']}")
        if row.get("ale_vectorenv_async_stepping"):
            details.append(f"async={row['ale_vectorenv_async_stepping']}")
        if row.get("jax_platform"):
            details.append(f"platform={row['jax_platform']}")

        prefix = (
            f"backend={row['backend']:8s} | env={row['env_name']:10s} | "
            f"fs={int(row['atari_frame_skip']):2d} | envs={int(row['num_envs']):4d} | steps={int(row['num_steps'])}"
        )
        details_part = f" | {' | '.join(details)}" if details else ""
        if status == "ok":
            print(
                f"{prefix}{details_part} | compile={float(row['compile_s']):.3f}s | "
                f"run={float(row['runtime_s']):.3f}s | throughput={float(row['throughput_env_steps_per_sec']):,.2f} env-steps/s"
            )
        else:
            print(f"{prefix}{details_part} | status={status} | ERROR={row.get('error', '')}")

        run_step += 1

    for target in benchmark_targets:
        game_name_jaxatari = target["game_name_jaxatari"]
        game_name_ale_vectorenv = target["game_name_ale_vectorenv"]
        game_name_ale_standard = target["game_name_ale_standard"]

        for atari_frame_skip in atari_frame_skips:
            for backend in backends:
                if backend not in (
                    JAXATARI_BACKEND,
                    GYM_ALE_BACKEND,
                    ALE_VECTORENV_BACKEND,
                ):
                    raise ValueError(
                        "Unknown backend "
                        f"'{backend}'. Expected one of: [{JAXATARI_BACKEND}, {GYM_ALE_BACKEND}, {ALE_VECTORENV_BACKEND}]"
                    )

                for num_envs in env_counts:
                    if ale_vectorenv_batch_size_override is None:
                        ale_vectorenv_batch_size = _default_ale_vectorenv_batch_size(num_envs)
                    else:
                        ale_vectorenv_batch_size = int(ale_vectorenv_batch_size_override)

                    if ale_vectorenv_batch_size < 1:
                        raise ValueError(f"ALE_VECTORENV_BATCH_SIZE must be >= 1, got {ale_vectorenv_batch_size}")
                    if ale_vectorenv_batch_size > num_envs:
                        raise ValueError(
                            "ALE_VECTORENV_BATCH_SIZE must be <= num_envs "
                            f"(num_envs={num_envs}), got {ale_vectorenv_batch_size}"
                        )

                    if backend == JAXATARI_BACKEND:
                        for jax_mode in jaxatari_modes:
                            mode_pixel_combos = pixel_option_combinations if jax_mode == JAX_MODE_PIXEL else [tuple()]

                            for pixel_options in mode_pixel_combos:
                                pixel_options_str = "+".join(pixel_options)

                                for jax_platform in jaxatari_platforms:
                                    env_seed = base_seed + run_step
                                    try:
                                        env, states = _prepare_jaxatari_states(
                                            game_name=game_name_jaxatari,
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
                                        row = _build_result_row(
                                            backend=backend,
                                            env_name=game_name_jaxatari,
                                            atari_frame_skip=atari_frame_skip,
                                            num_envs=num_envs,
                                            num_steps=num_steps,
                                            jaxatari_mode=jax_mode,
                                            pixel_options=pixel_options_str,
                                            jax_platform=jax_platform,
                                            compile_s=compile_s,
                                            runtime_s=runtime_s,
                                            total_env_steps=total_env_steps,
                                            throughput_env_steps_per_sec=throughput,
                                        )
                                    except Exception as error:
                                        row = _build_result_row(
                                            backend=backend,
                                            env_name=game_name_jaxatari,
                                            atari_frame_skip=atari_frame_skip,
                                            num_envs=num_envs,
                                            num_steps=num_steps,
                                            jaxatari_mode=jax_mode,
                                            pixel_options=pixel_options_str,
                                            jax_platform=jax_platform,
                                            status="error",
                                            error=f"{type(error).__name__}: {error}",
                                        )
                                    _record_result_row(row)
                        continue

                    if backend == GYM_ALE_BACKEND:
                        env_seed = base_seed + run_step
                        try:
                            ale_metrics = _run_with_timeout(
                                _run_gym_ale_benchmark,
                                timeout_seconds=gym_ale_timeout_s,
                                context=f"Gym ALE benchmark (envs={num_envs}, frame_skip={atari_frame_skip})",
                                env_name=game_name_ale_standard,
                                num_envs=num_envs,
                                num_steps=num_steps,
                                atari_frame_skip=atari_frame_skip,
                                seed=env_seed,
                                given_action=given_action,
                            )
                            compile_s = ale_metrics["compile_s"]
                            runtime_s = ale_metrics["runtime_s"]
                            total_env_steps = ale_metrics["total_env_steps"]
                            throughput = ale_metrics["throughput_env_steps_per_sec"]
                            row = _build_result_row(
                                backend=backend,
                                env_name=game_name_ale_standard,
                                atari_frame_skip=atari_frame_skip,
                                num_envs=num_envs,
                                num_steps=num_steps,
                                compile_s=compile_s,
                                runtime_s=runtime_s,
                                total_env_steps=total_env_steps,
                                throughput_env_steps_per_sec=throughput,
                            )
                        except Exception as error:
                            row = _build_result_row(
                                backend=backend,
                                env_name=game_name_ale_standard,
                                atari_frame_skip=atari_frame_skip,
                                num_envs=num_envs,
                                num_steps=num_steps,
                                status="error",
                                error=f"{type(error).__name__}: {error}",
                            )
                        _record_result_row(row)
                        continue

                    if backend == ALE_VECTORENV_BACKEND:
                        for ale_vectorenv_async in ale_vectorenv_async_values:
                            env_seed = base_seed + run_step
                            async_str = "true" if ale_vectorenv_async else "false"
                            try:
                                ale_metrics = _run_with_timeout(
                                    _run_ale_vectorenv_benchmark,
                                    timeout_seconds=gym_ale_timeout_s,
                                    context=(
                                        "ALE VectorEnv benchmark "
                                        f"(envs={num_envs}, frame_skip={atari_frame_skip}, async={async_str})"
                                    ),
                                    env_name=game_name_ale_vectorenv,
                                    num_envs=num_envs,
                                    num_steps=num_steps,
                                    atari_frame_skip=atari_frame_skip,
                                    seed=env_seed,
                                    given_action=given_action,
                                    async_stepping=ale_vectorenv_async,
                                    batch_size=ale_vectorenv_batch_size,
                                    thread_affinity_offset=ale_vectorenv_thread_affinity_offset,
                                )
                                compile_s = ale_metrics["compile_s"]
                                runtime_s = ale_metrics["runtime_s"]
                                total_env_steps = ale_metrics["total_env_steps"]
                                throughput = ale_metrics["throughput_env_steps_per_sec"]
                                row = _build_result_row(
                                    backend=backend,
                                    env_name=game_name_ale_vectorenv,
                                    atari_frame_skip=atari_frame_skip,
                                    num_envs=num_envs,
                                    num_steps=num_steps,
                                    ale_vectorenv_async_stepping=async_str,
                                    compile_s=compile_s,
                                    runtime_s=runtime_s,
                                    total_env_steps=total_env_steps,
                                    throughput_env_steps_per_sec=throughput,
                                )
                            except Exception as error:
                                row = _build_result_row(
                                    backend=backend,
                                    env_name=game_name_ale_vectorenv,
                                    atari_frame_skip=atari_frame_skip,
                                    num_envs=num_envs,
                                    num_steps=num_steps,
                                    ale_vectorenv_async_stepping=async_str,
                                    status="error",
                                    error=f"{type(error).__name__}: {error}",
                                )
                            _record_result_row(row)
                        continue

                    raise RuntimeError(f"Unhandled backend: {backend}")

    if run is not None:
        table = wandb.Table(
            columns=RESULT_COLUMNS
        )
        for row in results:
            table.add_data(
                row["backend"],
                row["env_name"],
                row["jaxatari_mode"],
                row["pixel_options"],
                row["ale_vectorenv_async_stepping"],
                row["atari_frame_skip"],
                row["jax_platform"],
                row["num_envs"],
                row["num_steps"],
                row["total_env_steps"],
                row["compile_s"],
                row["runtime_s"],
                row["throughput_env_steps_per_sec"],
                row["status"],
                row["error"],
            )
        # wandb.log({"benchmark/results_table": table})

        if save_results_csv:
            print(f"Saved results CSV: {csv_output_path}")

        run.finish()

    if csv_file is not None:
        csv_file.close()

    return results


@hydra.main(version_base=None, config_path="./config", config_name="throughput_analysis")
def main(cfg: DictConfig):
    # print("Config:\n", OmegaConf.to_yaml(cfg))
    config = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    run_throughput_benchmark(config)


if __name__ == "__main__":
    main()
