import os
import csv
import time
import importlib
from itertools import combinations
from typing import Any, Dict, List, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import ale_py
from omegaconf import DictConfig, OmegaConf
import wandb

import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper, PixelObsWrapper

JAXATARI_BACKEND = "jaxatari"
ALE_BACKEND = "ale"
GXM_BACKEND = "gxm"
JAX_MODE_OC = "oc"
JAX_MODE_PIXEL = "pixel"
PIXEL_OPT_RESIZED = "resized"
PIXEL_OPT_GRAYSCALE = "grayscale"
PIXEL_OPT_NATIVE = "native"
CPU_PLATFORM = "cpu"
GPU_PLATFORM = "gpu"


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


def _run_ale_benchmark(
    ale_env_id: str,
    num_envs: int,
    num_steps: int,
    seed: int,
    given_action: int,
) -> Dict[str, Any]:
    env_fns = [lambda: gym.make(ale_env_id) for _ in range(num_envs)]
    vector_env = gym.vector.SyncVectorEnv(env_fns)
    vector_env.reset(seed=seed)

    actions = np.full((num_envs,), given_action, dtype=np.int32)
    start = time.perf_counter()
    for _ in range(num_steps):
        _, _, _, _, _ = vector_env.step(actions)
    runtime_s = time.perf_counter() - start
    vector_env.close()

    total_env_steps = num_envs * num_steps
    throughput = total_env_steps / runtime_s
    return {
        "compile_s": 0.0,
        "runtime_s": runtime_s,
        "total_env_steps": total_env_steps,
        "throughput_env_steps_per_sec": throughput,
    }


def _create_gxm_env(
    gxm_env_id: str,
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
            lambda: gxm_module.make(gxm_target_env),
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
    seed: int,
    given_action: int,
    gxm_platform: str,
) -> Dict[str, Any]:
    env, gxm_impl = _create_gxm_env(
        gxm_env_id=gxm_env_id,
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

    total_env_steps = num_envs * num_steps
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


def run_throughput_benchmark(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    env_counts = [int(x) for x in config["ENV_COUNTS"]]
    num_steps = int(config["NUM_STEPS"])
    game_name = str(config["GAME_NAME"]).lower()
    backends = [str(x).lower() for x in config.get("BENCHMARK_BACKENDS", [JAXATARI_BACKEND])]
    jaxatari_modes = [str(x).lower() for x in config.get("JAXATARI_MODES", [JAX_MODE_OC, JAX_MODE_PIXEL])]
    jaxatari_platforms = [str(x).lower() for x in config.get("JAXATARI_PLATFORMS", [GPU_PLATFORM])]
    atari_frame_skip = int(config.get("ATARI_FRAME_SKIP", 4))
    pixel_resize_shape = tuple(config.get("PIXEL_RESIZE_SHAPE", [84, 84]))
    pixel_option_combinations = _expand_pixel_option_combinations(config)
    ale_env_id = str(config.get("ALE_ENV_ID") or _default_ale_env_id(game_name))
    gxm_env_id = str(config.get("GXM_ENV_ID") or _default_gxm_env_id(game_name))
    gxm_env_type = str(config.get("GXM_ENV_TYPE", "envpool"))
    gxm_platforms = [str(x).lower() for x in config.get("GXM_PLATFORMS", [GPU_PLATFORM])]
    base_seed = int(config["SEED"])
    given_action = int(config["GIVEN_ACTION"])
    save_results_csv = bool(config.get("SAVE_RESULTS_CSV", False))
    results_csv_path = str(config.get("RESULTS_CSV_PATH", "./scripts/benchmarks/outputs/throughput_results.csv"))

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

    for backend in backends:
        if backend not in (JAXATARI_BACKEND, ALE_BACKEND, GXM_BACKEND):
            raise ValueError(
                f"Unknown backend '{backend}'. Expected one of: [{JAXATARI_BACKEND}, {ALE_BACKEND}, {GXM_BACKEND}]"
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
                            total_env_steps = num_envs * num_steps
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
                                f"backend={backend:8s} | mode={jax_mode:5s} | pixel={pixel_options_str or '-':15s} | "
                                f"platform={jax_platform:3s} | envs={num_envs:4d} | steps={num_steps} | "
                                f"compile={compile_s:.3f}s | run={runtime_s:.3f}s | "
                                f"throughput={throughput:,.2f} env-steps/s"
                            )

                            run_step += 1
                continue
            elif backend == ALE_BACKEND:
                env_seed = base_seed + run_step
                ale_metrics = _run_ale_benchmark(
                    ale_env_id=ale_env_id,
                    num_envs=num_envs,
                    num_steps=num_steps,
                    seed=env_seed,
                    given_action=given_action,
                )
                compile_s = ale_metrics["compile_s"]
                runtime_s = ale_metrics["runtime_s"]
                total_env_steps = ale_metrics["total_env_steps"]
                throughput = ale_metrics["throughput_env_steps_per_sec"]
                gxm_impl = ""
            else:
                for gxm_platform in gxm_platforms:
                    env_seed = base_seed + run_step
                    gxm_metrics = _run_gxm_benchmark(
                        gxm_env_id=gxm_env_id,
                        num_envs=num_envs,
                        num_steps=num_steps,
                        seed=env_seed,
                        given_action=given_action,
                        gxm_platform=gxm_platform,
                    )
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
                        f"backend={backend:8s} | impl={gxm_impl:16s} | platform={gxm_platform:3s} | "
                        f"envs={num_envs:4d} | steps={num_steps} | "
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
