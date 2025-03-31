from pathlib import Path

import jax
import jax.numpy as jnp
import time
import numpy as np
from functools import partial
import multiprocessing as mp
from typing import Tuple, Dict
import psutil
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
import threading
from jax_game import JAXAtari
from ocatari import OCAtari


class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_percentages = []
        self.memory_percentages = []
        self.gpu_utilization = []
        self.gpu_memory = []
        self._stop = False

    def _get_gpu_info(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                util, mem = map(float, result.stdout.strip().split(","))
                return util, mem
            return 0.0, 0.0
        except:
            try:
                result = subprocess.run(
                    [
                        'rocm-smi -u --showmemuse --json | jq -rc \'.card0["GPU use (%)"], .card0["GPU Memory Allocated (VRAM%)"]\'',
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    util = float(lines[0])
                    mem = float(lines[1])
                return 0.0, 0.0
            except:
                return 0.0, 0.0

    def monitor(self):
        while not self._stop:
            self.cpu_percentages.append(psutil.cpu_percent(interval=None))
            self.memory_percentages.append(psutil.virtual_memory().percent)
            gpu_util, gpu_mem = self._get_gpu_info()
            self.gpu_utilization.append(gpu_util)
            self.gpu_memory.append(gpu_mem)
            time.sleep(self.interval)

    def start(self):
        self._stop = False
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.start()

    def stop(self):
        self._stop = True
        self.monitor_thread.join()

    def get_averages(self):
        return {
            "cpu_avg": np.mean(self.cpu_percentages),
            "memory_avg": np.mean(self.memory_percentages),
            "gpu_util_avg": np.mean(self.gpu_utilization),
            "gpu_memory_avg": np.mean(self.gpu_memory),
        }


def run_parallel_jax(
    game_name,
    num_steps: int = 1_000_000,
    num_envs: int = 2000,
    num_actions: int = None,
    render: bool = False,
) -> Tuple[float, float, int, Dict]:
    monitor = ResourceMonitor()
    monitor.start()

    jaxgame = JAXAtari(game_name)
    init_state = jaxgame.get_init_state()
    states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), init_state)

    if render:

        @partial(jax.vmap, in_axes=(0, 0))
        def parallel_step(states, actions):
            return jaxgame.step_with_render(states, actions)

    else:

        @partial(jax.vmap, in_axes=(0, 0))
        def parallel_step(states, actions):
            return jaxgame.step_state_only(states, actions)

    jit_parallel_step = jax.jit(parallel_step)
    rng_key = jax.random.PRNGKey(0)

    @jax.jit
    def run_one_step(carry, _):
        states, rng_key = carry
        rng_key, action_key = jax.random.split(rng_key)
        actions = jax.random.randint(
            action_key, shape=(num_envs,), minval=0, maxval=num_actions
        )
        next_states = jit_parallel_step(states, actions)
        return (next_states, rng_key), None

    # run a warmup of about 1000 steps
    warmup_steps = 1000
    (states, _), _ = jax.lax.scan(
        run_one_step, (states, rng_key), None, length=warmup_steps
    )

    steps_per_env = num_steps // num_envs
    start_time = time.time()

    (final_states, _), _ = jax.lax.scan(
        run_one_step, (states, rng_key), None, length=steps_per_env
    )

    total_time = time.time() - start_time
    total_steps = steps_per_env * num_envs
    steps_per_second = total_steps / total_time

    monitor.stop()
    resource_usage = monitor.get_averages()

    return total_time, steps_per_second, total_steps, resource_usage


def run_ocatari_worker(steps_per_env: int, game_name: str, num_actions: int) -> int:
    env = OCAtari(game_name, frameskip=1, mode="ram")
    env.reset()
    completed_steps = 0
    for _ in range(steps_per_env):
        action = np.random.randint(0, num_actions)
        env.step(action)
        completed_steps += 1
    return completed_steps


def run_parallel_ocatari(
    game_name, num_steps: int = 1_000_000, num_envs: int = None, num_actions: int = None
) -> Tuple[float, float, int, Dict]:
    if num_envs is None:
        num_envs = mp.cpu_count()

    steps_per_env = num_steps // num_envs
    monitor = ResourceMonitor()
    monitor.start()
    start_time = time.time()

    run_ocatari_worker_set = partial(
        run_ocatari_worker, game_name=game_name, num_actions=num_actions
    )
    with mp.Pool(processes=num_envs) as pool:
        results = pool.map(run_ocatari_worker_set, [steps_per_env] * num_envs)

    total_time = time.time() - start_time
    total_steps = sum(results)
    steps_per_second = total_steps / total_time

    monitor.stop()
    resource_usage = monitor.get_averages()

    return total_time, steps_per_second, total_steps, resource_usage


def run_scaling_benchmarks(
    jax_game_name,
    oc_atari_game_name,
    num_steps: int = 1_000_000,
    num_actions: int = None,
    render: bool = False,
):
    # CPU scaling (OCAtari)
    cpu_workers = [1, 2, 4, 8, 16]
    cpu_results = []
    print("\nRunning OCAtari scaling tests...")
    for workers in cpu_workers:
        if workers <= mp.cpu_count():
            print(f"Testing with {workers} workers...")
            results = run_parallel_ocatari(
                oc_atari_game_name,
                num_steps=num_steps,
                num_envs=workers,
                num_actions=num_actions,
            )
            cpu_results.append(results)
    cpu_workers = cpu_workers[: len(cpu_results)]

    # GPU scaling (JAX)
    gpu_workers = [32, 64, 128, 1024, 4096]
    gpu_results = []
    print("\nRunning JAX scaling tests...")
    for workers in gpu_workers:
        print(f"Testing with {workers} parallel environments...")
        results = run_parallel_jax(
            jax_game_name,
            num_steps=num_steps,
            num_envs=workers,
            num_actions=num_actions,
            render=render,
        )
        gpu_results.append(results)

    return cpu_workers, cpu_results, gpu_workers, gpu_results


def plot_benchmark_comparison(jax_results, ocatari_results, timestamp, name):
    metrics = {
        "Time (s)": (0, "Time (seconds)"),
        "Steps per Second": (1, "Steps per Second"),
        "CPU Usage (%)": (lambda x: x[3]["cpu_avg"], "Percentage"),
        "Memory Usage (%)": (lambda x: x[3]["memory_avg"], "Percentage"),
        "GPU Utilization (%)": (lambda x: x[3]["gpu_util_avg"], "Percentage"),
        "GPU Memory (MB)": (lambda x: x[3]["gpu_memory_avg"], "Memory (MB)"),
    }

    # Create individual plots
    for metric_name, (metric_idx, ylabel) in metrics.items():
        plt.figure(figsize=(8, 6))
        data = []
        for impl, results in [("JAX", jax_results), ("OCAtari", ocatari_results)]:
            if callable(metric_idx):
                value = metric_idx(results)
            else:
                value = results[metric_idx]
            data.append(value)

        plt.bar(["JAX", "OCAtari"], data)
        plt.title(f"{metric_name} Comparison")
        plt.ylabel(ylabel)
        plt.savefig(
            f'./results/{name}/benchmark_{metric_name.lower().replace(" ", "_")}_{timestamp}_{name}.png'
        )
        plt.close()

    # Combined plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (metric_name, (metric_idx, ylabel)) in enumerate(metrics.items()):
        data = []
        for impl, results in [("JAX", jax_results), ("OCAtari", ocatari_results)]:
            if callable(metric_idx):
                value = metric_idx(results)
            else:
                value = results[metric_idx]
            data.append(value)

        axes[i].bar(["JAX", "OCAtari"], data)
        axes[i].set_title(metric_name)
        axes[i].set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f"./results/{name}/benchmark_combined_{timestamp}_{name}.png")
    plt.close()


def plot_scaling_results(
    cpu_workers, cpu_results, gpu_workers, gpu_results, timestamp, name
):
    metrics = {"Time": (0, "Time (seconds)"), "Throughput": (1, "Steps per Second")}

    # Create individual scaling plots
    for metric_name, (metric_idx, ylabel) in metrics.items():
        plt.figure(figsize=(10, 6))

        # Plot CPU scaling (OCAtari)
        cpu_values = [results[metric_idx] for results in cpu_results]
        plt.plot(cpu_workers, cpu_values, "b-o", label="OCAtari (CPU)")

        # Plot GPU scaling (JAX)
        gpu_values = [results[metric_idx] for results in gpu_results]
        plt.plot(gpu_workers, gpu_values, "r-o", label="JAX (GPU)")

        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Number of Workers/Environments")
        plt.ylabel(ylabel)
        plt.title(f"{metric_name} Scaling Comparison")
        plt.legend()
        plt.grid(True)

        # Add value annotations
        for x, y in zip(cpu_workers, cpu_values):
            plt.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
        for x, y in zip(gpu_workers, gpu_values):
            plt.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
            )

        plt.savefig(
            f"./results/{name}/scaling_{metric_name.lower()}_{timestamp}_{name}.png"
        )
        plt.close()

    # Combined scaling plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, (metric_name, (metric_idx, ylabel)) in enumerate(metrics.items()):
        # Plot CPU scaling
        cpu_values = [results[metric_idx] for results in cpu_results]
        axes[i].plot(cpu_workers, cpu_values, "b-o", label="OCAtari (CPU)")

        # Plot GPU scaling
        gpu_values = [results[metric_idx] for results in gpu_results]
        axes[i].plot(gpu_workers, gpu_values, "r-o", label="JAX (GPU)")

        axes[i].set_xscale("log", base=2)
        axes[i].set_yscale("log")
        axes[i].set_xlabel("Number of Workers/Environments")
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(f"{metric_name} Scaling")
        axes[i].legend()
        axes[i].grid(True)

        # Add value annotations
        for x, y in zip(cpu_workers, cpu_values):
            axes[i].annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
        for x, y in zip(gpu_workers, gpu_values):
            axes[i].annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
            )

    plt.tight_layout()
    plt.savefig(f"./results/{name}/scaling_combined_{timestamp}_{name}.png")
    plt.close()


def print_benchmark_results(
    name: str,
    total_time: float,
    steps_per_second: float,
    total_steps: int,
    resource_usage: Dict,
):
    folder = name.split(" ")[1]
    with open(f"./results/{folder}/logging_{timestamp}_{name}.txt", "a") as f:
        print(f"\n{name} Benchmark Results:", file=f)
        print(f"Total steps completed: {total_steps:,}", file=f)
        print(f"Total time: {total_time:.2f} seconds", file=f)
        print(f"Average steps per second: {steps_per_second:,.2f}", file=f)
        print(
            f"Microseconds per step: {(total_time * 1_000_000 / total_steps):.2f}",
            file=f,
        )
        print("\nResource Usage:", file=f)
        print(f"Average CPU Usage: {resource_usage['cpu_avg']:.1f}%", file=f)
        print(f"Average Memory Usage: {resource_usage['memory_avg']:.1f}%", file=f)
        print(f"Average GPU Utilization: {resource_usage['gpu_util_avg']:.1f}%", file=f)
        print(
            f"Average GPU Memory Usage: {resource_usage['gpu_memory_avg']:.1f} MB",
            file=f,
        )


def save_raw_files(data, path):
    name = ["total_time", "steps_per_second", "total_steps", "resource_usage"]
    for d, name in zip(data, name):
        p = path / name
        np.save(p, d)


def save_raw_files_scaling(cpu_results, gpu_results, cpu_workers, gpu_workers, path):

    atari_values_time = [results[0] for results in cpu_results]
    p = path / "atari_values_time"
    np.save(p, atari_values_time)
    atari_values_throughput = [results[0] for results in cpu_results]
    p = path / "atari_values_throughput"
    np.save(p, atari_values_throughput)
    atari_workers = cpu_workers
    p = path / "atari_workers"
    np.save(p, atari_workers)

    jax_values_time = [results[0] for results in gpu_results]
    p = path / "jax_values_time"
    np.save(p, jax_values_time)
    jax_values_throughput = [results[1] for results in gpu_results]
    p = path / "jax_values_throughput"
    np.save(p, jax_values_throughput)
    jax_workers = gpu_workers
    p = path / "jax_workers"
    np.save(p, jax_workers)


if __name__ == "__main__":
    GAMES_TO_TEST = [
        ("pong", "Pong-v4", 6),
        # ("breakout", "Breakout", 4),
        # ("kangaroo", "Kangaroo", 18),
        ("freeway", "Freeway", 3),
        ("seaquest", "Seaquest", 18),
        ("skiing", "Skiing", 3),
        ("tennis", "Tennis", 18),
    ]
    NUMBER_OF_STEPS = 1_000_000
    USE_RENDERER = False

    for game in GAMES_TO_TEST:
        jax_game_name, oc_atari_game_name, actions = game

        # Create results folder
        Path(f"./results/{jax_game_name}").mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save device information
        with open(
            f"./results/{jax_game_name}/logging_{timestamp}_{jax_game_name}.txt", "a"
        ) as f:
            print("\nSystem Information:", file=f)
            print(f"CPU cores available: {mp.cpu_count()}", file=f)
            print("Available devices:", jax.devices(), file=f)
            print("Default device:", jax.default_device(), file=f)

        # Run scaling benchmarks
        print("\nRunning scaling benchmarks...")
        cpu_workers, cpu_results, gpu_workers, gpu_results = run_scaling_benchmarks(
            jax_game_name,
            oc_atari_game_name,
            num_steps=NUMBER_OF_STEPS,
            num_actions=actions,
            render=USE_RENDERER,
        )
        Path(f"./results/{jax_game_name}/raw").mkdir(parents=True, exist_ok=True)
        save_raw_files_scaling(
            cpu_results,
            gpu_results,
            cpu_workers,
            gpu_workers,
            Path(f"./results/{jax_game_name}/raw"),
        )

        # Plot scaling results
        print("\nGenerating scaling plots...")
        plot_scaling_results(
            cpu_workers, cpu_results, gpu_workers, gpu_results, timestamp, jax_game_name
        )

        # Run standard benchmarks for detailed comparison
        print("\nRunning standard benchmarks...")
        Path(f"./results/{jax_game_name}/raw/jax").mkdir(parents=True, exist_ok=True)
        jax_results = run_parallel_jax(
            jax_game_name,
            num_actions=actions,
            render=USE_RENDERER,
            num_steps=NUMBER_OF_STEPS,
            num_envs=2000,
        )
        save_raw_files(jax_results, Path(f"./results/{jax_game_name}/raw/jax"))
        print_benchmark_results(f"JAX {jax_game_name}", *jax_results)

        Path(f"./results/{jax_game_name}/raw/oc").mkdir(parents=True, exist_ok=True)
        ocatari_results = run_parallel_ocatari(
            oc_atari_game_name,
            num_actions=actions,
            num_steps=NUMBER_OF_STEPS,
            num_envs=16,
        )
        save_raw_files(jax_results, Path(f"./results/{jax_game_name}/raw/oc"))
        print_benchmark_results(f"OCAtari {jax_game_name}", *ocatari_results)

        # Plot comparison results
        print("\nGenerating comparison plots...")
        plot_benchmark_comparison(
            jax_results, ocatari_results, timestamp, jax_game_name
        )

        print(
            "\nBenchmark complete! Check the generated plot files for visualizations."
        )
