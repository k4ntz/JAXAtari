import argparse
import csv
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import multiprocessing as mp

# It is assumed that gymnasium[atari] and ale-py are installed.
try:
    import gymnasium as gym
    from ale_py import ALEInterface
except ImportError as e:
    print(f"Fatal: Could not import gymnasium or ale_py: {e}")
    print("Please ensure 'gymnasium[atari]' is installed (`pip install gymnasium[atari] ale-py`).")
    sys.exit(1)

import numpy as np
import psutil
import gc

# --- System Information Utilities (Identical to JAX script) ---

def get_cpu_name() -> str:
    """Gets the CPU model name."""
    try:
        if sys.platform == "win32":
            return subprocess.check_output(["wmic", "cpu", "get", "name"], text=True).strip().split('\n')[1]
        elif sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif sys.platform == "linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception as e:
        print(f"Warning: Could not determine CPU name: {e}")
    return "N/A"

def get_gpu_details() -> Dict[str, Any]:
    """
    Gets GPU details like name, and total VRAM using available tools.
    Returns a dictionary with 'gpu_name' (str) and 'total_vram_mb' (float).
    """
    details = {'gpu_name': 'N/A', 'total_vram_mb': 0.0}
    try: # NVIDIA
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        name, total_mem = result.stdout.strip().split(',')
        details['gpu_name'] = name.strip()
        details['total_vram_mb'] = float(total_mem)
        return details
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    except Exception as e:
        print(f"Warning: Error querying NVIDIA GPU details: {e}")

    if details['gpu_name'] == 'N/A':
        try: # AMD
            result = subprocess.run(
                ['rocm-smi', '--showproductname', '--showmeminfo', 'VRAM', '--json'],
                capture_output=True, text=True, check=True
            )
            import json
            data = json.loads(result.stdout)
            card_key = next(iter(data))
            details['gpu_name'] = data[card_key].get('Card series', 'AMD ROCm Device')
            total_vram_str = data[card_key].get('VRAM Total Memory (B)', '0')
            details['total_vram_mb'] = int(total_vram_str) / (1024**2)
            return details
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        except Exception as e:
            print(f"Warning: Error querying AMD GPU details: {e}")

    return details


# --- Resource Monitoring (Identical to JAX script) ---

class ResourceMonitor:
    """A thread-based monitor for CPU, RAM, and GPU resources."""
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.cpu_percentages = []
        self.memory_percentages = []
        self.gpu_utilization = []
        self.gpu_memory_mb = []
        self._stop = False
        self.monitor_thread = None

    def _get_gpu_runtime_info(self) -> Tuple[float, float]:
        """Queries instantaneous GPU utilization and memory usage."""
        try: # NVIDIA
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,nounits,noheader"],
                capture_output=True, text=True, check=True
            )
            util, mem_used = map(float, result.stdout.strip().split(","))
            return util, mem_used
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        try: # AMD
            result = subprocess.run(
                ['rocm-smi', '--showuse', '--showmemuse', '--json'],
                capture_output=True, text=True, check=True
            )
            import json
            data = json.loads(result.stdout)
            card_key = next(iter(data))
            util = float(data[card_key].get('GPU use (%)', 0.0))
            mem_used_bytes = int(data[card_key].get('VRAM Total Memory (B)', 0))
            return util, mem_used_bytes / (1024**2)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

    def _monitor_loop(self):
        """The target function for the monitoring thread."""
        while not self._stop:
            self.cpu_percentages.append(psutil.cpu_percent(interval=None))
            self.memory_percentages.append(psutil.virtual_memory().percent)
            gpu_util, gpu_mem = self._get_gpu_runtime_info()
            self.gpu_utilization.append(gpu_util)
            self.gpu_memory_mb.append(gpu_mem)
            time.sleep(self.interval)

    def start(self):
        """Starts the resource monitoring thread."""
        self._stop = False
        self.cpu_percentages, self.memory_percentages = [], []
        self.gpu_utilization, self.gpu_memory_mb = [], []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self) -> Dict[str, float]:
        """Stops the monitoring thread and returns the averaged results."""
        self._stop = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()

        return {
            "cpu_percent_avg": np.mean(self.cpu_percentages) if self.cpu_percentages else 0.0,
            "ram_percent_avg": np.mean(self.memory_percentages) if self.memory_percentages else 0.0,
            "gpu_percent_avg": np.mean(self.gpu_utilization) if self.gpu_utilization else 0.0,
            "gpu_vram_mb_avg": np.mean(self.gpu_memory_mb) if self.gpu_memory_mb else 0.0,
        }

# --- ALE Benchmarking Logic ---

def run_ale_worker(worker_id: int, game_name: str, num_steps: int, use_render: bool, seed: int) -> int:
    """
    A single worker process that creates an ALE environment and runs steps.
    """
    try:
        # Note: 'ALE/Pong-v5' is the standard. We add frameskip=1 to disable it.
        # This makes the comparison to the JAX environment 1-to-1.
        env = gym.make(f"ALE/{game_name}-v5", frameskip=1, render_mode="rgb_array" if use_render else None)
        env.reset(seed=seed + worker_id)
        
        num_actions = env.action_space.n
        
        for _ in range(num_steps):
            action = np.random.randint(num_actions)
            env.step(action)
            if use_render:
                env.render() # The rendered frame is computed but not returned.
        
        env.close()
        return num_steps
    except Exception as e:
        print(f"Error in worker {worker_id} for game '{game_name}': {e}")
        return 0

def run_ale_benchmark(
    game_name: str,
    num_steps_per_env: int,
    num_envs: int,
    seed: int,
    use_render: bool,
) -> Dict[str, Any]:
    """
    Runs a parallelized ALE benchmark using multiprocessing.

    Returns:
        A dictionary containing all key benchmark metrics.
    """
    render_status = "ENABLED" if use_render else "DISABLED"
    print(f"\nALE: Benchmarking '{game_name}' with {num_envs} environments (Rendering {render_status})...")

    monitor = ResourceMonitor()
    
    # Create arguments for each worker
    worker_args = [(i, game_name, num_steps_per_env, use_render, seed) for i in range(num_envs)]

    monitor.start()
    execution_start_time = time.time()

    # Use a multiprocessing pool to run workers in parallel
    with mp.Pool(processes=num_envs) as pool:
        results = pool.starmap(run_ale_worker, worker_args)

    execution_time = time.time() - execution_start_time
    
    resource_usage = monitor.stop()
    gc.collect()

    total_steps = sum(results)
    steps_per_second = total_steps / execution_time if execution_time > 0 else 0

    return {
        "game_name": game_name,
        "spawned_instances": num_envs,
        "total_steps": total_steps,
        "total_runtime_secs": execution_time,
        "compilation_time_secs": 0.0,  # No JIT compilation for ALE
        "steps_per_second": steps_per_second,
        **resource_usage,
    }

# --- CSV Logging (Identical to JAX script) ---

CSV_HEADER = [
    "timestamp", "game_name", "rendering_enabled", "backend", "cpu_name", "gpu_name",
    "total_vram_mb", "spawned_instances", "total_steps",
    "total_runtime_secs", "compilation_time_secs", "steps_per_second",
    "cpu_percent_avg", "ram_percent_avg", "gpu_percent_avg", "gpu_vram_mb_avg"
]

def setup_csv_logging(filepath: Path) -> None:
    """Creates the CSV file and writes the header if it doesn't exist."""
    if not filepath.exists():
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Use 'backend' instead of 'jax_backend'
            header = CSV_HEADER.copy()
            if 'jax_backend' in header:
                header[header.index('jax_backend')] = 'backend'
            writer.writerow(header)
        print(f"Created log file: {filepath}")

def append_to_csv(filepath: Path, result_dict: Dict[str, Any]) -> None:
    """Appends a new row of benchmark results to the CSV file."""
    try:
        with open(filepath, 'a', newline='') as f:
            # Use 'backend' instead of 'jax_backend'
            header = CSV_HEADER.copy()
            if 'jax_backend' in header:
                header[header.index('jax_backend')] = 'backend'
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(result_dict)
    except IOError as e:
        print(f"Error: Could not write to CSV file {filepath}: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A modular ALE/Gymnasium performance benchmark script.")
    parser.add_argument(
        "--game-name", type=str, default=None,
        help="Name of the ALE game to benchmark (e.g., 'Pong', 'Breakout'). If not specified, all available games will be benchmarked."
    )
    parser.add_argument(
        "--steps-per-env", type=int, default=10_000,
        help="Number of simulation steps to run for each parallel environment."
    )
    parser.add_argument(
        "--output-file", type=str, default="./ale_benchmark_results.csv",
        help="Path to the output CSV file for storing results."
    )
    parser.add_argument(
        "--env-counts", type=int, nargs='+', default=[2, 4, 8, 16],
        help="A space-separated list of parallel environment counts to test for scaling."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility."
    )
    args = parser.parse_args()

    # --- Setup ---
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_csv_logging(output_path)

    # --- Collect Static System Info ---
    print("Collecting system information...")
    static_info = {
        "backend": "Gymnasium/ALE",
        "cpu_name": get_cpu_name(),
        **get_gpu_details()
    }
    print(f"  Backend:     {static_info['backend']}")
    print(f"  CPU Name:    {static_info['cpu_name']}")
    print(f"  GPU Name:    {static_info['gpu_name']}")
    print(f"  Total VRAM:  {static_info['total_vram_mb']:.0f} MB")
    
    # --- Determine Games to Run ---
    if args.game_name:
        games_to_run = [args.game_name]
    else:
        ale = ALEInterface()
        games_to_run = [g.replace("_", " ").title() for g in ale.getAvailableROMs()]
        print(f"\nNo specific game specified. Benchmarking all {len(games_to_run)} available games.")

    # --- Run Benchmarks ---
    cpu_cores = psutil.cpu_count()
    for game in games_to_run:
        for use_render_flag in [False, True]:
            for env_count in sorted(args.env_counts):
                
                if env_count > (cpu_cores * 2):
                    print(f"\nALE: SKIPPING - Env count {env_count} exceeds 2x CPU cores ({cpu_cores*2}).")
                    print("Stopping further scaling for this game.")
                    break
                
                try:
                    results = run_ale_benchmark(
                        game_name=game,
                        num_steps_per_env=args.steps_per_env,
                        num_envs=env_count,
                        seed=args.seed,
                        use_render=use_render_flag
                    )

                    full_log_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rendering_enabled": use_render_flag,
                        **static_info,
                        **results
                    }
                    
                    append_to_csv(output_path, full_log_entry)

                    render_str = "Render: ON" if use_render_flag else "Render: OFF"
                    if results['total_steps'] > 0:
                        print(f"ALE: SUCCESS - '{game}' with {env_count} envs ({render_str}) | SPS: {results['steps_per_second']:,.0f}")
                    else:
                        print(f"ALE: FAILED - '{game}' with {env_count} envs ({render_str}) | Workers returned 0 steps.")

                except Exception as e:
                    render_str = "ON" if use_render_flag else "OFF"
                    print(f"ALE: FAILED - Unhandled error benchmarking '{game}' with {env_count} envs (Render: {render_str}): {e}")

    print(f"\n--- Benchmark run complete! Results are saved in '{output_path}'. ---")