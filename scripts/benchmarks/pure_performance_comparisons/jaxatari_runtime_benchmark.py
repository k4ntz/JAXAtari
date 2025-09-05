import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# optional:
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import argparse
import csv
import subprocess
import sys
import threading
import time
import importlib.util
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from jaxatari.renderers import JAXGameRenderer
from jaxatari.environment import JaxEnvironment

import numpy as np


def load_game_environment(game: str) -> Tuple[JaxEnvironment, JAXGameRenderer]:
    """
    Dynamically loads a game environment and the renderer from a .py file.
    It looks for a class that inherits from JaxEnvironment.
    """
    # Get the project root directory (parent of parent of parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    game_file_path = os.path.join(project_root, "src", "jaxatari", "games", f"jax_{game.lower()}.py")
    
    if not os.path.exists(game_file_path):
        raise FileNotFoundError(f"Game file not found: {game_file_path}")

    module_name = os.path.splitext(os.path.basename(game_file_path))[0]

    # Add the directory of the game file to sys.path to handle relative imports within the game file
    game_dir = os.path.dirname(os.path.abspath(game_file_path))
    if game_dir not in sys.path:
        sys.path.insert(0, game_dir)

    spec = importlib.util.spec_from_file_location(module_name, game_file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {game_file_path}")

    game_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(game_module)
    except Exception as e:
        if game_dir in sys.path and sys.path[0] == game_dir:  # Clean up sys.path if we added to it
            sys.path.pop(0)
        raise ImportError(f"Could not execute module {module_name}: {e}")

    if game_dir in sys.path and sys.path[0] == game_dir:  # Clean up sys.path if we added to it
        sys.path.pop(0)

    game = None
    renderer = None
    # Find the class that inherits from JaxEnvironment
    for name, obj in inspect.getmembers(game_module):
        if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
            print(f"Found game environment: {name}")
            game = obj()  # Instantiate and return

        if inspect.isclass(obj) and issubclass(obj, JAXGameRenderer) and obj is not JAXGameRenderer:
            print(f"Found renderer: {name}")
            renderer = obj()

    if game is None:
        raise ImportError(f"No class found in {game_file_path} that inherits from JaxEnvironment")

    return game, renderer


# Set JAX_PLATFORM_NAME before importing jax
# This can be overridden by the --force-cpu flag logic later
if 'JAX_PLATFORM_NAME' not in os.environ:
    # Default to GPU if available, otherwise JAX will auto-select
    pass

import jax
import jax.numpy as jnp
import psutil
import gc

try:
    from jaxatari.core import make, list_available_games
    from jaxatari.environment import JaxEnvironment
except ImportError as e:
    print(f"Fatal: Could not import from jaxatari: {e}")
    print("Please ensure 'jaxatari' is installed and accessible in your PYTHONPATH.")
    sys.exit(1)


# --- System Information Utilities ---

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
        pass # nvidia-smi not found or failed, try AMD
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
            card_key = next(iter(data)) # Get first card e.g. 'card0'
            details['gpu_name'] = data[card_key].get('Card series', 'AMD ROCm Device')
            total_vram_str = data[card_key].get('VRAM Total Memory (B)', '0')
            details['total_vram_mb'] = int(total_vram_str) / (1024**2)
            return details
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass # rocm-smi not found or failed
        except Exception as e:
            print(f"Warning: Error querying AMD GPU details: {e}")

    return details


# --- Resource Monitoring ---

class ResourceMonitor:
    """
    A thread-based monitor for CPU, RAM, and GPU resources.
    """
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
            pass # Try AMD next
        
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
            return 0.0, 0.0 # No supported GPU tool found
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
        """
        Stops the monitoring thread and returns the averaged results.
        """
        self._stop = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
        
        return {
            "cpu_percent_avg": np.mean(self.cpu_percentages) if self.cpu_percentages else 0.0,
            "ram_percent_avg": np.mean(self.memory_percentages) if self.memory_percentages else 0.0,
            "gpu_percent_avg": np.mean(self.gpu_utilization) if self.gpu_utilization else 0.0,
            "gpu_vram_mb_avg": np.mean(self.gpu_memory_mb) if self.gpu_memory_mb else 0.0,
        }


# --- JAX Benchmarking Logic ---

def run_jax_benchmark(
    game_name: str,
    num_steps_per_env: int,
    num_envs: int,
    seed: int,
    use_render: bool,
) -> Dict[str, Any]:
    """
    Runs a parallelized JAXAtari benchmark for a single game and environment count.

    Returns:
        A dictionary containing all key benchmark metrics.
    """
    render_status = "ENABLED" if use_render else "DISABLED"
    print(f"\nJAX: Benchmarking '{game_name}' with {num_envs} environments (Rendering {render_status})...")


    monitor = ResourceMonitor()
    try:
        env = make(game_name)
    except Exception as e:
        env, _ = load_game_environment(game_name)
        if not env:
            print(f"JAX: FAILED - Error loading environment (both core and dynamically) for '{game_name}': {e}")
            return {"error": str(e)}

    # --- JIT Compilation Phase ---
    print(f"JAX: Compiling JIT functions...")
    compilation_start_time = time.time()

    master_key = jax.random.PRNGKey(seed)
    reset_key, run_key = jax.random.split(master_key)

    # Vmap all necessary environment functions
    vmapped_reset = jax.vmap(env.reset)
    vmapped_step = jax.vmap(env.step)
    vmapped_render = jax.vmap(env.render)
    vmapped_action_sample = jax.vmap(env.action_space().sample)

    # Prepare initial states
    batch_reset_keys = jax.random.split(reset_key, num_envs)
    _obs_batch, states = vmapped_reset(batch_reset_keys)

    # --- Define JIT-compiled functions ---
    @jax.jit
    def get_actions(keys):
        return vmapped_action_sample(keys)

    # We define two separate loop bodies. JAX will compile the one we use.
    # This is cleaner than having a conditional inside the loop.
    @jax.jit
    def run_one_step_no_render(carry, _):
        current_states, current_rng_key = carry
        step_rng_key, reset_rng_key, next_rng_key = jax.random.split(current_rng_key, 3)
        action_keys = jax.random.split(step_rng_key, num_envs)
        actions = get_actions(action_keys)
        _obs, next_states, rewards, terms, _infos = vmapped_step(current_states, actions)
        reset_keys = jax.random.split(reset_rng_key, num_envs)
        _obs_reset, reset_states = vmapped_reset(reset_keys)
        final_next_states = jax.tree.map(
            lambda r, n: jnp.where(terms.reshape(terms.shape + (1,) * (n.ndim - 1)), r, n),
            reset_states, next_states
        )
        return (final_next_states, next_rng_key), (rewards, terms)

    @jax.jit
    def run_one_step_render(carry, _):
        current_states, current_rng_key = carry
        step_rng_key, reset_rng_key, next_rng_key = jax.random.split(current_rng_key, 3)
        action_keys = jax.random.split(step_rng_key, num_envs)
        actions = get_actions(action_keys)
        _obs, next_states, rewards, terms, _infos = vmapped_step(current_states, actions)
        reset_keys = jax.random.split(reset_rng_key, num_envs)
        _obs_reset, reset_states = vmapped_reset(reset_keys)
        final_next_states = jax.tree.map(
            lambda r, n: jnp.where(terms.reshape(terms.shape + (1,) * (n.ndim - 1)), r, n),
            reset_states, next_states
        )
        
        rendered_images = vmapped_render(final_next_states)

        image_checksum = jnp.sum(rendered_images.astype(jnp.uint32))

        return (final_next_states, next_rng_key), (rewards, terms, image_checksum)


    # --- Execute and Time ---
    if use_render:
        # Compile by running for one step
        (c_states, _), (_, _, c_images) = jax.lax.scan(run_one_step_render, (states, run_key), None, length=1)
        jax.block_until_ready((c_states, c_images))
        compilation_time = time.time() - compilation_start_time
        print(f"JAX: Compilation finished in {compilation_time:.2f} seconds.")
        time.sleep(1)

        # Run benchmark
        monitor.start()
        execution_start_time = time.time()
        (f_states, _), (_, _, f_images) = jax.lax.scan(run_one_step_render, (states, run_key), None, length=num_steps_per_env)
        jax.block_until_ready((f_states, f_images))
        execution_time = time.time() - execution_start_time
    else:
        # Compile by running for one step
        (c_states, _), _ = jax.lax.scan(run_one_step_no_render, (states, run_key), None, length=1)
        jax.block_until_ready(c_states)
        compilation_time = time.time() - compilation_start_time
        print(f"JAX: Compilation finished in {compilation_time:.2f} seconds.")
        time.sleep(1)

        # Run benchmark
        monitor.start()
        execution_start_time = time.time()
        (f_states, _), _ = jax.lax.scan(run_one_step_no_render, (states, run_key), None, length=num_steps_per_env)
        jax.block_until_ready(f_states)
        execution_time = time.time() - execution_start_time

    resource_usage = monitor.stop()
    gc.collect()

    total_steps = num_steps_per_env * num_envs
    steps_per_second = total_steps / execution_time if execution_time > 0 else 0

    return {
        "game_name": game_name,
        "spawned_instances": num_envs,
        "total_steps": total_steps,
        "total_runtime_secs": execution_time,
        "compilation_time_secs": compilation_time,
        "steps_per_second": steps_per_second,
        **resource_usage,
    }
# --- CSV Logging ---

CSV_HEADER = [
    "timestamp", "game_name", "rendering_enabled", "jax_backend", "cpu_name", "gpu_name",
    "total_vram_mb", "spawned_instances", "total_steps",
    "total_runtime_secs", "compilation_time_secs", "steps_per_second",
    "cpu_percent_avg", "ram_percent_avg", "gpu_percent_avg", "gpu_vram_mb_avg"
]

def setup_csv_logging(filepath: Path) -> None:
    """Creates the CSV file and writes the header if it doesn't exist."""
    if not filepath.exists():
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        print(f"Created log file: {filepath}")

def append_to_csv(filepath: Path, result_dict: Dict[str, Any]) -> None:
    """Appends a new row of benchmark results to the CSV file."""
    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writerow(result_dict)
    except IOError as e:
        print(f"Error: Could not write to CSV file {filepath}: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A modular JAXAtari performance benchmark script.")
    parser.add_argument(
        "--game", type=str, default=None,
        help="Name of the JAXAtari game to benchmark (e.g., 'pong'). If not specified, all available games will be benchmarked."
    )
    parser.add_argument(
        "--steps-per-env", type=int, default=10_000,
        help="Number of simulation steps to run for each parallel environment."
    )
    parser.add_argument(
        "--output-file", type=str, default="./jaxatari_benchmark_results.csv",
        help="Path to the output CSV file for storing results."
    )
    parser.add_argument(
        "--env-counts", type=int, nargs='+', default=[64, 256, 1024, 4096, 16384],
        help="A space-separated list of parallel environment counts to test for scaling."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility."
    )
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="Force JAX to use the CPU backend, even if a GPU/TPU is available."
    )
    args = parser.parse_args()

    # --- Setup ---
    # Configure JAX backend BEFORE any JAX operations
    if args.force_cpu:
        jax.config.update('jax_platform_name', 'cpu')
        print("--- Forcing JAX to use CPU backend ---")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_csv_logging(output_path)
    
    # --- Collect Static System Info ---
    print("Collecting system information...")
    static_info = {
        "jax_backend": jax.default_backend(),
        "cpu_name": get_cpu_name(),
        **get_gpu_details()
    }
    print(f"  JAX Backend: {static_info['jax_backend']}")
    print(f"  CPU Name:    {static_info['cpu_name']}")
    print(f"  GPU Name:    {static_info['gpu_name']}")
    print(f"  Total VRAM:  {static_info['total_vram_mb']:.0f} MB")
    
    # --- Determine Games to Run ---
    if args.game:
        games_to_run = [args.game]
    else:
        games_to_run = list_available_games()
        print(f"\nNo specific game specified. Benchmarking all available games: {games_to_run}")

    # --- Run Benchmarks ---
    cpu_cores = psutil.cpu_count()
    for game in games_to_run:
        for use_render_flag in [False, True]: # Loop for rendering enabled/disabled
            for env_count in sorted(args.env_counts):
                
                if args.force_cpu and env_count > (cpu_cores * 2):
                    print(f"\nJAX: SKIPPING - Env count {env_count} exceeds 2x CPU cores ({cpu_cores*2}).")
                    print("Stopping further scaling for this game on CPU.")
                    break

                try:
                    results = run_jax_benchmark(
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
                    print(f"JAX: SUCCESS - '{game}' with {env_count} envs ({render_str}) | SPS: {results['steps_per_second']:,.0f}")
                    
                except Exception as e:
                    render_str = "ON" if use_render_flag else "OFF"
                    print(f"JAX: FAILED - Error benchmarking '{game}' with {env_count} envs (Render: {render_str}): {e}")
                    if "out of memory" in str(e).lower():
                        print("This was likely an Out-of-Memory error. Stopping benchmarks for this game.")
                        break
    
    print(f"\n--- Benchmark run complete! Results are saved in '{output_path}'. ---")