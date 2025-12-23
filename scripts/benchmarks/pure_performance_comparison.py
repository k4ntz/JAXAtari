import argparse
import importlib.util
import inspect
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Type

import gymnasium as gym
import ale_py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import psutil
import gc

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer

def load_game_environment(game_file_path: str) -> Tuple[JaxEnvironment, JAXGameRenderer]:
    """
    Dynamically loads a game environment and the renderer from a .py file.
    It looks for a class that inherits from JaxEnvironment.
    """
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

# It's assumed that jaxatari is installed or accessible in the PYTHONPATH
# If JaxEnvironment or JAXGameRenderer are part of your local project structure,
# ensure sys.path is set up accordingly before these imports if this script is moved.
try:
    from jaxatari.environment import JaxEnvironment
    from jaxatari.renderers import JAXGameRenderer
except ImportError:
    print(
        "Warning: Could not import JaxEnvironment or JAXGameRenderer from jaxatari."
    )
    print("Please ensure 'jaxatari' is installed or in PYTHONPATH.")
    exit()


# --- Resource Monitoring (Identical to original) ---
class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_percentages = []
        self.memory_percentages = []
        self.gpu_utilization = []
        self.gpu_memory = []
        self._stop = False
        self.monitor_thread = None

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
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                util, mem = map(float, result.stdout.strip().split(","))
                return util, mem
            return 0.0, 0.0
        except FileNotFoundError: # nvidia-smi not found
            pass
        except Exception: # Other errors with nvidia-smi
             pass
        try:
            result = subprocess.run(
                [
                    'rocm-smi', '-u', '--showmemuse', '--json'
                ],
                capture_output=True,
                text=True,
                check=False, # Don't raise exception for non-zero exit
                shell=False # Safer
            )
            if result.returncode == 0 and result.stdout.strip():
                # Attempt to parse rocm-smi json (this part may need adjustment based on exact rocm-smi output)
                # The original jq command was: jq -rc \'.card0["GPU use (%)"], .card0["GPU Memory Allocated (VRAM%)"]\'
                # This parsing is a simplified placeholder if jq is not available or direct parsing is preferred.
                # A more robust solution would use the json library.
                # For now, assuming a simplified output format or that the user has jq piped externally if this fails.
                # This part is tricky without knowing the exact JSON structure if jq is not used.
                # Let's assume for now it might fail gracefully.
                # Example (very basic, likely needs refinement):
                # data = json.loads(result.stdout)
                # util = float(data['card0']['GPU use (%)'])
                # mem_percentage = float(data['card0']['GPU Memory Allocated (VRAM%)'])
                # To get MB, one would need total GPU memory. The original script seems to expect MB directly for nvidia-smi
                # and percentage for rocm-smi memory. For consistency, this might need adjustment.
                # For now, returning 0,0 if parsing is complex without jq.
                # The original script with `jq` in the shell command:
                # 'rocm-smi -u --showmemuse --json | jq -rc \'.card0["GPU use (%)"], .card0["GPU Memory Allocated (VRAM%)"]\''
                # This suggests that if `jq` is available, the shell command would work.
                # Re-implementing the shell pipe with jq:
                try:
                    ps1 = subprocess.Popen(['rocm-smi', '-u', '--showmemuse', '--json'], stdout=subprocess.PIPE)
                    ps2 = subprocess.Popen(['jq', '-rc', '.card0["GPU use (%)"], .card0["GPU Memory Allocated (VRAM%)"]'], stdin=ps1.stdout, stdout=subprocess.PIPE, text=True)
                    ps1.stdout.close() # Allow ps1 to receive a SIGPIPE if ps2 exits.
                    output = ps2.communicate()[0]
                    if output:
                        lines = output.strip().split("\n")
                        util = float(lines[0])
                        mem = float(lines[1]) # This was VRAM % in original, not MB.
                                              # The plotting labels it as MB. This is a discrepancy.
                                              # For now, I'll keep it as the value from smi.
                        return util, mem
                except FileNotFoundError: # jq not found
                    pass # Fall through to 0.0, 0.0
                except Exception: # Other errors
                    pass # Fall through to 0.0, 0.0
            return 0.0, 0.0
        except FileNotFoundError: # rocm-smi not found
            return 0.0, 0.0
        except Exception: # Other errors with rocm-smi
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
        self.cpu_percentages = []
        self.memory_percentages = []
        self.gpu_utilization = []
        self.gpu_memory = []
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.daemon = True # Allow main program to exit even if thread is running
        self.monitor_thread.start()

    def stop(self):
        self._stop = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()

    def get_averages(self):
        return {
            "cpu_avg": np.mean(self.cpu_percentages) if self.cpu_percentages else 0.0,
            "memory_avg": np.mean(self.memory_percentages) if self.memory_percentages else 0.0,
            "gpu_util_avg": np.mean(self.gpu_utilization) if self.gpu_utilization else 0.0,
            "gpu_memory_avg": np.mean(self.gpu_memory) if self.gpu_memory else 0.0,
        }

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except:
        pass
    try:
        result = subprocess.run(
            ['rocm-smi', '--showmemuse', '--json'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Try to parse the JSON output
            import json
            data = json.loads(result.stdout)
            # Assuming the memory usage is in MB
            return float(data.get('card0', {}).get('GPU Memory Used', 0))
    except:
        pass
    return 0.0

def measure_single_env_memory_usage(env_creator, num_samples=5):
    """Measure memory usage of a single environment by creating and destroying it multiple times."""
    memory_samples = []
    
    for _ in range(num_samples):
        # Force garbage collection before creating environment
        gc.collect()
        
        # Get initial memory usage
        initial_ram = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        initial_gpu = get_gpu_memory_usage()
        
        # Create environment
        env = env_creator()
        
        # Get memory usage after creation
        after_ram = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        after_gpu = get_gpu_memory_usage()
        
        # Calculate memory usage
        ram_usage = after_ram - initial_ram
        gpu_usage = after_gpu - initial_gpu
        
        memory_samples.append((ram_usage, gpu_usage))
        
        # Clean up
        if hasattr(env, 'close'):
            env.close()
        del env
        gc.collect()
    
    # Calculate average memory usage
    avg_ram = np.mean([s[0] for s in memory_samples])
    avg_gpu = np.mean([s[1] for s in memory_samples])
    
    return avg_ram, avg_gpu

def run_parallel_jax(
    jax_env: Any,  # Instance of the loaded JAX environment
    num_steps_per_env: int = 1_000_000,
    num_envs: int = 2000,
    use_render_step: bool = False, # This corresponds to --render-jax
    seed: int = 0,
) -> Tuple[float, float, int, Dict]:
    # First measure single environment memory usage
    def create_jax_env():
        key = jax.random.PRNGKey(0)
        _, state = jax_env.reset(key)
        return state
    
    single_env_ram, single_env_gpu = measure_single_env_memory_usage(create_jax_env)
    
    monitor = ResourceMonitor()
    monitor.start()

    master_key = jax.random.PRNGKey(seed)
    reset_key, run_key = jax.random.split(master_key)
    
    _obs_single, init_state_single = jax_env.reset(reset_key)
    
    states = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * num_envs), init_state_single
    )

    action_space = jax_env.action_space() 
    num_distinct_actions = action_space.n

    # This function will be vmapped and JITted.
    # It handles one environment's step and optional rendering.
    def _core_logic_for_one_env(current_state_one_env, action_one_env):
        # Always call the step function
        obs, next_state, reward, term, info = jax_env.step(current_state_one_env, jnp.array(0))
        
        if use_render_step: # Python boolean, static for JIT compilation
            # Call render on the next_state.
            # Its output must be part of this function's return to ensure computation by JIT.
            rendered_image = jax_env.render(next_state)
            aux_output = rendered_image 
        else:
            # Provide a structurally similar placeholder if render is not called.
            # Using a dummy array. The actual content doesn't matter if not used later,
            # but its presence ensures consistent function output structure for JIT.
            aux_output = jnp.array([]) # Placeholder for "no rendered image"
            
        return next_state, obs, aux_output

    # Vmap over all environments
    vmapped_core_logic = jax.vmap(_core_logic_for_one_env, in_axes=(0, 0))
    jit_parallel_step = jax.jit(vmapped_core_logic)

    # We need to vmap the action sampling as well to generate actions for all parallel environments.
    vmapped_action_sample = jax.vmap(action_space.sample)

    @jax.jit
    def run_one_step(carry, _):
        current_states_batch, current_rng_key = carry
        action_rng_key, next_rng_key_for_carry = jax.random.split(current_rng_key)
        
        # Generate a batch of actions by creating a batch of random keys and vmapping the sample function.
        action_keys = jax.random.split(action_rng_key, num_envs)
        actions_to_take = vmapped_action_sample(action_keys)
        
        # jit_parallel_step returns (next_states_batch, obs_batch, aux_output_batch)
        # The scan only needs to carry 'next_states_batch' and the new RNG key.
        next_states_batch, _obs_batch, aux_output_batch = jit_parallel_step(current_states_batch, actions_to_take)
        
        if use_render_step:
            # jnp.sum() is a very fast operation on device.
            output_metric = jnp.sum(aux_output_batch) 
        else:
            # If not rendering, the output is just a zero.
            output_metric = 0.0


        return (next_states_batch, next_rng_key_for_carry), output_metric # Scan's per-iteration output is None

    warmup_steps = 1000
    print(f"JAX: Starting warmup ({warmup_steps} steps) and compilation...")
    warmup_start_time = time.time()
    
    # Warmup/Compile jit_parallel_step by calling it once
    # Generate a batch of dummy actions for warmup using the same vmapped sampling.
    warmup_action_keys = jax.random.split(run_key, num_envs)
    dummy_actions = vmapped_action_sample(warmup_action_keys)
    
    # Execute to compile and ensure all parts are processed
    s_w, o_w, aux_w = jit_parallel_step(states, dummy_actions)
    jax.block_until_ready(s_w)
    jax.block_until_ready(o_w)
    jax.block_until_ready(aux_w) # Ensures render (if any) is compiled and run

    # Warmup/Compile scan loop
    (compiled_states, _), _ = jax.lax.scan(
        run_one_step, (states, run_key), None, length=warmup_steps
    )
    jax.block_until_ready(compiled_states) # States after warmup
    warmup_time = time.time() - warmup_start_time
    print(f"JAX: Warmup completed in {warmup_time:.2f} seconds.")

    # Use the states after warmup for the actual benchmark run
    states_for_benchmark_run = compiled_states 
    time.sleep(1)

    start_time = time.time()
    (final_states, _), collected_renders = jax.lax.scan(
        run_one_step, (states_for_benchmark_run, run_key), None, length=num_steps_per_env
    )
    if use_render_step:
        jax.block_until_ready(collected_renders)
    else:
        # If not rendering, just block on the final state as before.
        jax.block_until_ready(final_states)
    total_time = time.time() - start_time

    total_steps = num_steps_per_env * num_envs
    steps_per_second = total_steps / total_time if total_time > 0 else 0

    monitor.stop()
    resource_usage = monitor.get_averages()
    resource_usage["single_env_ram_mb"] = single_env_ram
    resource_usage["single_env_gpu_mb"] = single_env_gpu

    return total_time, steps_per_second, total_steps, resource_usage

# --- Gymnasium Benchmarking ---
def run_gym_worker(
    worker_id: int, steps_per_env: int, ale_game_name: str, num_actions: int, seed_offset: int = 0
) -> int:
    env = gym.make(f"ALE/{ale_game_name}", frameskip=1)
    env.reset(seed=worker_id + seed_offset)
    completed_steps = 0
    for _ in range(steps_per_env):
        action = np.random.randint(0, num_actions)
        env.step(action)
        completed_steps += 1
    env.close()
    return completed_steps


def run_parallel_gym(
    ale_game_name: str,
    num_steps_per_env: int = 1_000_000,
    num_envs: int = None,
    base_seed: int = 42,
) -> Tuple[float, float, int, Dict]:
    if num_envs is None:
        num_envs = mp.cpu_count()

    # First measure single environment memory usage
    def create_gym_env():
        return gym.make(f"ALE/{ale_game_name}", frameskip=1)
    
    single_env_ram, single_env_gpu = measure_single_env_memory_usage(create_gym_env)

    # Fetch num_actions dynamically
    try:
        temp_env = gym.make(f"ALE/{ale_game_name}")
        gym_num_actions = temp_env.action_space.n
        temp_env.close()
    except Exception as e:
        print(f"Error creating Gym env '{ale_game_name}' to get action count: {e}")
        raise

    monitor = ResourceMonitor()
    monitor.start()
    start_time = time.time()

    run_gym_worker_partial = partial(
        run_gym_worker,
        steps_per_env=num_steps_per_env,
        ale_game_name=ale_game_name,
        num_actions=gym_num_actions,
        seed_offset=base_seed # Each worker will use its ID + this offset as seed
    )
    
    print(f"Gymnasium: Launching {num_envs} workers for '{ale_game_name}'.")
    with mp.Pool(processes=num_envs) as pool:
        # Pass worker_id (0 to num_envs-1) to run_gym_worker
        results = pool.map(run_gym_worker_partial, range(num_envs))

    total_time = time.time() - start_time
    total_steps = sum(results)
    steps_per_second = total_steps / total_time if total_time > 0 else 0

    monitor.stop()
    resource_usage = monitor.get_averages()
    resource_usage["single_env_ram_mb"] = single_env_ram
    resource_usage["single_env_gpu_mb"] = single_env_gpu

    return total_time, steps_per_second, total_steps, resource_usage


# --- Scaling Benchmark Orchestration ---
def run_scaling_benchmarks(
    jax_env_instance: Any, # Loaded JAX environment
    ale_game_name_gym: str,
    num_steps_per_env_val: int = 250_000,
    use_render_jax: bool = False,
    output_dir_path: Path = None,
    current_timestamp: str = "",
    jax_game_name_str: str = "jax_game",
    run_gymnasium: bool = True, # New parameter to control Gymnasium benchmark execution
    save_to_file: bool = True
):
    # CPU scaling (Gymnasium)
    # Max workers to test should not exceed available CPUs by too much, 
    # or be configurable. Defaulting to original list for now.
    cpu_workers_to_test = [w for w in [1, 2, 4, 8, 16, 32] if w <= mp.cpu_count() * 2] 
    if not cpu_workers_to_test or max(cpu_workers_to_test) < mp.cpu_count() // 2 : # Ensure some meaningful tests
        cpu_workers_to_test.append(mp.cpu_count())
        cpu_workers_to_test = sorted(list(set(cpu_workers_to_test)))

    gym_results_scaling = []
    actual_cpu_workers_tested = []
    if run_gymnasium: # Only run Gymnasium benchmarks if enabled
        print(f"\nRunning Gymnasium (ALE/{ale_game_name_gym}) scaling tests...")
        for workers in cpu_workers_to_test:
            if workers <= 0: continue
            print(
                f"Gymnasium: Testing with {workers} workers ({num_steps_per_env_val // 1000}K steps per worker, "
                f"{workers * num_steps_per_env_val // 1000}K total steps)..."
            )
            try:
                results = run_parallel_gym(
                    ale_game_name_gym,
                    num_steps_per_env=num_steps_per_env_val,
                    num_envs=workers,
                )
                gym_results_scaling.append(results)
                actual_cpu_workers_tested.append(workers)
                print_benchmark_results(
                    f"Gymnasium Scaling ({workers} workers)",
                    results[0], results[1], results[2], results[3],
                    workers, num_steps_per_env_val,
                    output_dir_path, current_timestamp, jax_game_name_str, is_scaling_log=True, save_to_file=save_to_file
                )
            except Exception as e:
                print(f"Error during Gymnasium scaling test with {workers} workers: {e}")
                break # Stop if one configuration fails

    # GPU scaling (JAX)
    try:
        print(f"JAX is using backend: {jax.default_backend()}")
        if jax.default_backend() == 'cpu':
            print("Warning: JAX is using CPU backend. GPU scaling results may not be representative.")
    except Exception as e:
        print(f"Could not determine JAX backend: {e}")

    gpu_envs_to_test = [1, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    jax_results_scaling = []
    actual_gpu_envs_tested = []
    print(f"\nRunning JAX ({jax_game_name_str}) scaling tests...")
    for num_envs_val in gpu_envs_to_test:
        print(
            f"JAX: Testing with {num_envs_val} parallel environments ({num_steps_per_env_val // 1000}K steps per env, "
            f"{num_envs_val * num_steps_per_env_val // 1000}K total steps)..."
        )
        try:
            results = run_parallel_jax(
                jax_env_instance,
                num_steps_per_env=num_steps_per_env_val,
                num_envs=num_envs_val,
                use_render_step=use_render_jax,
            )
            jax_results_scaling.append(results)
            actual_gpu_envs_tested.append(num_envs_val)
            print_benchmark_results(
                f"JAX Scaling ({num_envs_val} envs)",
                results[0], results[1], results[2], results[3],
                num_envs_val, num_steps_per_env_val,
                output_dir_path, current_timestamp, jax_game_name_str, is_scaling_log=True, save_to_file=save_to_file
            )
        except Exception as e:
            print(f"Error during JAX scaling test with {num_envs_val} environments: {e}")
            if "out of memory" in str(e).lower():
                print("Likely OOM error. Stopping JAX scaling tests.")
                break
            break

    return actual_cpu_workers_tested, gym_results_scaling, actual_gpu_envs_tested, jax_results_scaling


# --- Plotting Functions (Adapted from original) ---
def plot_benchmark_comparison(
    jax_results_std: Tuple, gym_results_std: Tuple, timestamp_str: str, jax_game_name_str: str, base_path: Path
):
    metrics_map = {
        "Time (s)": (0, "Time (seconds)"),
        "Steps per Second": (1, "Steps per Second"),
        "CPU Usage (%)": (lambda res: res[3]["cpu_avg"], "Percentage"),
        "Memory Usage (%)": (lambda res: res[3]["memory_avg"], "Percentage"),
        "GPU Utilization (%)": (lambda res: res[3]["gpu_util_avg"], "Percentage"),
        "GPU Memory (MB)": (lambda res: res[3]["gpu_memory_avg"], "GPU Memory"), # Note: ROCm might be %
    }

    impl_names = ["JAX", "Gymnasium (CPU)"]
    all_results = [jax_results_std, gym_results_std]

    # Create individual plots
    for metric_name, (metric_accessor, y_label) in metrics_map.items():
        plt.figure(figsize=(8, 6))
        data_values = []
        for res_item in all_results:
            if callable(metric_accessor):
                value = metric_accessor(res_item)
            else:
                value = res_item[metric_accessor]
            data_values.append(value)

        plt.bar(impl_names, data_values)
        plt.title(f"{metric_name} Comparison ({jax_game_name_str})")
        plt.ylabel(y_label)
        plot_filename = base_path / f"std_comp_{metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}_{timestamp_str}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved comparison plot: {plot_filename}")

    # Combined plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) # Increased width for labels
    axes = axes.flatten()
    fig.suptitle(f"Standard Benchmark Comparison: {jax_game_name_str}", fontsize=16)

    for i, (metric_name, (metric_accessor, y_label)) in enumerate(metrics_map.items()):
        if i >= len(axes): break # Should not happen with 2x3 and 6 metrics
        data_values = []
        for res_item in all_results:
            if callable(metric_accessor):
                value = metric_accessor(res_item)
            else:
                value = res_item[metric_accessor]
            data_values.append(value)
        
        axes[i].bar(impl_names, data_values)
        axes[i].set_title(metric_name)
        axes[i].set_ylabel(y_label)
        axes[i].tick_params(axis='x', rotation=15) # Rotate x-labels slightly

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    combined_plot_filename = base_path / f"std_comp_combined_{timestamp_str}.png"
    plt.savefig(combined_plot_filename)
    plt.close()
    print(f"Saved combined comparison plot: {combined_plot_filename}")


def plot_scaling_results(
    gym_workers_list: list, gym_results_list: list,
    jax_envs_list: list, jax_results_list: list,
    timestamp_str: str, jax_game_name_str: str, base_path: Path
):
    metrics_map = {"Time": (0, "Time (seconds)"), "Throughput": (1, "Steps per Second")}

    if not gym_results_list and not jax_results_list:
        print("No scaling results to plot.")
        return

    # Create individual scaling plots
    for metric_name, (metric_idx, y_label) in metrics_map.items():
        plt.figure(figsize=(12, 7)) # Increased width for annotations

        if gym_results_list and gym_workers_list:
            gym_values = [res[metric_idx] for res in gym_results_list]
            plt.plot(gym_workers_list, gym_values, "b-o", label="Gymnasium (CPU)")
            for x, y_val in zip(gym_workers_list, gym_values):
                plt.annotate(f"{y_val:.1f}", (x, y_val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

        if jax_results_list and jax_envs_list:
            jax_values = [res[metric_idx] for res in jax_results_list]
            plt.plot(jax_envs_list, jax_values, "r-o", label=f"JAX ({jax_game_name_str}, GPU/TPU)")
            for x, y_val in zip(jax_envs_list, jax_values):
                plt.annotate(f"{y_val:.1f}", (x, y_val), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, color='red')
        
        plt.xscale("log", base=2)
        plt.yscale("log")
        # Use custom ticks for x-axis if lists are not empty
        combined_x_ticks = sorted(list(set((gym_workers_list if gym_workers_list else []) + (jax_envs_list if jax_envs_list else []))))
        if combined_x_ticks:
            plt.xticks(combined_x_ticks, labels=[str(tick) for tick in combined_x_ticks], rotation=45, ha="right")

        plt.xlabel("Number of Workers/Environments (log scale)")
        plt.ylabel(f"{y_label} (log scale)")
        plt.title(f"{metric_name} Scaling: {jax_game_name_str}")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plot_filename = base_path / f"scaling_{metric_name.lower()}_{timestamp_str}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved scaling plot: {plot_filename}")

    # Combined scaling plot
    if (gym_results_list or jax_results_list) and len(metrics_map) > 0 :
        fig, axes = plt.subplots(1, len(metrics_map), figsize=(15, 7))
        if len(metrics_map) == 1: axes = [axes] # Ensure axes is iterable
        fig.suptitle(f"Scaling Comparison: {jax_game_name_str}", fontsize=16)

        for i, (metric_name, (metric_idx, y_label)) in enumerate(metrics_map.items()):
            ax = axes[i]
            if gym_results_list and gym_workers_list:
                gym_values = [res[metric_idx] for res in gym_results_list]
                ax.plot(gym_workers_list, gym_values, "b-o", label="Gymnasium (CPU)")
                for x, y_val in zip(gym_workers_list, gym_values):
                    ax.annotate(f"{y_val:.1f}", (x, y_val), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8)

            if jax_results_list and jax_envs_list:
                jax_values = [res[metric_idx] for res in jax_results_list]
                ax.plot(jax_envs_list, jax_values, "r-o", label=f"JAX ({jax_game_name_str}, GPU/TPU)")
                for x, y_val in zip(jax_envs_list, jax_values):
                    ax.annotate(f"{y_val:.1f}", (x, y_val), textcoords="offset points", xytext=(0,-15), ha="center", fontsize=8, color='red')

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            combined_x_ticks_subplot = sorted(list(set((gym_workers_list if gym_workers_list else []) + (jax_envs_list if jax_envs_list else []))))
            if combined_x_ticks_subplot:
                 ax.set_xticks(combined_x_ticks_subplot, labels=[str(tick) for tick in combined_x_ticks_subplot], rotation=45, ha="right")

            ax.set_xlabel("Number of Workers/Environments (log scale)")
            ax.set_ylabel(f"{y_label} (log scale)")
            ax.set_title(f"{metric_name} Scaling")
            ax.legend()
            ax.grid(True, which="both", ls="-", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
        combined_plot_filename = base_path / f"scaling_combined_{timestamp_str}.png"
        plt.savefig(combined_plot_filename)
        plt.close()
        print(f"Saved combined scaling plot: {combined_plot_filename}")


# --- Utility Functions (Adapted from original) ---
def print_benchmark_results(
    run_name_str: str,
    total_time_val: float,
    steps_per_second_val: float,
    total_steps_val: int,
    resource_usage_dict: Dict,
    num_environments_val: int,
    steps_per_env_val: int,
    output_path: Path,
    timestamp_str: str,
    jax_game_name_for_log: str,
    is_scaling_log: bool = False, # To slightly alter log file name for scaling runs
    save_to_file: bool = True
):
    # Always print a concise console summary (includes Steps per Second)
    print(f"[{run_name_str}] time={total_time_val:.2f}s, sps={steps_per_second_val:,.2f}, steps={total_steps_val:,}, envs={num_environments_val:,}, spe={steps_per_env_val:,}")

    if not save_to_file or output_path is None:
        return

    log_file_suffix = "scaling_log" if is_scaling_log else "std_results"
    log_filename = output_path / f"benchmark_{log_file_suffix}_{timestamp_str}_{jax_game_name_for_log}.txt"
    
    with open(log_filename, "a") as f:
        print(f"\n--- {run_name_str} Benchmark Results ---", file=f)
        print(f"Game: {jax_game_name_for_log}", file=f)
        print(f"Timestamp: {timestamp_str}", file=f)
        print(f"Total steps completed: {total_steps_val:,}", file=f)
        print(f"Steps per environment: {steps_per_env_val:,}", file=f)
        print(f"Number of environments/workers: {num_environments_val:,}", file=f)
        print(f"Total time: {total_time_val:.2f} seconds", file=f)
        print(f"Average steps per second: {steps_per_second_val:,.2f}", file=f)
        if total_steps_val > 0:
            print(f"Microseconds per step: {(total_time_val * 1_000_000 / total_steps_val):.2f}\u03BCs", file=f)
        
        print("\nResource Usage (Averages):", file=f)
        print(f"  CPU Usage: {resource_usage_dict['cpu_avg']:.1f}%", file=f)
        print(f"  Memory Usage: {resource_usage_dict['memory_avg']:.1f}%", file=f)
        print(f"  GPU Utilization: {resource_usage_dict['gpu_util_avg']:.1f}%", file=f)
        print(f"  GPU Memory Usage: {resource_usage_dict['gpu_memory_avg']:.1f} (Unit varies by SMI tool, see GPU Mem label in plots)", file=f)
        print(f"  Single Environment RAM Usage: {resource_usage_dict['single_env_ram_mb']:.1f} MB", file=f)
        print(f"  Single Environment GPU Usage: {resource_usage_dict['single_env_gpu_mb']:.1f} MB", file=f)
        print("--- End of Results ---", file=f)
    print(f"Results for '{run_name_str}' logged to: {log_filename}")


def save_raw_results(data_tuple: Tuple, data_names: list, raw_path: Path, prefix: str):
    raw_path.mkdir(parents=True, exist_ok=True)
    for data_item, name_str in zip(data_tuple, data_names):
        if name_str == "resource_usage" and isinstance(data_item, dict):
            # Save each key in the resource_usage dict separately
            for res_key, res_val in data_item.items():
                np.save(raw_path / f"{prefix}_{name_str}_{res_key}.npy", res_val)
        else:
            np.save(raw_path / f"{prefix}_{name_str}.npy", data_item)
    print(f"Raw data saved to: {raw_path} with prefix '{prefix}_'")


def save_raw_scaling_results(
    gym_results: list, jax_results: list,
    gym_workers: list, jax_envs: list,
    raw_path: Path
):
    raw_path.mkdir(parents=True, exist_ok=True)
    if gym_results:
        np.save(raw_path / "gym_scaling_times.npy", [res[0] for res in gym_results])
        np.save(raw_path / "gym_scaling_sps.npy", [res[1] for res in gym_results])
        np.save(raw_path / "gym_scaling_total_steps.npy", [res[2] for res in gym_results])
        np.save(raw_path / "gym_scaling_workers.npy", gym_workers)
        # Could also save resource usage per scaling point if needed

    if jax_results:
        np.save(raw_path / "jax_scaling_times.npy", [res[0] for res in jax_results])
        np.save(raw_path / "jax_scaling_sps.npy", [res[1] for res in jax_results])
        np.save(raw_path / "jax_scaling_total_steps.npy", [res[2] for res in jax_results])
        np.save(raw_path / "jax_scaling_envs.npy", jax_envs)
    print(f"Raw scaling data saved to: {raw_path}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark JAX vs Gymnasium Atari environments.")
    parser.add_argument("--jax-game-path", type=str, required=True, help="Path to the Python file for the JAX game environment.")
    parser.add_argument("--ale-game-name", type=str, required=False, help="Name of the ALE ROM for Gymnasium (e.g., Pong-v5, Breakout-v5). If not provided, only JAX benchmarks will run.")
    
    parser.add_argument("--steps-per-env", type=int, default=50_000, help="Steps per environment for benchmarks.")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Directory to save results.")
    parser.add_argument("--save-output", action=argparse.BooleanOptionalAction, default=False, help="Save logs, raw data, and plots to disk.")

    parser.add_argument("--run-std-benchmark", action=argparse.BooleanOptionalAction, default=True, help="Run the standard (single point) benchmark.")
    parser.add_argument("--run-scaling-benchmark", action=argparse.BooleanOptionalAction, default=False, help="Run the scaling benchmark.")
    
    parser.add_argument("--render-jax", action=argparse.BooleanOptionalAction, default=False, help="Use JAX environment's 'step_with_render' method (if available).")
    
    parser.add_argument("--num-envs-jax-std", type=int, default=128, help="Number of environments for JAX standard benchmark.")
    parser.add_argument("--num-envs-gym-std", type=int, default=mp.cpu_count(), help="Number of workers for Gymnasium standard benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")

    args = parser.parse_args()

    # Check if ALE game is provided
    run_gymnasium_benchmarks = args.ale_game_name is not None
    if not run_gymnasium_benchmarks:
        print("Note: No ALE game specified. Running JAX-only benchmarks.")

    try:
        print(f"Loading JAX game from: {args.jax_game_path}")
        jax_env_main, _ = load_game_environment(args.jax_game_path) # Renderer not used by benchmark logic directly
        jax_game_name_main = Path(args.jax_game_path).stem # e.g., JaxPong from /path/to/JaxPong.py
    except Exception as e:
        print(f"Fatal: Could not load JAX game environment: {e}")
        sys.exit(1)

    # Create base output directory for this game (only if saving is enabled)
    if args.save_output:
        main_output_path = Path(args.output_dir) / jax_game_name_main
        main_output_path.mkdir(parents=True, exist_ok=True)
        raw_output_path = main_output_path / "raw_data"
        raw_output_path.mkdir(parents=True, exist_ok=True)
    else:
        main_output_path = None
        raw_output_path = None

    timestamp_main = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save system information (only if saving is enabled)
    if args.save_output:
        sys_info_log = (Path(args.output_dir) / jax_game_name_main) / f"system_info_{timestamp_main}_{jax_game_name_main}.txt"
        with open(sys_info_log, "w") as f:
            print("--- System Information ---", file=f)
            print(f"Timestamp: {timestamp_main}", file=f)
            print(f"JAX Game: {jax_game_name_main} (from {args.jax_game_path})", file=f)
            print(f"Gymnasium ALE Game: {args.ale_game_name if run_gymnasium_benchmarks else 'Not specified (JAX-only mode)'}", file=f)
            print(f"Steps per Environment (nominal): {args.steps_per_env:,}", file=f)
            print(f"JAX Standard Envs: {args.num_envs_jax_std}", file=f)
            if run_gymnasium_benchmarks:
                print(f"Gym Standard Envs: {args.num_envs_gym_std}", file=f)
            print(f"JAX Render Step Used: {args.render_jax}", file=f)
            print(f"CPU cores available (multiprocessing): {mp.cpu_count()}", file=f)
            try:
                print(f"JAX Devices: {jax.devices()}", file=f)
                print(f"JAX Default Backend: {jax.default_backend()}", file=f)
            except Exception as e:
                print(f"Could not get JAX device info: {e}", file=f)
        print(f"System info logged to: {sys_info_log}")
    else:
        print("--- System Information ---")
        print(f"Timestamp: {timestamp_main}")
        print(f"JAX Game: {jax_game_name_main} (from {args.jax_game_path})")
        print(f"Gymnasium ALE Game: {'Not specified (JAX-only mode)' if not run_gymnasium_benchmarks else args.ale_game_name}")
        print(f"Steps per Environment (nominal): {args.steps_per_env:,}")
        print(f"JAX Standard Envs: {args.num_envs_jax_std}")
        if run_gymnasium_benchmarks:
            print(f"Gym Standard Envs: {args.num_envs_gym_std}")
        print(f"JAX Render Step Used: {args.render_jax}")
        try:
            print(f"JAX Devices: {jax.devices()}")
            print(f"JAX Default Backend: {jax.default_backend()}")
        except Exception as e:
            print(f"Could not get JAX device info: {e}")

    # --- Run Scaling Benchmarks ---
    if args.run_scaling_benchmark:
        if run_gymnasium_benchmarks:
            print(f"\n--- Starting Scaling Benchmarks for {jax_game_name_main} vs {args.ale_game_name} ---")
        else:
            print(f"\n--- Starting JAX Scaling Benchmarks for {jax_game_name_main} ---")
        
        gym_workers_tested, gym_scaling_res, jax_envs_tested, jax_scaling_res = run_scaling_benchmarks(
            jax_env_instance=jax_env_main,
            ale_game_name_gym=args.ale_game_name,
            num_steps_per_env_val=args.steps_per_env,
            use_render_jax=args.render_jax,
            output_dir_path=(main_output_path if args.save_output else None),
            current_timestamp=timestamp_main,
            jax_game_name_str=jax_game_name_main,
            run_gymnasium=run_gymnasium_benchmarks,
            save_to_file=args.save_output
        )
        if (gym_scaling_res or jax_scaling_res) and args.save_output:
            save_raw_scaling_results(
                gym_scaling_res, jax_scaling_res,
                gym_workers_tested, jax_envs_tested,
                raw_output_path
            )
            print("\nGenerating scaling plots...")
            plot_scaling_results(
                gym_workers_tested, gym_scaling_res,
                jax_envs_tested, jax_scaling_res,
                timestamp_main, jax_game_name_main, main_output_path
            )
        elif not (gym_scaling_res or jax_scaling_res):
            print("No data from scaling benchmarks to plot.")
    else:
        print("\nSkipping scaling benchmarks as per arguments.")

    # --- Run Standard Benchmarks for Detailed Comparison ---
    if args.run_std_benchmark:
        if run_gymnasium_benchmarks:
            print(f"\n--- Starting Standard Benchmark for {jax_game_name_main} vs {args.ale_game_name} ---")
        else:
            print(f"\n--- Starting JAX Standard Benchmark for {jax_game_name_main} ---")
        print(f"JAX ({jax_game_name_main}): Running with {args.num_envs_jax_std} environments, {args.steps_per_env} steps each.")
        
        std_jax_results = None
        std_gym_results = None

        try:
            std_jax_results = run_parallel_jax(
                jax_env=jax_env_main,
                num_steps_per_env=args.steps_per_env,
                num_envs=args.num_envs_jax_std,
                use_render_step=args.render_jax,
                seed=args.seed
            )
            print_benchmark_results(
                f"JAX Standard ({jax_game_name_main})", *std_jax_results,
                args.num_envs_jax_std, args.steps_per_env,
                (main_output_path if args.save_output else None), timestamp_main, jax_game_name_main,
                save_to_file=args.save_output
            )
            if args.save_output:
                save_raw_results(std_jax_results, ["time", "sps", "total_steps", "resource_usage"], raw_output_path, "jax_std")
        except Exception as e:
            print(f"Error during JAX standard benchmark: {e}")

        if run_gymnasium_benchmarks:
            print(f"\nGymnasium ({args.ale_game_name}): Running with {args.num_envs_gym_std} workers, {args.steps_per_env} steps each.")
            try:
                std_gym_results = run_parallel_gym(
                    ale_game_name=args.ale_game_name,
                    num_steps_per_env=args.steps_per_env,
                    num_envs=args.num_envs_gym_std,
                    base_seed=args.seed
                )
                print_benchmark_results(
                    f"Gymnasium Standard ({args.ale_game_name})", *std_gym_results,
                    args.num_envs_gym_std, args.steps_per_env,
                    (main_output_path if args.save_output else None), timestamp_main, jax_game_name_main, # Log gym under jax game name for grouping
                    save_to_file=args.save_output
                )
                if args.save_output:
                    save_raw_results(std_gym_results, ["time", "sps", "total_steps", "resource_usage"], raw_output_path, "gym_std")
            except Exception as e:
                print(f"Error during Gymnasium standard benchmark: {e}")

            if std_jax_results and std_gym_results and args.save_output:
                print("\nGenerating standard comparison plots...")
                plot_benchmark_comparison(
                    std_jax_results, std_gym_results, timestamp_main, jax_game_name_main, main_output_path
                )
            elif not (std_jax_results and std_gym_results):
                print("Could not generate standard comparison plots due to missing results from one or both frameworks.")
        else:
            print("\nSkipping Gymnasium benchmarks (JAX-only mode).")
    else:
        print("\nSkipping standard benchmark comparison as per arguments.")

    if run_gymnasium_benchmarks:
        if args.save_output:
            print(f"\n--- Benchmark run for {jax_game_name_main} vs {args.ale_game_name} complete! Check the '{main_output_path}' directory. ---")
        else:
            print(f"\n--- Benchmark run for {jax_game_name_main} vs {args.ale_game_name} complete! No files were saved (--no-save-output). ---")
    else:
        if args.save_output:
            print(f"\n--- JAX-only benchmark run for {jax_game_name_main} complete! Check the '{main_output_path}' directory. ---")
        else:
            print(f"\n--- JAX-only benchmark run for {jax_game_name_main} complete! No files were saved (--no-save-output). ---")
