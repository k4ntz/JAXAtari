import argparse
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

# You can modify this list to include the exact environments you want to run.
ATARI_ENVS = [
    "bankheist",
    "beamrider", # still missing pixel runs
    "enduro", 
    "freeway", "frostbite", 
    "kangaroo",
    "montezumarevenge",
    "phoenix", "pong", "qbert",
    "seaquest", "skiing",
    "tennis",
    "venture",
    "timepilot", "asteroids", "breakout", 
    "gravitar",
    "mspacman",
]

# Setting to control how often to rerun an exp (with different seeds)
N_SEEDS = 1
# Setting to control maximum concurrent processes per GPU
WORKERS_PER_GPU = 1

CONFIGS = [
    "ppo_jaxatari_object_large",
    "ppo_jaxatari_pixel_large",
]

# game_specific_configs = {
#     "kangaroo": ("ppo_jaxatari_pixel_large", [3]),
#     "beamrider": ("ppo_jaxatari_pixel", [1,2,3])
# }
game_specific_configs = {}

def worker(gpu_id: str, worker_id: int, task_queue: queue.Queue, extra_args: list):
    """
    Worker function that continuously fetches tasks from the queue
    and executes them on the assigned GPU.
    """
    while not task_queue.empty():
        try:
            env_id, seed, alg_config = task_queue.get_nowait()
        except queue.Empty:
            break
            
        print(f"[GPU {gpu_id} | Worker {worker_id}] Starting {env_id} (seed {seed}, config {alg_config})...")
        
        # Set the target GPU ID for the subprocess
        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        cmd = [
            "uv", "run", "scripts/benchmarks/ppo_jaxatari_scan.py",
            f"+alg={alg_config}",
            f"alg.ENV_ID={env_id}",
            f"SEED={seed}"
        ] + extra_args
        
        try:
            # Run the command
            subprocess.run(cmd, env=env_vars, check=True)
            print(f"[GPU {gpu_id} | Worker {worker_id}] Successfully finished {env_id} (seed {seed}, config {alg_config}).")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id} | Worker {worker_id}] Failed {env_id} (seed {seed}, config {alg_config}) with exit code {e.returncode}.")
        finally:
            task_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="Run PPO JaxAtari scan on multiple GPUs concurrently.")
    parser.add_argument(
        "--gpus", 
        type=str, 
        default="0,1,2,3", 
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')."
    )
    
    # Parse known args, anything else gets passed directly to the ppo_jaxatari_scan script
    args, extra_args = parser.parse_known_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    
    if not gpus:
        print("Error: No GPUs specified.")
        return

    # Create a thread-safe queue and populate it with environments
    task_queue = queue.Queue()
    for env in ATARI_ENVS:
        if env in game_specific_configs:
            alg_config, seeds = game_specific_configs[env]
            for seed in seeds:
                task_queue.put((env, seed, alg_config))
        else:
            for seed in range(1, N_SEEDS + 1):
                for alg_config in CONFIGS:
                    task_queue.put((env, seed, alg_config))
        
    print(f"Starting {task_queue.qsize()} jobs across {len(gpus)} GPU(s): {gpus} ({WORKERS_PER_GPU} workers per GPU)")
    print(f"Extra args for ppo_jaxatari_scan.py: {' '.join(extra_args) if extra_args else 'None'}")
    
    total_workers = len(gpus) * WORKERS_PER_GPU
    # Launch multiple worker threads per GPU
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        for gpu_id in gpus:
            for worker_id in range(WORKERS_PER_GPU):
                executor.submit(worker, gpu_id, worker_id, task_queue, extra_args)
            
    # Wait for all queue tasks to be processed
    task_queue.join()
    print("All experiments finished.")

if __name__ == "__main__":
    main()
