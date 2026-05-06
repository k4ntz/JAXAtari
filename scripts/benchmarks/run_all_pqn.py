import argparse
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

# You can modify this list to include the exact environments you want to run.
ATARI_ENVS = [
    "freeway", 
    "kangaroo",
    "montezumarevenge",
    "mspacman",
    "phoenix", "pong", "qbert",
    "seaquest", "skiing",
    "tennis",
    "venture",
    "timepilot", "asteroids", "breakout", 
    "frostbite", "gravitar",
    "bankheist",
    "beamrider",
    "enduro", 
]

# Setting to control how often to rerun an exp (with different seeds)
N_SEEDS = 1 
# Setting to control maximum concurrent processes per GPU
WORKERS_PER_GPU = 1 # we already run three seeds per GPU/Env

CONFIGS = [
    "pqn_short_eval_run_object",
    "pqn_short_eval_run_pixel",
]

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
            "uv", "run", "scripts/benchmarks/pqn_agent.py",
            f"+alg={alg_config}",
            f"alg.ENV_NAME={env_id}",
            f"SEED={seed}",
            f"NUM_SEEDS=1" #use only 1 seed for the large runs
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
    parser = argparse.ArgumentParser(description="Run PQN JaxAtari scan on multiple GPUs concurrently.")
    parser.add_argument(
        "--gpus", 
        type=str, 
        default="0,1,2,3", 
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')."
    )
    
    # Parse known args, anything else gets passed directly to the pqn_agent script
    args, extra_args = parser.parse_known_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    
    if not gpus:
        print("Error: No GPUs specified.")
        return

    # Create a thread-safe queue and populate it with environments
    task_queue = queue.Queue()
    for env in ATARI_ENVS:
        for seed in range(1, N_SEEDS + 1):
            for alg_config in CONFIGS:
                task_queue.put((env, seed, alg_config))
        
    print(f"Starting {len(ATARI_ENVS) * N_SEEDS * len(CONFIGS)} jobs across {len(gpus)} GPU(s): {gpus} ({WORKERS_PER_GPU} workers per GPU)")
    print(f"Extra args for pqn_agent.py: {' '.join(extra_args) if extra_args else 'None'}")
    
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
