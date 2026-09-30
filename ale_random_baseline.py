"""
Random-policy baseline on the real ALE StarGunner.

Mirrors the settings of the JAX environment:
  - frameskip 4
  - sticky actions 0.25
  - 18 actions (full action set)
  - 108000 ALE frames per episode (== 27000 agent steps * frameskip)

Install:
    pip install ale-py gymnasium numpy

Run:
    python ale_random_baseline.py --episodes 200 --jax-ep-ret 798.4 --jax-reward-per-step 0.9088
"""
import argparse
import time
from collections import Counter

import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--sticky", type=float, default=0.25)
    p.add_argument("--max-steps", type=int, default=27_000,
                   help="agent steps per episode (27000 * frameskip = 108000 frames)")
    # JAX-side reference numbers — pass them so the script doesn't hardcode stale values
    p.add_argument("--jax-ep-ret", type=float, default=798.4,
                   help="random-policy episodic return measured on the JAX env")
    p.add_argument("--jax-reward-per-step", type=float, default=0.9088,
                   help="random-policy reward-per-step measured on the JAX env")
    p.add_argument("--smoke", action="store_true",
                   help="Run only 3 episodes for a quick sanity check.")
    args = p.parse_args()

    if args.smoke:
        args.episodes = 3

    env = gym.make(
        "ALE/StarGunner-v5",
        frameskip=args.frameskip,
        repeat_action_probability=args.sticky,
        full_action_space=True,                                 # 18 actions
        max_num_frames_per_episode=args.max_steps * args.frameskip,
    )
    print(f"actions={env.action_space.n}  frameskip={args.frameskip}  "
          f"sticky={args.sticky}  episodes={args.episodes}")

    rng = np.random.default_rng(args.seed)
    env.action_space.seed(args.seed)

    ep_returns, ep_lengths, nonzero_frac = [], [], []
    reward_values = []
    t0 = time.perf_counter()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total, steps, nz = 0.0, 0, 0
        done = False
        while not done:
            a = int(rng.integers(env.action_space.n))
            obs, r, terminated, truncated, info = env.step(a)
            total += r
            steps += 1
            if r != 0:
                nz += 1
                reward_values.append(r)
            done = terminated or truncated
        ep_returns.append(total)
        ep_lengths.append(steps)
        nonzero_frac.append(nz / steps)
        if (ep + 1) % 20 == 0:
            print(f"  {ep + 1}/{args.episodes}  mean return so far "
                  f"{np.mean(ep_returns):.1f}")
    env.close()
    dt = time.perf_counter() - t0

    R = np.array(ep_returns)
    L = np.array(ep_lengths)
    per_step = R.sum() / L.sum()

    print(f"\nALE random policy ({args.episodes} episodes, {dt:.0f}s):")
    print(f"  episodic return : mean {R.mean():.1f}  std {R.std():.1f}  "
          f"median {np.median(R):.1f}  min {R.min():.0f}  max {R.max():.0f}")
    print(f"  episode length  : mean {L.mean():.0f} agent steps "
          f"({L.mean() * args.frameskip:.0f} frames)")
    print(f"  reward per step : {per_step:.4f}")
    print(f"  steps with reward != 0: {100 * np.mean(nonzero_frac):.1f}%")
    print(f"\n  unique reward values: {Counter(reward_values).most_common(10)}")

    print("\nComparison with the JAX environment (random policy):")
    print(f"  episodic return : ALE {R.mean():.1f}  vs  JAX {args.jax_ep_ret:.1f}  "
          f"(ratio JAX/ALE = {args.jax_ep_ret / R.mean():.2f})")
    print(f"  reward per step : ALE {per_step:.4f}  vs  JAX {args.jax_reward_per_step:.4f}  "
          f"(ratio JAX/ALE = {args.jax_reward_per_step / per_step:.2f})")

    np.savez("ale_random_baseline.npz",
             returns=R, lengths=L, rewards=np.array(reward_values))
    print("saved ale_random_baseline.npz")


if __name__ == "__main__":
    main()