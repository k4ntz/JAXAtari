import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

def start_game(env):
    env.reset(seed=0)
    for _ in range(60):
        env.step(1)  # FIRE
    # Bestätige, dass wir in PLAY sind
    for _ in range(5):
        env.step(0)

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
start_game(env)

# --- Spieler X über RAM[35] ---
def ram():
    return env.unwrapped.ale.getRAM().copy()

x0 = int(ram()[35])
for _ in range(20):
    env.step(3)  # RIGHT
x1 = int(ram()[35])
print(f"PLAYER RIGHT: {x1-x0} px in 20 steps = {(x1-x0)/20:.3f} px/step")

# --- Bobo bewegt sich mit NOOP ---
start_game(env)
bobo_positions = []
for _ in range(100):
    env.step(0)
    bobo_positions.append(int(ram()[35]))  # temporär — nicht Bobo!

# Bobo-Adresse aus RAM-Änderungen finden
start_game(env)
before = ram()
for _ in range(100):
    env.step(0)
after = ram()
print("\n=== RAM-Änderungen bei 100x NOOP ===")
for i in range(len(before)):
    if before[i] != after[i]:
        print(f"  RAM[{i}]: {int(before[i])} -> {int(after[i])}  delta={int(after[i])-int(before[i])}")

env.close()
