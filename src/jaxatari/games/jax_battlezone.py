import gymnasium as gym
import ale_py
from gymnasium.utils import play
import matplotlib.pyplot as plt

gym.register_envs(ale_py)


def try_gym_battlezone():
    env = gym.make("ALE/BattleZone-v5", render_mode="rgb_array", frameskip=1)
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    play.play(env, zoom=3, fps=60)  # , keys_to_action=keys_to_action)
    env.close()

def try_gym_battlezone_pixel():
    env = gym.make("ALE/BattleZone-v5", render_mode="rgb_array", frameskip=1)
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    stepsize = 100
    for i in range(1000):
        action = 3
        obs, reward, terminated, truncated, info = env.step(action)
        if i%stepsize==0:
            im = plt.imshow(obs, interpolation='none', aspect='auto')
            plt.show()
    env.close()


if __name__ == "__main__":
    try_gym_battlezone()