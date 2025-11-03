import gymnasium as gym
import ale_py
from gymnasium.utils import play

gym.register_envs(ale_py)

if __name__ == "__main__":
    #print(sorted(gym.envs.registry.keys()))
    env = gym.make("ALE/LostLuggage", render_mode="rgb_array", frameskip=1)

    # Reset the environment to generate the first observation
    #trained_agent_test(env)
    observation, info = env.reset(seed=42)
    play.play(env,zoom=3,fps=60)#, keys_to_action=keys_to_action)
    env.close()