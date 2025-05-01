import gymnasium as gym
import ale_py
import keyboard  # Global key listener
import time

# Map keys to ALE actions
KEY_TO_ACTION = {
    'left': 4,
    'right': 3,
    'up': 2,
    'down': 5,
    'space': 1,
}

def get_current_action():
    for key, action in KEY_TO_ACTION.items():
        if keyboard.is_pressed(key):
            return action
    return 0

def main():
    env = gym.make("ALE/Tennis-v5", render_mode="human")
    print(env.unwrapped.get_action_meanings())
    obs, info = env.reset()

    done = False
    while not done:
        action = get_current_action()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(1 / 30)  # 30 FPS

    env.close()

if __name__ == "__main__":
    main()