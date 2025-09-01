import gymnasium as gym
import ale_py

import time

import keyboard

# Action constants (matching the previous context)
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

def get_current_action():
    # Check all relevant keys
    up = keyboard.is_pressed('up') or keyboard.is_pressed('w')
    down = keyboard.is_pressed('down') or keyboard.is_pressed('s')
    left = keyboard.is_pressed('left') or keyboard.is_pressed('a')
    right = keyboard.is_pressed('right') or keyboard.is_pressed('d')
    fire = keyboard.is_pressed('space')

    # Diagonal movements with fire
    if up and right and fire:
        return UPRIGHTFIRE
    if up and left and fire:
        return UPLEFTFIRE
    if down and right and fire:
        return DOWNRIGHTFIRE
    if down and left and fire:
        return DOWNLEFTFIRE

    # Cardinal directions with fire
    if up and fire:
        return UPFIRE
    if down and fire:
        return DOWNFIRE
    if left and fire:
        return LEFTFIRE
    if right and fire:
        return RIGHTFIRE

    # Diagonal movements
    if up and right:
        return UPRIGHT
    if up and left:
        return UPLEFT
    if down and right:
        return DOWNRIGHT
    if down and left:
        return DOWNLEFT

    # Single directions
    if up:
        return UP
    if down:
        return DOWN
    if left:
        return LEFT
    if right:
        return RIGHT
    if fire:
        return FIRE

    return NOOP

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