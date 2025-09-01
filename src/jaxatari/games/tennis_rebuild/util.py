import os
import chex
import pygame
import jax.numpy as jnp
import jaxatari.rendering.atraJaxis as aj

def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    BG = aj.loadFrame(os.path.join(MODULE_DIR, "../sprites/tennis/bg/1.npy"))

    frames_pl_r = []
    for i in range(1, 5):
        frame = aj.loadFrame(os.path.join(MODULE_DIR, f"../sprites/tennis/pl_red/{i}.npy"))
        frames_pl_r.append(frame)
    PL_R = jnp.array(aj.pad_to_match(frames_pl_r))

    frames_bat_r = []
    for i in range(1, 5):
        frame = aj.loadFrame(os.path.join(MODULE_DIR, f"../sprites/tennis/bat_r/{i}.npy"))
        frames_bat_r.append(frame)
    BAT_R = jnp.array(aj.pad_to_match(frames_bat_r))

    frames_pl_b = []
    for i in range(1, 5):
        frame = aj.loadFrame(os.path.join(MODULE_DIR, f"../sprites/tennis/pl_blue/{i}.npy"))
        frames_pl_b.append(frame)
    PL_B = jnp.array(aj.pad_to_match(frames_pl_b))

    frames_bat_b = []
    for i in range(1, 5):
        frame = aj.loadFrame(os.path.join(MODULE_DIR, f"../sprites/tennis/bat_b/{i}.npy"))
        frames_bat_b.append(frame)
    BAT_B = jnp.array(aj.pad_to_match(frames_bat_b))

    BALL = aj.loadFrame(os.path.join(MODULE_DIR, "../sprites/tennis/ball/1.npy"))
    BALL_SHADE = aj.loadFrame(os.path.join(MODULE_DIR, "../sprites/tennis/ball_shade/1.npy"))

    DIGITS_R = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "../sprites/tennis/digits_r/{}.npy"))
    DIGITS_B = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "../sprites/tennis/digits_b/{}.npy"))

    return BG, PL_R, BAT_R, PL_B, BAT_B, BALL, BALL_SHADE, DIGITS_R, DIGITS_B

# Action constants
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


def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(UPFIRE)
    if down and fire:
        return jnp.array(DOWNFIRE)
    if left and fire:
        return jnp.array(LEFTFIRE)
    if right and fire:
        return jnp.array(RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(UPRIGHT)
    if up and left:
        return jnp.array(UPLEFT)
    if down and right:
        return jnp.array(DOWNRIGHT)
    if down and left:
        return jnp.array(DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(UP)
    if down:
        return jnp.array(DOWN)
    if left:
        return jnp.array(LEFT)
    if right:
        return jnp.array(RIGHT)
    if fire:
        return jnp.array(FIRE)

    return jnp.array(NOOP)