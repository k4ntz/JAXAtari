# Frostbite (Atari 2600) — Arcade Learning Environment (ALE) Analysis

Below is the comprehensive technical breakdown of the Atari 2600 version of *Frostbite*, structured for Reinforcement Learning research within the ALE / Farama Gymnasium framework.

## Game Mechanics & Action Space

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Full Discrete Space** | Frostbite utilizes the full `Discrete(18)` Atari action space, meaning `NOOP`, `FIRE`, standard directional inputs, and combined firing are all active. | [jax_frostbite.py:L461-484](src/jaxatari/games/jax_frostbite.py#L461-484) |
| **Joystick (Directions)** | Moving the joystick controls Bailey's jumps back and forth between the ice floes and the shore. | [jax_frostbite.py:L461-484](src/jaxatari/games/jax_frostbite.py#L461-484) |
| **FIRE Button** | Pressing `FIRE` reverses the directional flow of the ice row the player is currently standing on. | [jax_frostbite.py:L1775-1820](src/jaxatari/games/jax_frostbite.py#L1775-1820) |

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Core Objective** | Jump across an Arctic river consisting of moving ice floes. Gather ice to build an igloo on the top shore. | Implicit in game loop |
| **Building the Igloo** | Landing on a white ice floe changes its color to blue and adds one block to the igloo. The agent must collect exactly 15 blocks to finish the igloo. | [jax_frostbite.py:L2326-2361](src/jaxatari/games/jax_frostbite.py#L2326-2361) |
| **Floe Resetting** | Once every ice block in a specific row has been turned blue, the entire row instantly reverts back to white. | [jax_frostbite.py:L2363-2370](src/jaxatari/games/jax_frostbite.py#L2363-2370) |
| **Reversal Penalty** | Reversing the direction of the ice row removes one uncompleted block from the igloo. | [jax_frostbite.py:L1811-1820](src/jaxatari/games/jax_frostbite.py#L1811-1820) |
| **Physics and Collision** | The agent can safely jump *over* or *around* enemies mid-flight because hazards only kill the player when both feet are firmly planted on the ice. | [jax_frostbite.py:L2546](src/jaxatari/games/jax_frostbite.py#L2546) |
| **Timer (Temperature)** | A temperature gauge serves as a strict timer. If it drops to zero before the agent finishes, the character freezes. | [jax_frostbite.py:L204-205](src/jaxatari/games/jax_frostbite.py#L204-205) |
| **Adversaries** | As levels progress, dodge aquatic enemies like Alaskan King Crabs, Killer Clams, and Snow Geese that travel along the ice. | [jax_frostbite.py:L2500-2637](src/jaxatari/games/jax_frostbite.py#L2500-2637) |
| **The Polar Grizzly** | Starting on Level 4 (internally Level 3 in codebase), a Polar Grizzly begins to patrol the top shore to block the player. | [jax_frostbite.py:L2638-2650](src/jaxatari/games/jax_frostbite.py#L2638-2650) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Free Reversals** | A fascinating detail in the implementation reveals that while `FIRE` usually deducts an igloo block as a penalty, once the igloo is complete (15 blocks), all subsequent ice reversals become completely *free* for the rest of the stage. [verified] same in the original game, you cannot modify a completely built igloo. | [jax_frostbite.py:L1814-1820](src/jaxatari/games/jax_frostbite.py#L1814-1820) |

---

## Reward Signals

Because ALE environments output step rewards based directly on the game's internal score counter ($\Delta \text{score}$), the RL agent receives the exact point values hardcoded into the classic game's scoring logic.

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Coloring an Ice Block** | Jumping on an uncolored (white) ice block yields 10 points multiplied by the current level number. | [jax_frostbite.py:L2349-2352](src/jaxatari/games/jax_frostbite.py#L2349-2352) |
| **Entering the Igloo** | Successfully completing a level awards a terminal reward of 160 points multiplied by the current level number (awarded per-block). | [jax_frostbite.py:L1307-1310](src/jaxatari/games/jax_frostbite.py#L1307-1310) |
| **Maximum Level Multiplier** | The level multiplier caps out at level 8. From level 9 onward, the agent receives a constant maximum of 90 points per ice block. | [jax_frostbite.py:L493-500](src/jaxatari/games/jax_frostbite.py#L493-500) |
| **Catching a Fish** | Colliding with a fish grants a flat bonus of 200 points. | [jax_frostbite.py:L2572-2580](src/jaxatari/games/jax_frostbite.py#L2572-2580) |
| **Time/Temperature Bonus** | Upon entering the igloo, any remaining time is converted into a score bonus: 10 points × remaining degrees × current level number. | [jax_frostbite.py:L1337-1341](src/jaxatari/games/jax_frostbite.py#L1337-1341) |

