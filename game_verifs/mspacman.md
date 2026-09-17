Here is the research breakdown for the Atari 2600 implementation of *Ms. Pac-Man* within the Arcade Learning Environment (ALE), formulated to assist with reinforcement learning environment design.

## Game Mechanics & Action Space

### Action Space

Within the Gymnasium/ALE framework, the Atari 2600 environment translates the physical 4-way joystick into a discrete space. When configuring the environment (typically under default settings where `full_action_space=False`), *Ms. Pac-Man* utilizes a 9-dimensional discrete action space mapping to directional inputs. There is no "FIRE" action, as the game requires only directional movement.

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Standard Action Space (Discrete 9)** | The environment accepts integer actions from 0 to 8 corresponding to: `NOOP`, `UP`, `RIGHT`, `LEFT`, `DOWN`, `UPRIGHT`, `UPLEFT`, `DOWNRIGHT`, and `DOWNLEFT`. [Source: [https://gymnasium.farama.org/v0.26.3/environments/atari/ms_pacman/](https://gymnasium.farama.org/v0.26.3/environments/atari/ms_pacman/)] | [jax_mspacman.py:L189-199](src/jaxatari/games/jax_mspacman.py#L189-199), [jax_mspacman.py:L203-214](src/jaxatari/games/jax_mspacman.py#L203-214) |

### Core Mechanics

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Player Objective** | Navigate the maze to clear it of all standard dots while actively avoiding four chasing ghosts. Clearing all dots triggers the next maze layout. [Source: [https://gymnasium.farama.org/v0.26.3/environments/atari/ms_pacman/](https://gymnasium.farama.org/v0.26.3/environments/atari/ms_pacman/)] | [jax_mspacman.py:L461-537](src/jaxatari/games/jax_mspacman.py#L461-537) |
| **Ghost AI & Stochasticity** | Ghost movement relies on randomized components rather than deterministic patterns. [Source: [https://gizmodo.com/microsofts-ai-just-shattered-the-ms-pac-man-high-score-1796091352](https://gizmodo.com/microsofts-ai-just-shattered-the-ms-pac-man-high-score-1796091352)] | [jax_mspacman.py:L659-674](src/jaxatari/games/jax_mspacman.py#L659-674) |
| **Power Pellets** | Eating one of the four energizers temporarily renders ghosts vulnerable. [Source: [https://pacman.fandom.com/wiki/Ms.*Pac-Man*(game](https://www.google.com/search?q=https%3A%2F%2Fpacman.fandom.com%2Fwiki%2FMs._Pac-Man_%28game%29)] | [jax_mspacman.py:L495-504](src/jaxatari/games/jax_mspacman.py#L495-504), [jax_mspacman.py:L548-588](src/jaxatari/games/jax_mspacman.py#L548-588) |
| **Fruits** | Bonus items spawn during the round and actively bounce through the maze paths. [Source: [https://pacman.fandom.com/wiki/Ms.*Pac-Man*(game](https://www.google.com/search?q=https%3A%2F%2Fpacman.fandom.com%2Fwiki%2FMs._Pac-Man_%28game%29)] | [jax_mspacman.py:L258-261](src/jaxatari/games/jax_mspacman.py#L258-261) |
| **Warp Tunnels** | Mazes feature connecting tunnels on the outer boundaries that transport sprites. [Source: [https://pacman.fandom.com/wiki/Ms.*Pac-Man*(game](https://www.google.com/search?q=https%3A%2F%2Fpacman.fandom.com%2Fwiki%2FMs._Pac-Man_%28game%29)] | Implicit in `get_new_position` |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Ghost Behavior Timers** | The chase and scatter modes apply a stochastic offset (`jax.random.randint`) to their base duration to introduce true randomness in routing behavior. [verified] In this game, we do not the same pseudo randomness of the original game. | [jax_mspacman.py:L591-608](src/jaxatari/games/jax_mspacman.py#L591-608) |

### Level Progression

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Environment Scaling** | The game continuously cycles through four unique maze layouts. As the player completes rounds, the baseline speed of the ghosts increases. [Source: [https://pacman.fandom.com/wiki/Ms.*Pac-Man*(game](https://www.google.com/search?q=https%3A%2F%2Fpacman.fandom.com%2Fwiki%2FMs._Pac-Man_%28game%29)] | [jax_mspacman.py:L243-244](src/jaxatari/games/jax_mspacman.py#L243-244) |
| **Adversarial Mechanics** | As the game progresses, the duration of ghost vulnerability decreases. [Source: [https://forums.atariage.com/topic/266695-microsofts-ai-just-shattered-the-ms-pac-man-high-score/](https://forums.atariage.com/topic/266695-microsofts-ai-just-shattered-the-ms-pac-man-high-score/)] | [jax_mspacman.py:L553-557](src/jaxatari/games/jax_mspacman.py#L553-557), [jax_mspacman.py:L627-631](src/jaxatari/games/jax_mspacman.py#L627-631) |
| **Ultimate Goal** | The ultimate terminal state in ALE occurs when the player loses all their discrete lives. [Source: [https://gymnasium.farama.org/v0.26.3/environments/atari/ms_pacman/](https://gymnasium.farama.org/v0.26.3/environments/atari/ms_pacman/)] | [jax_mspacman.py:L370-371](src/jaxatari/games/jax_mspacman.py#L370-371) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Frightened Decay** | The duration of frightened ghosts scales with a geometric decay factor exactly set to `0.85 ** (level - 1)`. [verified] It was a lookup table in the original game, this approximates it. | [jax_mspacman.py:L553-557](src/jaxatari/games/jax_mspacman.py#L553-557) |
| **Bonus Life** | An extra life is awarded precisely at 10,000 points. [verified] Also exists in the orginal game | [jax_mspacman.py:L53](src/jaxatari/games/jax_mspacman.py#L53), [jax_mspacman.py:L278-282](src/jaxatari/games/jax_mspacman.py#L278-282) |

---

## Rewards

Because ALE derives its step reward strictly from the internal $\Delta \text{score}$ of the emulator's memory space, your RL reward function should mirror these exact discrete point triggers mapped to the Atari 2600 scoring tables.

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Standard Objectives** | Dot (Standard Pellet): 10 points. Energizer (Power Pellet): 50 points. | [jax_mspacman.py:L92-93](src/jaxatari/games/jax_mspacman.py#L92-93), [jax_mspacman.py:L506-513](src/jaxatari/games/jax_mspacman.py#L506-513) |
| **Vulnerable Ghosts** | First Ghost: 200, Second: 400, Third: 800, Fourth: 1600. | [jax_mspacman.py:L95](src/jaxatari/games/jax_mspacman.py#L95), [jax_mspacman.py:L786-793](src/jaxatari/games/jax_mspacman.py#L786-793) |
| **Fruit & Snack Prizes** | Cherry: 100, Strawberry: 200, Orange: 500, Pretzel: 700, Apple: 1000, Pear: 2000, Banana: 5000. | [jax_mspacman.py:L94](src/jaxatari/games/jax_mspacman.py#L94) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Ghost Multiplier Logic** | The ghost eating reward dynamically scales by `200 * (2 ** eaten_ghosts)`, applying cumulatively and fully vectorized using a prefix sum (`jnp.cumsum`) over ghosts eaten on the exact same frame. [verified] Maps the original version of the game. | [jax_mspacman.py:L782-795](src/jaxatari/games/jax_mspacman.py#L782-795) |
