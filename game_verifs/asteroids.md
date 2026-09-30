Here is a comprehensive research report on the Atari 2600 version of **Asteroids** as implemented in the Arcade Learning Environment (ALE) and Farama Gymnasium.

[INCOMPLETE]

### Task 1: Game Mechanics & Action Space

**The Action Space**
In the ALE/Gymnasium framework, *Asteroids* is played using a reduced subset of the 18 standard Atari actions. There are 14 meaningful actions active in the default environment:

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Standard Action Space (Discrete 14)** | The environment accepts 14 meaningul actions: `NOOP`, `FIRE`, `UP`, `RIGHT`, `LEFT`, `DOWN`, `UPRIGHT`, `UPLEFT`, `UPFIRE`, `RIGHTFIRE`, `LEFTFIRE`, `DOWNFIRE`, `UPRIGHTFIRE`, and `UPLEFTFIRE`. [Source: [https://gymnasium.farama.org/v0.27.1/environments/atari/asteroids/](https://gymnasium.farama.org/v0.27.1/environments/atari/asteroids/)] | [jax_asteroids.py:L307-325](src/jaxatari/games/jax_asteroids.py#L307-325) |

**Core Mechanics**

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Physics** | The player controls a spaceship in a wrapping 2D plane. The ship's movement relies on inertia; applying thrust (`UP`) pushes the ship forward, and it will continue drifting until counter-thrust is applied. | [jax_asteroids.py:L348-353](src/jaxatari/games/jax_asteroids.py#L348-353), [jax_asteroids.py:L355-456](src/jaxatari/games/jax_asteroids.py#L355-456) |
| **Asteroid Splitting** | Shooting a Large asteroid splits it into two Medium asteroids. Shooting a Medium asteroid splits it into two Small asteroids. Shooting a Small asteroid destroys it completely. | [jax_asteroids.py:L822-869](src/jaxatari/games/jax_asteroids.py#L822-869) |
| **Hazards** | Colliding with any asteroid instantly destroys the player's ship, costing one life. | [jax_asteroids.py:L636-679](src/jaxatari/games/jax_asteroids.py#L636-679), [jax_asteroids.py:L1073-1078](src/jaxatari/games/jax_asteroids.py#L1073-1078) |
| **Defensive Features** | Pulling `DOWN` activates Hyperspace: Teleports the ship to a random location on the screen. (Shields and Flip are not implemented). | [jax_asteroids.py:L457-467](src/jaxatari/games/jax_asteroids.py#L457-467) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Hyperspace Randomness** | Hyperspace teleportation uses `jax.random.randint` bounded by `MIN_PLAYER_X` / `Y` and `MAX_PLAYER_X` / `Y`. | [jax_asteroids.py:L458-461](src/jaxatari/games/jax_asteroids.py#L458-461) |



**Level Progression**

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Waves** | The game begins with a set of large asteroids. Once every asteroid is completely destroyed, a new wave begins. | [jax_asteroids.py:L893-949](src/jaxatari/games/jax_asteroids.py#L893-949), [jax_asteroids.py:L1116-1121](src/jaxatari/games/jax_asteroids.py#L1116-1121) |
| **Difficulty Scaling** | Subsequent waves spawn more large asteroids and objects speed increases. | N/A (Not implemented) |
| **Adversaries** | Enemy spacecraft (Satellites and UFOs) appear periodically. | N/A (Not implemented) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Missing Adversaries & Scaling** | In JAXAtari, difficulty does not currently scale with level progression. `NEW_ASTEROIDS_COUNT` remains constant at 6. Additionally, UFOs and Satellites are completely missing from the codebase. | [jax_asteroids.py:L229](src/jaxatari/games/jax_asteroids.py#L229), [jax_asteroids.py:L949](src/jaxatari/games/jax_asteroids.py#L949) |



---

### Task 2: Reward Signals

In ALE environments, the reward signal ($\Delta \text{score}$) at any given step is strictly equal to the points earned in the game during that frame. The precise point values in the Atari 2600 version of Asteroids are as follows:

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Asteroids** | +20 for Large, +50 for Medium, +100 for Small. | [jax_asteroids.py:L874-891](src/jaxatari/games/jax_asteroids.py#L874-891) |
| **Adversaries** | +200 for Satellite, +1,000 for UFO. | N/A (Not implemented) |
| **Extra Life** | An extra life is earned at regular intervals (default 5,000 points). | [jax_asteroids.py:L1108](src/jaxatari/games/jax_asteroids.py#L1108) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Score Calculation** | The transition score is calculated based strictly on the differences in asteroid states (sizes) between frames, comparing previous size to new size directly to deduce hits. | [jax_asteroids.py:L874-891](src/jaxatari/games/jax_asteroids.py#L874-891) |

---

**Sources Used:**

* Farama Gymnasium Asteroids Environment Documentation: [https://gymnasium.farama.org/v0.27.1/environments/atari/asteroids/](https://gymnasium.farama.org/v0.27.1/environments/atari/asteroids/)
* Original Atari 2600 Asteroids Game Manual (Hosted by PixelatedArcade): [https://pixelatedarcade.s3.us-east-005.dream.io/pdf/Game/1622/Asteroids-1988-Release-Atari-2600-Instruction-Manual.pdf](https://pixelatedarcade.s3.us-east-005.dream.io/pdf/Game/1622/Asteroids-1988-Release-Atari-2600-Instruction-Manual.pdf)
