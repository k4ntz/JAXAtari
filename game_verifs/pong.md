Here is the research breakdown for the Atari 2600 version of Pong (originally released on the *Video Olympics* cartridge), as implemented in the Arcade Learning Environment (ALE) and Farama Gymnasium.

## Task 1: Game Mechanics & Action Space

### Action Space

The default reduced action space in ALE/Gymnasium is `Discrete(6)`. The valid actions are: `NOOP`, `FIRE`, `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`.

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Standard Action Space (Discrete 6)** | The environment accepts integer actions from 0 to 5. | [jax_pong.py:L99-102](src/jaxatari/games/jax_pong.py#L99-102) |
| **Movement Mapping** | Due to how the Atari 2600 maps the analog paddle to joystick pins, the `RIGHT` and `RIGHTFIRE` actions move the paddle **UP**, while `LEFT` and `LEFTFIRE` actions move it **DOWN**. [Source: [https://ale.farama.org/environments/pong/](https://ale.farama.org/environments/pong/)] | [jax_pong.py:L110-111](src/jaxatari/games/jax_pong.py#L110-111) |

### Core Mechanics

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Player vs Opponent** | The RL agent controls the right paddle, while the left paddle is controlled by a built-in automated computer opponent. | [jax_pong.py:L109-148](src/jaxatari/games/jax_pong.py#L109-148), [jax_pong.py:L275-284](src/jaxatari/games/jax_pong.py#L275-284) |
| **Deflection Physics** | The ball bounces off the top and bottom screen boundaries by reversing its Y velocity, and off paddle front faces by reversing its X velocity. | [jax_pong.py:L154-158](src/jaxatari/games/jax_pong.py#L154-158), [jax_pong.py:L160-186](src/jaxatari/games/jax_pong.py#L160-186) |
| **Segmented Paddle Bounce Angles** | The paddle is divided into 5 vertical sections. Depending on which section of the paddle the ball hits, the resulting vertical trajectory (Y velocity) changes drastically, simulating angle control. | [jax_pong.py:L188-230](src/jaxatari/games/jax_pong.py#L188-230) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Analog Paddle Emulation** | Instead of simple direct pixel movement, JAXAtari implements an "RC Capacitor Acceleration" curve (`new_speed = speed + (target - speed) * 0.3`) to smoothly mimic turning the physical analog knob using discrete inputs. [verified] roughly corresponds to the analog -> discrete ALE version | [jax_pong.py:L126-128](src/jaxatari/games/jax_pong.py#L126-128) |
| **Bottom Dampening (Squishy Wall)** | Paddle deceleration follows an asymptotic "squishy wall" curve when nearing the bottom boundary to match ALE analog drift behavior. [verified] Also corresponds to ALE version | [jax_pong.py:L130-137](src/jaxatari/games/jax_pong.py#L130-137) |
| **Fire Boost** | Hitting the ball while the paddle is at max speed, or while holding `FIRE`/`RIGHTFIRE`/`LEFTFIRE`, explicitly boosts the ball's horizontal speed. [verified] corresponds to ALE version | [jax_pong.py:L244-260](src/jaxatari/games/jax_pong.py#L244-260) |
| **Enemy AI Handicap** | The built-in enemy paddle skips updating its position every 8th step (`step_counter % 8 != 0`) to give the agent a deterministic window to out-maneuver it. [verified] 1-1 exact ALE version behavior. | [jax_pong.py:L275-284](src/jaxatari/games/jax_pong.py#L275-284) |

### Level Progression

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **First to 21 Points** | There are no progressive stages. The game is an endless volley until either the player or the opponent reaches 21 points, which triggers the end of the game (`done=True`). [Source: [https://ale.farama.org/environments/pong/](https://ale.farama.org/environments/pong/)] | [jax_pong.py:L488-493](src/jaxatari/games/jax_pong.py#L488-493) |

---

## Task 2: Reward Signals

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Player Scores Point** | A `+1.0` reward is granted at the exact step the ball passes the opponent's paddle (ball X coordinate goes out of bounds on the left). | [jax_pong.py:L287-287](src/jaxatari/games/jax_pong.py#L287-287), [jax_pong.py:L483-486](src/jaxatari/games/jax_pong.py#L483-486) |
| **Opponent Scores Point** | A `-1.0` reward is granted when the ball passes the agent's paddle (ball X coordinate goes out of bounds on the right). | [jax_pong.py:L288-288](src/jaxatari/games/jax_pong.py#L288-288), [jax_pong.py:L483-486](src/jaxatari/games/jax_pong.py#L483-486) |
| **Volleying** | A `0.0` reward is given on every other time step while the ball is in play. | Implicit in `_get_reward` |
