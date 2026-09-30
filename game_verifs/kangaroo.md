Greetings! As an Atari Explorer and AI researcher specializing in Reinforcement Learning and the Arcade Learning Environment (ALE), I have compiled the technical analysis you requested for the Atari 2600 version of **Kangaroo**.

Below is the detailed breakdown of the game mechanics, action space, and reward signals strictly as they are implemented within the ALE and documented in the classic manuals.

---

## Task 1: Game Mechanics & Action Space

### Action Space

In the ALE framework (accessed via Farama Gymnasium), `Kangaroo-v5` utilizes a discrete action space of 18 possible actions, which map directly to the Atari 2600 joystick and button.

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Standard Action Space (Discrete 18)** | The environment accepts integer actions from 0 to 17, which correspond to: `NOOP`, `FIRE`, `UP`, `RIGHT`, `LEFT`, `DOWN`, `UPRIGHT`, `UPLEFT`, `DOWNRIGHT`, `DOWNLEFT`, `UPFIRE`, `RIGHTFIRE`, `LEFTFIRE`, `DOWNFIRE`, `UPRIGHTFIRE`, `UPLEFTFIRE`, `DOWNRIGHTFIRE`, and `DOWNLEFTFIRE`. [Source: [https://gymnasium.farama.org/environments/atari/kangaroo/](https://www.google.com/search?q=https://gymnasium.farama.org/environments/atari/kangaroo/)] | [jax_kangaroo.py:L240-263](src/jaxatari/games/jax_kangaroo.py#L240-263) |
| **Active Movements** | Pressing `UP`, `LEFT`, `RIGHT`, and `DOWN` moves Mama Kangaroo along branches and up/down ladders. Diagonal inputs combined with `UP` trigger a jump (e.g., `UPRIGHT` jumps right). Pressing `DOWN` while on a branch makes Mama Kangaroo duck. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L832-901](src/jaxatari/games/jax_kangaroo.py#L832-901), [jax_kangaroo.py:L978-985](src/jaxatari/games/jax_kangaroo.py#L978-985), [jax_kangaroo.py:L423-546](src/jaxatari/games/jax_kangaroo.py#L423-546) |
| **Fire Button** | The `FIRE` action causes Mama Kangaroo to execute a punch. Combining `FIRE` with directional inputs allows for punching while jumping or ducking. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L872-888](src/jaxatari/games/jax_kangaroo.py#L872-888), [jax_kangaroo.py:L944-949](src/jaxatari/games/jax_kangaroo.py#L944-949) |

### Core Mechanics

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Movement & Physics** | Mama Kangaroo is constrained to walking on horizontal branches and climbing vertical ladders. Gravity is only a factor if she jumps or walks off a broken branch, leading to a loss of life if she falls to the bottom of the screen. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L547-639](src/jaxatari/games/jax_kangaroo.py#L547-639), [jax_kangaroo.py:L1022-1054](src/jaxatari/games/jax_kangaroo.py#L1022-1054), [jax_kangaroo.py:L1117-1120](src/jaxatari/games/jax_kangaroo.py#L1117-1120) |
| **Interaction with Adversaries** | Touching a monkey or a thrown apple without actively striking it with a punch results in a lost life. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1123-1198](src/jaxatari/games/jax_kangaroo.py#L1123-1198) |
| **Combat Mechanics** | Mama Kangaroo can neutralize threats by using the `FIRE` action to punch monkeys. She can also punch apples thrown by monkeys to destroy them. Alternatively, she can duck under high-thrown apples or jump over low-thrown apples. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1613-1653](src/jaxatari/games/jax_kangaroo.py#L1613-1653), [jax_kangaroo.py:L1286-1310](src/jaxatari/games/jax_kangaroo.py#L1286-1310) |
| **Bell and Fruits** | Jumping to hit the bell at the top of the screen replenishes the collectible fruits (like strawberries and pineapples) scattered across the branches. Eating these fruits grants bonus points. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L678-708](src/jaxatari/games/jax_kangaroo.py#L678-708), [jax_kangaroo.py:L709-757](src/jaxatari/games/jax_kangaroo.py#L709-757) |

### Level Progression

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Primary Objective** | The ultimate goal is to navigate from the bottom of the screen to the top branch to rescue Baby Kangaroo. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1048-1052](src/jaxatari/games/jax_kangaroo.py#L1048-1052) |
| **Adversarial Mechanics** | Monkeys spawn continuously, moving along the branches and throwing apples. As levels progress, monkeys will drop from the top of the screen to lower branches, and the frequency and speed of thrown apples increase. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1416-1666](src/jaxatari/games/jax_kangaroo.py#L1416-1666) |
| **Screen Variations** | The Atari 2600 version features three distinct screen layouts. Screen 1 focuses on basic ladders and branches; Screen 2 introduces a monkey pyramid blocking the path; Screen 3 features a complex ladder maze with heavy apple barrages. Completing Screen 3 loops the game back with increased difficulty. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L805-824](src/jaxatari/games/jax_kangaroo.py#L805-824), [jax_kangaroo.py:L1088-1104](src/jaxatari/games/jax_kangaroo.py#L1088-1104), [jax_kangaroo.py:L1852-1857](src/jaxatari/games/jax_kangaroo.py#L1852-1857) |
| **Timer Constraint** | Each screen has a strict time limit (represented by a bar or numerical value). Failing to reach Baby Kangaroo before time runs out results in a lost life. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1082-1087](src/jaxatari/games/jax_kangaroo.py#L1082-1087), [jax_kangaroo.py:L1107-1107](src/jaxatari/games/jax_kangaroo.py#L1107-1107), [jax_kangaroo.py:L1192-1198](src/jaxatari/games/jax_kangaroo.py#L1192-1198) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Punch Anti-Spam** | Holding the `FIRE` action for more than 28 consecutive steps disables punching. The agent must release the `FIRE` action to punch again. [verified] this cooldown mechanism is also present in the original game. | [jax_kangaroo.py:L926-940](src/jaxatari/games/jax_kangaroo.py#L926-940) |
| **Monkey Pyramid Absent** | The "Monkey Pyramid" on Screen 2 is not implemented in this version. Monkeys spawn individually up to a maximum of 4 on screen at once across all levels. [verified] this is not present in the Screen 2 of the ALE game | [jax_kangaroo.py:L1419-1423](src/jaxatari/games/jax_kangaroo.py#L1419-1423) |
| **Coconuts vs Apples** | The projectiles are explicitly referred to and modeled as coconuts (`thrown_coconuts` and `falling_coconut`) instead of apples. [verified] This is just renaming | [jax_kangaroo.py:L1227-1237](src/jaxatari/games/jax_kangaroo.py#L1227-1237) |

---

## Task 2: Reward Signals

In ALE, the step reward $r_t$ is strictly equivalent to the change in the in-game score ($\Delta \text{score}$). The reinforcement learning agent receives rewards based on the following specific in-game events:

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Eating Fruit (Strawberry, Apple, Cherry, etc.)** | Yields between 100 to 800 points, depending on the specific fruit type and the current level progression. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L714-720](src/jaxatari/games/jax_kangaroo.py#L714-720) |
| **Punching a Monkey** | Yields exactly 200 points per monkey destroyed. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1645-1645](src/jaxatari/games/jax_kangaroo.py#L1645-1645) |
| **Punching a Thrown Apple** | Yields exactly 200 points per apple destroyed. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1311-1311](src/jaxatari/games/jax_kangaroo.py#L1311-1311) |
| **Punching the Monkey Pyramid (Screen 2)** | Each monkey punched in the stack yields 200 points, until the stack is cleared. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1645-1645](src/jaxatari/games/jax_kangaroo.py#L1645-1645) |
| **Rescuing Baby Kangaroo** | Yields a flat 2,000 points for completing the screen. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1842-1857](src/jaxatari/games/jax_kangaroo.py#L1842-1857) |
| **Time Bonus** | Upon rescuing Baby Kangaroo, the remaining time is converted to score. Yields 100 points for every tick remaining on the timer. [Source: [https://atariage.com/manual_html_page.php?SoftwareLabelID=256](https://www.google.com/search?q=https://atariage.com/manual_html_page.php%3FSoftwareLabelID%3D256)] | [jax_kangaroo.py:L1842-1843](src/jaxatari/games/jax_kangaroo.py#L1842-1843) |
| **Penalties** | Standard ALE does not typically assign negative rewards for losing a life in `Kangaroo` unless a life-loss penalty is artificially wrapped around the environment in Gymnasium. The baseline score simply ceases to increase, and the agent loses the opportunity for the time bonus. [Source: [https://gymnasium.farama.org/environments/atari/kangaroo/](https://gymnasium.farama.org/environments/atari/kangaroo/)] | [jax_kangaroo.py:L1192-1200](src/jaxatari/games/jax_kangaroo.py#L1192-1200) |

#### In JAXAtari

| Mechanic | Description | Code Implementation |
| :--- | :--- | :--- |
| **Fruit Upgrades** | Ringing the bell does not just replenish fruits; it increases their stage up to a maximum of 3. Fruit rewards scale dynamically as `100 * (2^stage)`, providing 100, 200, 400, or 800 points depending on the stage. [verified] same behavior in the original ALE version. | [jax_kangaroo.py:L714-720](src/jaxatari/games/jax_kangaroo.py#L714-720), [jax_kangaroo.py:L740-744](src/jaxatari/games/jax_kangaroo.py#L740-744) |
| **Combined Rescue and Time Reward** | Unlike the original game which gave a flat 2,000 points PLUS a time bonus, JAXAtari starts the timer at 2000 and decrements it by 100. Upon rescuing Baby Kangaroo, the agent is simply awarded the remaining timer value (max 2000). [verified] This is the correct ALE behavior. | [jax_kangaroo.py:L1842-1843](src/jaxatari/games/jax_kangaroo.py#L1842-1843) |
