# Contributing to JAXAtari

Thank you for contributing! This guide covers the three most common contribution paths:

- [Adding a mod](#adding-a-mod)
- [Adding an environment](#adding-an-environment)
- [Adding a wrapper](#adding-a-wrapper)

For any of these, the workflow is:

1. Fork the repo and create a feature branch
2. Implement your change (details below)
3. Open a pull request — the CI suite runs targeted tests automatically

---

## Adding a mod

All modifications are located under `src/jaxatari/games/mods/`. Each environment that has modifications has its own subfolder for specific mods and a top-level controller file to register them. If there is no folder or controller file yet, one should be created in the same pattern as for the existing environments.

### File structure

```
src/jaxatari/games/mods/
├── pong_mods.py                   ← mod controller (REGISTRY + JaxAtariModController)
└── pong/
    ├── pong_mod_plugins.py        ← individual plugin classes
    └── sprites/                   ← custom sprites for optional overrides
```

### Step 1 — write a plugin class

Open (or create) `src/jaxatari/games/mods/<environment>/<environment>_mod_plugins.py`.

Choose one of two base classes depending on what you want to do:

#### `JaxAtariInternalModPlugin` — patch methods or override constants

Use this when your mod replaces a method that runs *inside* the game's step logic (e.g. enemy AI, collision logic, movement).

```python
from functools import partial
import jax
from jaxatari.modification import JaxAtariInternalModPlugin

class LazyEnemyMod(JaxAtariInternalModPlugin):
    # Optional: declare conflicts with other mod keys. Will throw an error if mods with conflicting functionality are enabled together.
    conflicts_with = ["random_enemy"]

    # Optional: override numeric constants at construction time
    constants_overrides = {"ENEMY_STEP_SIZE": 1}

    # Optional: override instance attributes set in __init__
    attribute_overrides = {"ACTION_SET": ...}

    # Override a method by defining it with the same name.
    # self._env gives access to the wrapped environment instance.
    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state):
        direction = jnp.sign(state.ball_y - state.enemy_y)
        new_y = state.enemy_y + direction * self._env.consts.ENEMY_STEP_SIZE
        return state.replace(enemy_y=new_y.astype(jnp.int32))
```

Every method you define on the plugin replaces the corresponding method on the environment. Only define what you want to change — everything else is left untouched. Sometimes the environments are very monolithic and there are no specific methods to replace. In that case it is possible to insert hook functions (empty functions called at the point at which you need to modify the environment logic) into the environment logic itself. This is a more advanced technique and should be used with caution.

#### `JaxAtariPostStepModPlugin` — run logic after the step completes

Use this when your mod reads or modifies state *after* the main step has already run (e.g. clamping a score, injecting an event). This will cover most use cases that dont necessitate in-depth mechanics modifications.

```python
from functools import partial
import jax
from jaxatari.modification import JaxAtariPostStepModPlugin

class AlwaysZeroScoreMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # prev_state = state before step, new_state = state after step
        # Return the (possibly modified) new state.
        return new_state.replace(player_score=jnp.array(0, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        # Called after reset(). Return (obs, state).
        return obs, state
```

`PostStep` plugins also support `constants_overrides`, `attribute_overrides`, and `asset_overrides` exactly like internal plugins.

### Step 2 — register the modifications in the mod controller

Open (or create) `src/jaxatari/games/mods/<environment>_mods.py`:

```python
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.<environment>.<environment>_mod_plugins import LazyEnemyMod, AlwaysZeroScoreMod

class PongEnvMod(JaxAtariModController):
    REGISTRY = {
        "lazy_enemy": LazyEnemyMod,
        "zero_score": AlwaysZeroScoreMod,
    }

    def __init__(self, env, mods_config=[], allow_conflicts=False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
```

### Step 3 — register the controller in `core.py`
(only necessary if the environment is not already registered in `GAME_MODULES` / does not have modifications yet)

Add an entry to `MOD_MODULES` in `src/jaxatari/core.py`:

```python
MOD_MODULES = {
    ...
    "pong": "jaxatari.games.mods.pong_mods.PongEnvMod",
}
```

### Testing

Once registered in `MOD_MODULES`, `tests/test_all_mods.py` picks up your mod automatically — no changes to test files needed. Run targeted mod tests with:

```bash
pytest tests/test_all_mods.py --game pong --slow
```

---

## Adding an environment

### Step 1 — implement `JaxEnvironment`

Create `src/jaxatari/games/jax_<environment>.py`. Your class must subclass `JaxEnvironment` and implement the full required interface, including helper hooks used by utility code:

```python
import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from jaxatari.environment import JaxEnvironment, ObjectObservation
from jaxatari.renderers import JAXGameRenderer

@struct.dataclass
class MyGameState:
    # All mutable state fields go here. Use jnp arrays for everything
    # that changes during a step so JAX can trace through it.
    player_x: jnp.ndarray
    player_y: jnp.ndarray
    score: jnp.ndarray
    step_counter: jnp.ndarray
    key: jnp.ndarray

class JaxMyGame(JaxEnvironment):

    def __init__(self):
        super().__init__()
        self.renderer = JAXGameRenderer(self)
        # Define a minimal action set (subset of the 18 ALE actions)
        self.ACTION_SET = jnp.array([0, 1, 2, 3], dtype=jnp.int32)  # NOOP, FIRE, UP, RIGHT

    # ------------------------------------------------------------------ #
    #  Required methods                                                    #
    # ------------------------------------------------------------------ #

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray):
        state = MyGameState(
            player_x=jnp.array(80, dtype=jnp.int32),
            player_y=jnp.array(100, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=key,
        )
        obs = self._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: MyGameState, action: jnp.ndarray):
        # ... game logic ...
        new_state = state.replace(step_counter=state.step_counter + 1)
        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)
        return obs, new_state, reward, done, info

    def observation_space(self):
        # Return a spaces.Dict of ObjectObservation spaces — one entry per object type.
        from jaxatari.spaces import Dict, Box
        return Dict({
            "player": Box(low=0, high=255, shape=(8,), dtype=jnp.int32),
        })

    def action_space(self):
        from jaxatari.spaces import Discrete
        return Discrete(len(self.ACTION_SET))

    def image_space(self):
        from jaxatari.spaces import Box
        return Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def render(self, state: MyGameState) -> jnp.ndarray:
        # Return an RGB image array of shape (H, W, 3), dtype uint8.
        return self.renderer.render(state)

    # ------------------------------------------------------------------ #
    #  Required helper hooks                                                #
    # ------------------------------------------------------------------ #

    def _get_observation(self, state: MyGameState):
        # Use ObjectObservation.create() for structured per-object observations.
        player = ObjectObservation.create(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(8, dtype=jnp.int32),
            height=jnp.array(8, dtype=jnp.int32),
        )
        return player  # or a dataclass/namedtuple grouping multiple objects

    def _get_info(self, state: MyGameState):
        return {}

    def _get_reward(self, prev_state: MyGameState, state: MyGameState) -> jnp.ndarray:
        return (state.score - prev_state.score).astype(jnp.float32)

    def _get_done(self, state: MyGameState) -> jnp.ndarray:
        return jnp.array(False)
```

Key requirements:
- **State must be a `@struct.dataclass`** — all fields must be JAX-traceable (jnp arrays or pytrees). No Python-side mutable state.
- **All step/reset logic must be JIT-compatible** — no Python control flow on traced values; use `jax.lax.cond`, `jax.lax.switch`, etc.
- **Runtime methods are required**: `reset`, `step`, `render`, `action_space`, `observation_space`, `image_space`.
- **Helper hooks are required**: `_get_observation`, `_get_info`, `_get_reward`, `_get_done`.
- **Define `ACTION_SET` for every environment** so action indexing is consistent across wrappers, tools, and evaluation code.
- **`_get_reward` must accept `(prev_state, state)`** — utility/mod post-step pipelines depend on this signature.
- **For full mod pipeline support, provide environment constants** (e.g. via `AutoDerivedConstants`) so mods can introspect shared game metadata.
- **Use `ObjectObservation.create()`** for object-centric observations so wrappers can handle them generically.

### Renderer requirements

If your environment provides a custom renderer, follow these requirements:

- **Inherit from `JAXGameRenderer`** so renderer behavior stays compatible with wrappers and tooling.
- **Keep the renderer constructor signature compatible**: `__init__(self, consts=None, config=None)`.
- **Store/pass through `config` as `RendererConfig`** (for example `self.config = config or RendererConfig(...)`) so render settings propagate correctly.
- **Respect `self.config.downscale` in `render(...)`** so downscaled rendering works end-to-end.
- **Use environment constants (`consts`) inside the renderer** so metadata-driven mod hooks can introspect and patch consistently.

Generally compare your environment against existing environments and how they implement framework related logic or contact a maintainer through the PR process.

### Step 2 — register in `core.py`

Add to `GAME_MODULES` in `src/jaxatari/core.py`:

```python
GAME_MODULES = {
    ...
    "mygame": "jaxatari.games.jax_mygame",
}
```

### Step 3 — update `games_covered.md`

Add a row to the appropriate section in [games_covered.md](games_covered.md) with an honest quality rating:

| Rating | Meaning |
|--------|---------|
| 🥇 | Very close to the original, well optimised |
| 🥈 | Close to the original, may miss some mechanics, is not yet optimised or has bugs |
| 🥉 | Differs significantly from the original |
| ❌ | Not yet supported |

### Testing

`conftest.py` discovers environments automatically by scanning for `jax_*.py` files — no changes to test files needed. The full wrapper test matrix runs against your new environment. Run it with:

```bash
pytest tests/ --game mygame --slow
```

This exercises all wrapper recipes (`AtariWrapper`, `PixelObsWrapper`, `ObjectCentricWrapper`, etc.) against your environment and checks shape consistency, space containment, and basic reset/step behaviour.

For regression testing, record a baseline trajectory:

```bash
python scripts/trajectory_regression.py --game mygame --record
```

Then verify it on future changes:

```bash
python scripts/trajectory_regression.py --game mygame
```

---

## Adding a wrapper

### Step 1 — subclass `JaxatariWrapper`

Add your wrapper to `src/jaxatari/wrappers.py`:

```python
import functools
import jax
import chex
from jaxatari.wrappers import JaxatariWrapper

class MyWrapper(JaxatariWrapper):
    """One-line description of what this wrapper does."""

    def __init__(self, env, my_param: float = 1.0):
        super().__init__(env)   # sets self._env; __getattr__ proxies everything else
        self._my_param = my_param

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, state = self._env.reset(key)
        obs = self._transform(obs)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, new_state, reward, terminated, truncated, info = self._env.step(state, action)
        obs = self._transform(obs)
        return obs, new_state, reward, terminated, truncated, info

    def observation_space(self):
        # Return the transformed observation space.
        # self._env.observation_space() gives you the upstream space.
        ...

    def action_space(self):
        return self._env.action_space()

    def _transform(self, obs):
        ...
```

Things to keep in mind:

- **`super().__init__(env)` is required** — it stores `self._env` and enables the `__getattr__` proxy that forwards unknown attributes (like `render`, `image_space`, `ACTION_SET`, etc.) to the wrapped environment.
- **Decorate `reset` and `step` with `@functools.partial(jax.jit, static_argnums=(0,))`** — this is the convention across all existing wrappers.
- **Do not store mutable Python state** that changes after `__init__` — wrapper instances are treated as static by JIT.
- **Wrapper ordering** — if your wrapper depends on `AtariWrapper` being upstream (e.g. it reads `AtariState`), add an `assert isinstance(env, AtariWrapper)` in `__init__` like the observation wrappers do.

### Step 2 — export from `__init__.py`

Add the wrapper to the exports in `src/jaxatari/__init__.py` and ensure it's importable from `jaxatari.wrappers`.

### Step 3 — add to the test suite

You can also add a recipe to `WRAPPER_RECIPES` in `tests/conftest.py` so every environment is tested against your wrapper:

```python
WRAPPER_RECIPES = {
    ...
    "MyWrapped": lambda env: MyWrapper(ObjectCentricWrapper(AtariWrapper(env)), my_param=0.5),
}
```

Run the test suite:

```bash
pytest tests/test_core_and_wrappers.py --slow
```

---

## Opening a pull request

When you open a PR against `dev`, the CI pipeline (`Internal PR Tests`) runs automatically. It detects which game files changed and runs the targeted test subset — you do not need to run the full suite locally for every game.

If you are adding a new game or mod, run the targeted tests locally first to catch issues early:

```bash
pytest tests/ --game <your_game> --slow
```

Please also check `games_covered.md` is up to date before contributing via PR.
