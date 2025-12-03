# JAXAtari AI Agent Instructions

## Architecture Overview

**JAXAtari** is a GPU-accelerated, object-centric Atari environment framework built on JAX. The codebase enables up to 16,000x faster RL training through JIT compilation, vectorization, and GPU parallelization.

### Core Components

- **`src/jaxatari/core.py`**: Entry point (`make()` function) that loads games and applies modifications. Maps game names to module paths.
- **`src/jaxatari/environment.py`**: Abstract `JaxEnvironment` base class with methods `reset()`, `step()`, `render()`, action/observation spaces. All games inherit this.
- **`src/jaxatari/wrappers.py`**: Composable wrapper system (`AtariWrapper`, `ObjectCentricWrapper`, `PixelObsWrapper`, `FlattenObservationWrapper`, `LogWrapper`) for stacking frames, feature extraction, and logging.
- **`src/jaxatari/spaces.py`**: JAX-compatible space abstractions (`Space`, `Discrete`, `Box`, `Dict`) supporting JIT/vmap.
- **`src/jaxatari/games/jax_*.py`**: Individual game implementations (50+ Atari games). Each defines:
  - `{Game}Constants` (NamedTuple): Immutable config with physics params, colors, asset paths.
  - `{Game}State` (NamedTuple): Immutable game state.
  - `{Game}Observation` (NamedTuple): Object-centric features extracted each frame.
  - `Jax{Game}` class: Environment logic with JIT-decorated methods.
  - `{Game}Renderer`: JAX-based rendering using pre-computed sprite assets.
- **`src/jaxatari/modification.py`**: Plugin system for game modifications. Base classes: `JaxAtariInternalModPlugin`, `JaxAtariPostStepModPlugin`.
- **`src/jaxatari/games/mods/*.py`**: Per-game mod registries (e.g., `pong_mods.py`) implementing variations for generalization testing.

### Data Structures

All state and observations are immutable **NamedTuples** to ensure JAX compatibility:
- State is never mutated; `step()` returns a new state.
- NamedTuple syntax: `class MyState(NamedTuple): field: chex.Array = default_value`

### Key Design Patterns

1. **Functional Purity**: Methods decorated with `@functools.partial(jax.jit, static_argnums=(0,))` for JIT compilation of instance methods.
2. **Vectorization-Ready**: Use `jax.lax.cond()`, `jax.vmap()`, `jax.lax.scan()` instead of Python control flow to enable parallelization.
3. **Asset Management**: Sprites stored as `.npy` files in `src/jaxatari/rendering/assets/`. Games load via `ASSET_CONFIG` in constants.
4. **Renderer Separation**: Game logic and rendering are decoupled; renderer computes RGB images from state.

## Critical Development Workflows

### Running Tests

```bash
# Run all tests for all discovered games (uses pytest_generate_tests dynamic parametrization)
pytest tests/

# Run tests for a specific game
pytest tests/ --game pong

# Run a specific test class
pytest tests/test_core_and_wrappers.py::TestWrapperChaining -v

# Run regression tests (snapshot-based, only games with existing snapshots)
pytest tests/test_regression.py
```

**Testing Infrastructure** (in `tests/conftest.py`):
- `discover_games()`: Scans `src/jaxatari/games/jax_*.py` to find all games.
- `load_game_environment(game_name)`: Dynamically imports game class via spec loader.
- Fixtures: `raw_env`, `wrapped_env` (parametrized over `WRAPPER_RECIPES`).
- `WRAPPER_RECIPES` dict defines standard wrapper combinations tested on all games.

### Adding a New Game

1. Create `src/jaxatari/games/jax_{game_name}.py` with:
   - Constants NamedTuple (physics, colors, sizes)
   - State NamedTuple (all mutable fields)
   - Observation NamedTuple (extracted features per frame)
   - Info NamedTuple (extra info like lives remaining)
   - `Jax{Game}` class inheriting `JaxEnvironment`
   - Methods: `reset()`, `step()`, `render()`, `observation_space()`, `action_space()`
   - All state transitions inside `step()` must use `jax.lax.cond/scan` for JIT compatibility.

2. Add game to `GAME_MODULES` dict in `src/jaxatari/core.py`.

3. If creating mods, create `src/jaxatari/games/mods/{game_name}_mods.py` with:
   - Mod plugin classes inheriting `JaxAtariInternalModPlugin` or `JaxAtariPostStepModPlugin`
   - `{Game}EnvMod` controller class with `REGISTRY` dict
   - Register in `MOD_MODULES` in `core.py`

### Running Games Interactively

```bash
python scripts/play.py -g Pong
```

### GPU/Vectorization Benchmarking

```bash
cd scripts/benchmarks
python pure_performance_comparison.py --game pong --jax --gymnasium  # Compare JAX vs Gym scaling
python train_jaxatari_agent.py  # PPO training example with vmap + jit
```

## Project-Specific Conventions

### JAX-Compatible Code Requirements

- **No Python Loops**: Use `jax.lax.scan()` for sequential logic, `jax.lax.fori_loop()` for fixed iterations.
- **No Mutable Objects**: Never modify arrays; instead return new arrays via `jnp.where()`, `jax.lax.cond()`.
- **Static Arguments in JIT**: Mark instance reference `(0,)` and any Python-level kwargs as static in `jax.jit`.
- **PRNG Management**: Pass `jax.random.PRNGKey` explicitly; always split keys before passing to independent functions.

Example:
```python
@functools.partial(jax.jit, static_argnums=(0,))
def step(self, state, action):
    # NOT: for i in range(n): -- instead use:
    def body_fn(state, _):
        return new_state
    state = jax.lax.scan(body_fn, state, None, length=n)[0]
    return obs, state, reward, done, info
```

### Observation Space Definition

Object-centric observations must be defined as `spaces.Dict` with keys for each object type:
```python
def observation_space(self):
    return spaces.Dict({
        "player": spaces.Box(0, 255, (player_height, player_width, 3)),
        "enemies": spaces.Box(0, 255, (max_enemies, height, width, 3)),
        "projectiles": spaces.Box(0, 255, (max_proj, 2))
    })
```

### Wrapper Composition

Stack wrappers in standard order:
```python
env = ObjectCentricWrapper(AtariWrapper(jaxatari.make("pong")))  # Recommended
# OR
env = PixelObsWrapper(AtariWrapper(jaxatari.make("pong")))       # Computer vision
# Then optionally add:
env = LogWrapper(env)  # For episode metrics
```

## Integration Points & External Dependencies

- **JAX (`jax`, `chex`)**: Core array operations, JIT, vmap. Requires GPU setup via `pip install "jax[cuda12]"`.
- **Flax (`flax.struct`)**: Dataclass decorators for pytree compatibility.
- **Gymnasium/OpenAI Gym**: For action/observation space definitions (imported but JAXAtari's own `spaces` module is used).
- **ALE-py (`ale_py`)**: Legacy reference; not used in JAX implementations.
- **NumPy**: Pre-processing, asset loading. JAX arrays used in step functions.

## Key Files to Study

- `src/jaxatari/games/jax_pong.py` (729 lines): Simplest game; good reference for structure and rendering.
- `src/jaxatari/games/jax_centipede.py` (2000+ lines): Complex physics; examples of nested vmap for grid logic.
- `tests/conftest.py`: Testing infrastructure; dynamic game discovery and parametrization.
- `tests/test_core_and_wrappers.py`: Wrapper behavior tests; how to compose and verify shapes.
- `scripts/benchmarks/pure_performance_comparison.py`: How to vmap + jit environments for GPU parallelization.

## Common Pitfalls

1. **Using Python control flow in step()**: Will break JIT. Use `jax.lax.cond/while_loop/scan` instead.
2. **Mutable default arguments**: NamedTuple defaults must be immutable; use `field(default_factory=...)` pattern sparingly.
3. **Forgetting frame stacking in observations**: `AtariWrapper` stacks frames; downstream code must handle this shape.
4. **Static args mismatch**: If wrapper parameters change, JIT re-traces; document expected static args.
5. **Asset file paths**: Always relative to game module; use `Path(__file__).parent / "assets"`.
