# 🎮 JAXAtari: JAX-Based Object-Centric Atari Environments

Quentin Delfosse, Raban Emunds, Jannis Blüml, Paul Seitz, Sebastian Wette, Dominik Mandok
[AI/ML Lab – TU Darmstadt](https://www.aiml.informatik.tu-darmstadt.de/)

> A GPU-accelerated, object-centric Atari environment suite built with JAX for fast, scalable reinforcement learning research.

---

<div class="collage">
  <div class="row" align="center">
    <img src="./docs/source/_static/gifs/pong.gif" alt="Pong" width="24%">
    <img src="./docs/source/_static/gifs/beamrider.gif" alt="Beamrider" width="24%">
    <img src="./docs/source/_static/gifs/phoenix.gif" alt="Phoenix" width="24%">
    <img src="./docs/source/_static/gifs/tennis.gif" alt="Tennis" width="24%">
  </div>

  <div class="row" align="center">
    <img src="./docs/source/_static/gifs/skiing.gif" alt="Skiing" width="24%">
    <img src="./docs/source/_static/gifs/montezumarevenge.gif" alt="Montezuma" width="24%">
    <img src="./docs/source/_static/gifs/seaquest.gif" alt="Seaquest" width="24%">
    <img src="./docs/source/_static/gifs/kangaroo.gif" alt="Kangaroo" width="24%">
  </div>

  <div class="row" align="center">
    <img src="./docs/source/_static/gifs/freeway.gif" alt="Freeway" width="24%">
    <img src="./docs/source/_static/gifs/venture.gif" alt="Venture" width="24%">
    <img src="./docs/source/_static/gifs/qbert.gif" alt="Qbert" width="24%">
    <img src="./docs/source/_static/gifs/frostbite.gif" alt="Frostbite" width="24%">
  </div>

  <div class="row" align="center">
    <img src="./docs/source/_static/gifs/bankheist.gif" alt="Bankheist" width="24%">
    <img src="./docs/source/_static/gifs/mspacman.gif" alt="Ms. PacMan" width="24%">
    <img src="./docs/source/_static/gifs/gravitar.gif" alt="Gravitar" width="24%">
    <img src="./docs/source/_static/gifs/enduro.gif" alt="Enduro" width="24%">
  </div>
</div>

---
## Available Environments

| Environment     | Mods available | Runtime |
| --------------- | -------------- | ------- |
| Pong            | -              | -       |
| Beamrider       | -              | -       |
| Phoenix         | -              | -       |
| Tennis          | -              | -       |
| Skiing          | -              | -       |
| Montezuma       | -              | -       |
| Seaquest        | -              | -       |
| Kangaroo        | -              | -       |
| Freeway         | -              | -       |
| Venture         | -              | -       |
| Qbert           | -              | -       |
| Frostbite       | -              | -       |
| Bankheist       | -              | -       |
| Ms. PacMan      | -              | -       |
| Gravitar        | -              | -       |
| Enduro          | -              | -       |

---

**JAXAtari** introduces a GPU-accelerated, object-centric Atari environment framework powered by [JAX](https://github.com/google/jax). Inspired by [OCAtari](https://github.com/k4ntz/OC_Atari), this framework enables up to **16,000x faster training speeds** through just-in-time (JIT) compilation, vectorization, and massive parallelization on GPU.
Similar to [HackAtari](https://github.com/k4ntz/HackAtari), it implements a number of small **game modifications** , for simple testing of the generalization capabilities of agents. 

<!-- --- -->

## Features
- **Object-centric extraction** of Atari game states with structured observations
- **JAX-based vectorized execution** with full GPU support and JIT compilation
- **Comprehensive wrapper system** for different observation types (pixel, object-centric, combined)
- **Game modifications** to test agent generalization across distribution shifts (+ simple implementation of custom modifications).

📘 [Read the Documentation](https://jaxatari.readthedocs.io/en/latest/) 

## Getting Started

<!-- ### Prerequisites -->
### Install
```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -U pip
pip3 install -e .
```

**Note**: This will install JAX without GPU acceleration.

**CUDA Users** should run the following to add GPU support:
```bash
pip install -U "jax[cuda12]"
```

For other accelerator types, please follow the instructions [here](https://docs.jax.dev/en/latest/installation.html).

**Note**: Next, you need to download the original Atari 2600 sprites. Before downloading, you will be asked to confirm ownership of the original ROMs.

```bash
.venv/bin/install_sprites
```

## Usage

### Basic Environment Creation

The main entry point is the `make()` function:

```python
import jax
import jaxatari

# Create an environment
env = jaxatari.make("pong")  # or "seaquest", "kangaroo", "freeway", etc.

# Get available games
available_games = jaxatari.list_available_games()
print(f"Available games: {available_games}")
```

### Using Modifications

JAXAtari provides some pre-implemented game modifications: 

```python
import jax
import jaxatari

# Create base environment
base_env = jaxatari.make("pong")
# pong environment with lazy_enemy mod
mod_env = jaxatari.make("pong", mods=["lazy_enemy"])
# you may apply multiple mods simultaneously
mod_env = jaxatari.make("pong", mods=["lazy_enemy", "shift_enemy"])
```
Developing custom modications is well supported via the JaxAtariModController.
Feel free to share them by opening a PR.

### Using Wrappers

JAXAtari provides a comprehensive wrapper system for different use cases:

```python
import jax
import jaxatari
from jaxatari.wrappers import (
    AtariWrapper, 
    ObjectCentricWrapper, 
    PixelObsWrapper,
    PixelAndObjectCentricWrapper,
    FlattenObservationWrapper,
    LogWrapper
)

# Create base environment
base_env = jaxatari.make("pong")

# Apply wrappers for different observation types
atari_env = AtariWrapper(base_env)
env = ObjectCentricWrapper(atari_env)  # Returns flattened object features
# OR
env = PixelObsWrapper(atari_env)  # Returns pixel observations
# OR
env = PixelAndObjectCentricWrapper(atari_env)  # Returns both
# OR
env = FlattenObservationWrapper(ObjectCentricWrapper(atari_env))  # Returns flattened observations

# Add logging wrapper for training
env = LogWrapper(env)
```

### Vectorized Stepping Example

```python
import jax
import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper

# Create environment with wrappers
base_env = jaxatari.make("pong")
env = FlattenObservationWrapper(ObjectCentricWrapper(AtariWrapper(base_env)))
n_envs = 1024
rng = jax.random.PRNGKey(0)
reset_keys = jax.random.split(rng, n_envs)

# Initialize n_envs parallel environments
init_obs, env_state = jax.vmap(env.reset)(reset_keys)

# Take one random step in each env
action = jax.random.randint(rng, (n_envs,), 0, env.action_space().n)
new_obs, new_env_state, reward, terminated, truncated, info = jax.vmap(env.step)(env_state, action)

# Take 100 steps with scan
def step_fn(carry, unused):
    _, env_state = carry
    new_obs, new_env_state, reward, terminated, truncated, info = jax.vmap(env.step)(env_state, action)
    return (new_obs, new_env_state), (reward, terminated, truncated, info)

carry = (init_obs, env_state)
_, (rewards, terminations, truncations, infos) = jax.lax.scan(
    step_fn, carry, None, length=100
)
```

### Manual Game Play

Run a game manually with human input (e.g. on Pong):
```bash
pip install pygame
```

```bash
python3 scripts/play.py -g Pong
```

---

## Supported Games
Please find a list of currently supported environments and their status in [games_covered](games_covered.md)

---

## Wrapper System

JAXAtari provides several wrappers to customize environment behavior:

- **`AtariWrapper`**: Base wrapper with atari-specific pre-processing steps. Sane defaults: same as stable_baselines and cleanRL.
- **`ObjectCentricWrapper`**: Returns flattened object-centric features (2D array: `[frame_stack, features]`)
- **`PixelObsWrapper`**: Returns pixel observations (4D array: `[frame_stack, height, width, channels]`)
- **`PixelAndObjectCentricWrapper`**: Returns both pixel and object-centric observations
- **`FlattenObservationWrapper`**: Flattens any observation structure to a single 1D array
- **`LogWrapper`**: Tracks episode returns and lengths for training
- **`MultiRewardWrapper`**: Allows evaluating additional rewards.
- **`MultiRewardLogWrapper`**: Tracks multiple reward components separately. Use in combination with MultiRewardWrapper.

---

## Contributing

Contributions are welcome!

1. Fork this repository  
2. Create your feature branch: `git checkout -b feature/my-feature`  
3. Commit your changes: `git commit -m 'Add some feature'`  
4. Push to the branch: `git push origin feature/my-feature`  
5. Open a pull request  

---
## Cite us

```bibtex
@misc{jaxatari2026,
  author = {Delfosse, Quentin and Emunds, Raban and Seitz, Paul and Wette, Sebastian and Bl{\"u}ml, Jannis and Kersting, Kristian},
  title = {JAXAtari: A High-Performance Framework for Reasoning agents},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/k4ntz/JAXAtari/},
}
```
---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---
