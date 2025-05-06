# 🎮 JAXAtari: JAX-Based Object-Centric Atari Environments

Quentin Delfosse, Daniel Kirn, Dominik Mandok, Paul Seitz, Lars Teubner, Sebastian Wette  
[Machine Learning Lab – TU Darmstadt](https://www.ml.informatik.tu-darmstadt.de/)

> A GPU-accelerated, object-centric Atari environment suite built with JAX for fast, scalable reinforcement learning research.

---

**JAXAtari** introduces a GPU-accelerated, object-centric Atari environment framework powered by [JAX](https://github.com/google/jax). Inspired by [OCAtari](https://github.com/k4ntz/OC_Atari), this framework enables up to **16,000x faster training speeds** through just-in-time (JIT) compilation, vectorization, and massive parallelization on GPU.

<!-- --- -->

## Features
- Object-centric extraction of Atari game states.
- JAX-based vectorized execution with GPU support.
- Compatible API with ALE to ease integration.
- Benchmarking tools.


📘 [Read the Documentation](https://jaxatari.readthedocs.io/en/latest/) 


<!-- [**📘 JAXAtari Documentation**] -->

## Getting Started

<!-- ### Prerequisites -->
### Install
```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -U pip
pip3 install -e .
```

## Usage

Using an environment:
```python
import jax

from jaxatari.games.jax_seaquest import JaxSeaquest
from jaxatari.wrappers import FlattenObservationWrapper, AtariWrapper

rng = jax.random.PRNGKey(0)

env = JaxSeaquest()
env = FlattenObservationWrapper(env)
env = AtariWrapper(env)

vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
    jax.random.split(rng, n_envs)
)
vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
    env.step
)(jax.random.split(rng, n_envs), env_state, action)

init_obs, env_state = vmap_reset(128)(rng)
action = jax.random.randint(rng, (128,), 0, env.action_space().n)

# Take one step
new_obs, new_env_state, reward, done, info = vmap_step(128)(rng, env_state, action)

# Take 100 steps with scan
def step_fn(carry, unused):
    _, env_state = carry
    new_obs, new_env_state, reward, done, info = vmap_step(128)(rng, env_state, action)
    return (new_obs, new_env_state), (reward, done, info)

carry = (init_obs, env_state)
_, (rewards, dones, infos) = jax.lax.scan(
    step_fn, carry, None, length=100
)
```


Running a game manually:
```bash
python3 -m jaxatari.games.jax_seaquest
```

---

## Supported Games

| Game     | Supported |
|----------|-----------|
| Seaquest | ✅        |
| Pong     | ✅        |
| Kangaroo | ✅        |
| Freeway  | ✅        |

> More games can be added via the uniform wrapper system.

---

## Contributing

Contributions are welcome!

1. Fork this repository  
2. Create your feature branch: `git checkout -b feature/my-feature`  
3. Commit your changes: `git commit -m 'Add some feature'`  
4. Push to the branch: `git push origin feature/my-feature`  
5. Open a pull request  

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---
