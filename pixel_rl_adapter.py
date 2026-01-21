import os
from typing import Sequence
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from jaxatari.environment import JaxEnvironment, JAXAtariAction
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper

# [CRITICAL] Prevent JAX from reserving 90% of VRAM on startup.
# This ensures PyTorch (GDINO) and JAX (PixelRL) can run on the same GPU.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"

# Network Backbone Definition
class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x shape before: (BATCH, 4, 84, 84)
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        print("net: ", x.shape)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)

class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

class PixelRLAgent:
    def __init__(self, env: JaxEnvironment, checkpoint_path="pixelrl.params", device="cuda"):
        print(f"[PixelRL] Initializing JAX Baseline from {checkpoint_path}...")

        # NOTE: I assume access to the original JAXAtari env.
        # TODO: Remove if original env is already wrapped
        # Required for preprocessing image
        self.env = PixelObsWrapper(
            env,
            do_pixel_resize=True,
            pixel_resize_shape=(84, 84),
            grayscale=True,
        )

        # Some initial key
        key = jax.random.PRNGKey(0)
        
        # CNN Backbone
        self.network = Network()
        # Actor-Critic Heads
        self.actor = Actor(action_dim=self.env.action_space().n)
        self.critic = Critic()

        key, network_key, actor_key, critic_key = jax.random.split(key, 4)
        self.key, network_key2, actor_key2, critic_key2 = jax.random.split(key, 4)
        obs_init_sample = self.env.observation_space().sample(network_key2).squeeze()[None, ...]
        self.network_params = self.network.init(network_key, obs_init_sample)
        self.actor_params = self.actor.init(actor_key, self.network.apply(self.network_params, np.array([self.env.observation_space().sample(actor_key2).squeeze()])))
        # NOTE: critic_params is not used (but can be used to estimate state values) 
        self.critic_params = self.critic.init(critic_key, self.network.apply(self.network_params, np.array([self.env.observation_space().sample(critic_key2).squeeze()]))) 

        # actually load the params from checkpoint
        with open(checkpoint_path, "rb") as f:
            (args, (self.network_params, self.actor_params, self.critic_params)) = flax.serialization.from_bytes(
                (None, (self.network_params, self.actor_params, self.critic_params)), f.read()
            )

    @partial(jax.jit, static_argnums=0)
    def get_action_and_value(
        self,
        network_params: flax.core.FrozenDict,
        actor_params: flax.core.FrozenDict,
        next_obs: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = self.network.apply(network_params, next_obs)
        logits = self.actor.apply(actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    def predict(self, frame_np):
        """
        Args:
            frame_np: Raw Atari frame (4, H, W, 3) in RGB.
        Returns:
            action_str: "UP", "DOWN", "LEFT", "RIGHT", "PUNCH", "WAIT"
        """
        # ---------------------------------------------------------
        # 2: Preprocessing (Resize, Normalize, etc.)
        # ---------------------------------------------------------
        # Example:
        # frame_resized = cv2.resize(frame_np, (84, 84))
        # obs = jnp.array(frame_resized / 255.0)

        # For ALE, obs are typically a stack of 4 images
        # So I just assume this is the case here? (4, H, W, 3)
        obs = jnp.array(frame_np)
        # This only resizes and grayscales the image
        # Vmap over the stack dimension -> preprocess each image in the stack
        obs = jax.vmap(self.env._preprocess_image)(obs)
        print("obs after preprocess_image:", obs.shape)
        # So now, obs is (4, 84, 84, 1)
        # make it (1, 4, 84, 84)
        obs = jnp.transpose(obs, (3, 0, 1, 2))
        # Final normalization (/255) is done in the network itself
        print("final preprocessed obs shape:", obs.shape)

        # ---------------------------------------------------------
        # 3: Run Inference
        # ---------------------------------------------------------
        # NOTE: this is sampled, we could also take argmax directly
        action, self.key = self.get_action_and_value(
            self.network_params,
            self.actor_params,
            obs,
            self.key
        )

        return self._map_to_thinkrl(action.item())

    def _map_to_thinkrl(self, action_idx):
        # Standard Atari (ALE) Mapping

        # Do you actually need actions as text?

        # JaxAtari Action Mapping
        mapping = {
            JAXAtariAction.NOOP: "NOOP",
            JAXAtariAction.FIRE: "FIRE",
            JAXAtariAction.UP: "UP",
            JAXAtariAction.RIGHT: "RIGHT",
            JAXAtariAction.LEFT: "LEFT",
            JAXAtariAction.DOWN: "DOWN",
            JAXAtariAction.UPRIGHT: "UPRIGHT",
            JAXAtariAction.UPLEFT: "UPLEFT",
            JAXAtariAction.DOWNRIGHT: "DOWNRIGHT",
            JAXAtariAction.DOWNLEFT: "DOWNLEFT",
            JAXAtariAction.UPFIRE: "UPFIRE",
            JAXAtariAction.RIGHTFIRE: "RIGHTFIRE",
            JAXAtariAction.LEFTFIRE: "LEFTFIRE",
            JAXAtariAction.DOWNFIRE: "DOWNFIRE",
            JAXAtariAction.UPRIGHTFIRE: "UPRIGHTFIRE",
            JAXAtariAction.UPLEFTFIRE: "UPLEFTFIRE",
            JAXAtariAction.DOWNRIGHTFIRE: "DOWNRIGHTFIRE",
            JAXAtariAction.DOWNLEFTFIRE: "DOWNLEFTFIRE"
        }

        return mapping.get(action_idx, "WAIT")


if __name__ == "__main__":
    checkpoint_path = "/home/remunds/projects/JAXAtari/runs/kangaroo__ppo_jaxatari_scan__42__1765889637/ppo_jaxatari_scan_9765.cleanrl_model"
    import jaxatari
    env = jaxatari.make("kangaroo")
    env = AtariWrapper(
            env,
            episodic_life=False, # only active during training 
            clip_reward=False, # only active during training
            max_episode_length=108000,
            frame_stack_size=4,
            max_pooling=True,
            frame_skip=4,
            noop_reset=30,
            sticky_actions=False, # seems to be default in envpool
            first_fire=True,
            #full_action_space=False # TODO: this is missing in jaxatari, although default is reduced action space
    )
    pixel_env = PixelObsWrapper(env)
    agent = PixelRLAgent(env=env, checkpoint_path=checkpoint_path)

    obs, state = pixel_env.reset(jax.random.PRNGKey(0))
    action_str = agent.predict(np.array(obs))