# Taken entirely from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py
# Adapted to JaxAtari

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from turtle import end_fill
from typing import Sequence, NamedTuple

import flax
import flax.linen as nn
import gym
from gymnasium.wrappers import NormalizeObservation
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
import jaxatari
from jaxatari.wrappers import NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper, AtariWrapper, LogWrapper, FlattenObservationWrapper
from jaxatari import spaces
from ppo_jaxatari_vmap_eval import evaluate

from rtpt import RTPT

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    eval_during_train: bool = False # If this is active, compile and run times will increase!
    """whether to evaluate the agent periodically during training"""
    eval_every: int = 1000 #  1000 -> all 1M steps
    """how often to evaluate the agent during training (in num. of iterations)"""
    pixel_based: bool = True # If False -> Object-centric observations
    """whether the environment should use pixel-based observations"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "pong"
    """the id of the environment"""
    mods: tuple[str] = ('lazy_enemy', 'random_enemy')
    """modifications applied during training"""
    eval_mods: tuple[str] = ()
    """modifications to use for evaluation (if empty, fall back to `mods`)"""
    total_timesteps: int = 10_000_000 # so with frameskip=4 -> 40M frames (?)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False # True in original paper, but envpool impl. used False (don't think this setting does something here, actually.)
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, seed, num_envs, mods=[], pixel_based=True, eval=False):
    def thunk():
        # For training (eval=False), avoid applying multiple potentially conflicting
        # mods at once. In that case, fall back to the base environment.
        # For evaluation (eval=True), we trust the caller to pass either a single
        # mod or an explicit list; this is used in the per-mod video generation.
        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []

        # Normalize to None or list for jaxatari.make
        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)
        env = AtariWrapper(
                env,
                episodic_life=not eval, # only active during training 
                clip_reward=not eval, # only active during training
                max_episode_length=108000,
                frame_stack_size=4,
                max_pooling=True,
                frame_skip=4,
                noop_reset=30,
                sticky_actions=False, # seems to be default in envpool
                first_fire=True,
                #full_action_space=False # TODO: this is missing in jaxatari, although default is reduced action space
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=True
            )
        else:
            env = FlattenObservationWrapper(NormalizeObservationWrapper(ObjectCentricWrapper(env)))
        env = LogWrapper(env)
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.is_vector_env = True
        #TODO: Do we need actionset_wrapper? (like the videopinball guys did)
        return env
    return thunk


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
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

class MLP_Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        # 1. Hidden Layer
        x = nn.Dense(
            461,  # Hidden size H=461
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        # 2. Output Layer (matches the last layer of the CNN)
        x = nn.Dense(
            512,  # Output size
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)  # The CNN's last layer also has a ReLU
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


class AgentParams(NamedTuple):
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}_{'oc' if not args.pixel_based else 'pixel'}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, obs_sample_key1, obs_sample_key2, obs_sample_key3 = jax.random.split(key, 4)

    # env setup
    env = make_env(args.env_id, args.seed, args.num_envs, list(args.mods), args.pixel_based)()

    renderer = None
    if args.capture_video:
        renderer = env.renderer
   
    # vmap and squeeze observations in order to get (B, F, H, W, 1) -> (B, F, H, W),
    # where F is the frame stack which becomes the channel for the convolutions
    @jax.jit
    def wrapped_reset(key):
        obs, state = jax.vmap(env.reset)(key)
        return obs.squeeze(), state
    
    @jax.jit
    def wrapped_step(state, action):
        next_obs, state, reward, next_done, info = jax.vmap(env.step)(state, action)
        return next_obs.squeeze(), state, reward, next_done, info

    vmap_reset = wrapped_reset
    vmap_step = wrapped_step
    
    assert isinstance(env.action_space(), spaces.Discrete), "only discrete action space is supported"
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    network = Network() if args.pixel_based else MLP_Network()
    actor = Actor(action_dim=env.action_space().n)
    critic = Critic()
    # network_params = network.init(network_key, env.observation_space().sample(obs_sample_key1).squeeze()[None, ...])
    # Init shape is (1,4,84,84) (which will be transposed inside the network to (1,84,84,4))
    network_params = network.init(network_key, env.observation_space().sample(obs_sample_key1).squeeze()[None, ...])
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params=network_params,
            actor_params=actor.init(actor_key, network.apply(network_params, np.array([env.observation_space().sample(obs_sample_key2).squeeze()]))),
            critic_params=critic.init(critic_key, network.apply(network_params, np.array([env.observation_space().sample(obs_sample_key3).squeeze()]))),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        hidden = network.apply(agent_state.params.network_params, next_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(
            agent_state.params.critic_params, network.apply(agent_state.params.network_params, next_obs)
        ).squeeze()

        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree.map(flatten, storage)
            shuffled_storage = jax.tree.map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key
    
    def eval_and_vid(iteration, global_step):
        model_path = f"runs/{run_name}/{args.exp_name}_{iteration}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params.network_params,
                            agent_state.params.actor_params,
                            agent_state.params.critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")

        # Run base-environment evaluation; per-mod videos are handled separately
        # in generate_final_video(), where mods are applied one at a time.
        episodic_returns, env_states = evaluate(
            model_path,
            partial(
                make_env,
                mods=[],
                pixel_based=args.pixel_based,
                eval=True,
            ),
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic) if args.pixel_based else (MLP_Network, Actor, Critic)
        )
        writer.add_scalar("eval/episodic_return", np.mean(jax.device_get(episodic_returns)), global_step) 

        if args.capture_video and renderer is not None: 
            # Mirror the pqn_agent final video behavior: log a video for the
            # current eval environment under a consistent wandb key.
            frames = jax.vmap(renderer.render)(env_states)
            frames = jnp.transpose(frames, (0, 3, 1, 2))
            video = wandb.Video(np.array(frames), fps=30, format="mp4")
            wandb.log(
                {
                    f"final_video_seed{args.seed}_eval": video,
                    f"final_return_seed{args.seed}_eval": np.mean(jax.device_get(episodic_returns)),
                },
                step=global_step,
            )
            print(f"Video (eval) logged to wandb with {frames.shape[0]} frames.")

    def _generate_single_final_video(mods_config, video_label, video_index=0):
        """Generate a single video for the given mod configuration and log it to wandb.

        This mirrors pqn_agent._generate_single_final_video so that rendered videos
        use the original color, full-resolution frames from the base environment.
        """
        if not args.capture_video:
            return None

        # Base env with mods (no training wrappers yet).
        env = jaxatari.make(args.env_id.lower(), mods=mods_config)
        renderer_local = env.renderer

        # Apply wrappers just for the agent's observations, like in pqn_agent.
        env = AtariWrapper(
            env,
            episodic_life=True,
            frame_skip=4,
            frame_stack_size=4,
            sticky_actions=True,
            max_pooling=True,
            clip_reward=False,
            noop_reset=30,
            max_episode_length=18000,
        )
        if not args.pixel_based:
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
        else:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=True,
            )
        env = NormalizeObservationWrapper(env)
        env = LogWrapper(env)

        # Reset environment
        rng = jax.random.PRNGKey(args.seed + video_index * 10000)
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        obs = obs.squeeze()  # (F, H, W)

        frames = []
        total_reward = 0.0
        max_steps = 5000

        for step in range(max_steps):
            # PPO network expects (B, F, H, W)
            policy_obs = obs[None, ...]

            hidden = network.apply(agent_state.params.network_params, policy_obs)
            logits = actor.apply(agent_state.params.actor_params, hidden)
            action = jnp.argmax(logits, axis=-1)[0]

            rng, step_rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(env_state, action)
            obs = obs.squeeze()
            total_reward += float(reward)

            # Render frame from the underlying base Atari state, using the original renderer.
            state_for_render = env_state
            while hasattr(state_for_render, "atari_state"):
                state_for_render = state_for_render.atari_state
            if hasattr(state_for_render, "env_state"):
                state_for_render = state_for_render.env_state

            frame = renderer_local.render(state_for_render)
            frames.append(np.array(frame, dtype=np.uint8))

            if bool(done):
                break

        print(f"Final video ({video_label}): {len(frames)} frames, total reward: {total_reward:.1f}")

        if len(frames) > 0:
            frames = np.stack(frames, axis=0)
            # (N, H, W, 3) -> (N, 3, H, W) for wandb
            frames = np.transpose(frames, (0, 3, 1, 2))
            video = wandb.Video(frames, fps=30, format="mp4")
            wandb.log(
                {
                    f"final_video_seed{args.seed}_{video_label}": video,
                    f"final_return_seed{args.seed}_{video_label}": total_reward,
                }
            )
            print(f"Video '{video_label}' logged to wandb.")

        return total_reward

    def generate_final_video():
        """Generate videos of the trained agent: one for train env, one per eval mod.

        This mirrors pqn_agent.generate_final_video: first a video on the
        training environment (no mods), then one per entry in eval_mods (or mods).
        """
        if not args.capture_video:
            return

        print(f"Generating final videos for seed {args.seed}...")

        video_configs = []
        # Always add train env (no mods)
        video_configs.append(([], "train"))

        # Prefer eval_mods, fall back to mods
        eval_mods = args.eval_mods if len(args.eval_mods) > 0 else args.mods
        if len(eval_mods) > 0:
            mods_list = list(eval_mods)
            for mod in mods_list:
                mods_config = [mod] if not isinstance(mod, (list, tuple)) else list(mod)
                mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_config)
                video_configs.append((mods_config, mod_label))

        for video_index, (mods_config, video_label) in enumerate(video_configs):
            _generate_single_final_video(mods_config, video_label, video_index)

    # TRY NOT TO MODIFY: start the game
    key, reset_key = jax.random.split(key)
    global_step = 0
    start_time = time.time()
    next_obs, env_state = vmap_reset(jax.random.split(reset_key, args.num_envs))
    next_done = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)

    # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py
    def step_once(carry, step, env_step_fn):
        agent_state, obs, done, key, env_state = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        next_obs, env_state, reward, next_done, _ = env_step_fn(env_state, action)
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return ((agent_state, next_obs, next_done, key, env_state), storage)

    def rollout(agent_state, next_obs, next_done, key, env_state, step_once_fn, max_steps):
        (agent_state, next_obs, next_done, key, env_state), storage = jax.lax.scan(
            step_once_fn, (agent_state, next_obs, next_done, key, env_state), (), max_steps
        )
        return agent_state, next_obs, next_done, storage, key, env_state

    rollout = partial(rollout, step_once_fn=partial(step_once, env_step_fn=vmap_step), max_steps=args.num_steps)

    rtpt = RTPT(name_initials='RE', experiment_name='PPO_JAXAtari', max_iterations=args.num_iterations)
    rtpt.start()
    start_time = time.time()
    compile_time = None
    for iteration in range(1, args.num_iterations + 1):
        rtpt.step()
        if args.eval_during_train and iteration > 0 and iteration % args.eval_every == 0:
           eval_and_vid(iteration, global_step) 

        iteration_time_start = time.time()
        agent_state, next_obs, next_done, storage, key, env_state = rollout(
            agent_state, next_obs, next_done, key, env_state
        )
        global_step += args.num_steps * args.num_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        if compile_time is None:
            compile_time = time.time()
            print(f"Compile + first iteration time: {compile_time - start_time:.2f} seconds.")
        avg_episodic_return = np.mean(jax.device_get(env_state.returned_episode_returns))
        # print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(env_state.returned_episode_lengths)), global_step
        )
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - iteration_time_start)), global_step
        )
        writer.add_scalar(
            "charts/time", time.time() - start_time, global_step
        )
    end_time = time.time()
    print("Training done.")
    if compile_time is not None:
        print(f"Run time after first iteration: {end_time - compile_time:.2f} seconds.")
    print(f"Total train time: {end_time - start_time:.2f} seconds / {(end_time - start_time)/60:.2f} minutes.")

    if args.save_model:
        eval_and_vid(iteration, global_step)
        generate_final_video()

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    writer.close()
