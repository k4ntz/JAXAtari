import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import constant, he_normal, variance_scaling
import flax.core
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax

# inspired by https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation_fn = nn.relu
            hidden_kernel_init_fn = he_normal()
            actor_logits_kernel_init_fn = variance_scaling(0.01, 'fan_in', 'normal')
            critic_value_kernel_init_fn = variance_scaling(1.0, 'fan_in', 'normal')
        elif self.activation == "tanh":
            activation_fn = nn.tanh
            from flax.linen.initializers import xavier_uniform
            hidden_kernel_init_fn = xavier_uniform()
            actor_logits_kernel_init_fn = variance_scaling(0.01, 'fan_in', 'normal')
            critic_value_kernel_init_fn = variance_scaling(1.0, 'fan_in', 'normal')
        else:
            activation_fn = nn.relu
            hidden_kernel_init_fn = he_normal()
            actor_logits_kernel_init_fn = variance_scaling(0.01, 'fan_in', 'normal')
            critic_value_kernel_init_fn = variance_scaling(1.0, 'fan_in', 'normal')

        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

        # Actor Stream
        actor_hidden = nn.Dense(
            128, kernel_init=hidden_kernel_init_fn, bias_init=constant(0.0)
        )(x)
        actor_hidden = activation_fn(actor_hidden)
        actor_hidden = nn.Dense(
            128, kernel_init=hidden_kernel_init_fn, bias_init=constant(0.0)
        )(actor_hidden)
        actor_hidden = activation_fn(actor_hidden)
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=actor_logits_kernel_init_fn, bias_init=constant(0.0)
        )(actor_hidden)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic Stream
        critic_hidden = nn.Dense(
            128, kernel_init=hidden_kernel_init_fn, bias_init=constant(0.0)
        )(x)
        critic_hidden = activation_fn(critic_hidden)
        critic_hidden = nn.Dense(
            128, kernel_init=hidden_kernel_init_fn, bias_init=constant(0.0)
        )(critic_hidden)
        critic_hidden = activation_fn(critic_hidden)
        critic_value = nn.Dense(1, kernel_init=critic_value_kernel_init_fn, bias_init=constant(0.0))(
            critic_hidden
        )
        return pi, jnp.squeeze(critic_value, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray 
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

def create_ppo_train_state(
    rng_key: jax.random.PRNGKey, 
    config: Dict, 
    obs_shape_flat: tuple, 
    action_dim: int
) -> TrainState:
    network = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
    init_x = jnp.zeros((1,) + obs_shape_flat) 
    network_params = network.init(rng_key, init_x)['params']

    def linear_schedule(count):
        num_total_optimizer_steps = config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
        frac = 1.0 - (count / num_total_optimizer_steps)
        frac = jnp.maximum(0.0, frac)
        return config["LR"] * frac

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    
    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

def ppo_loss_fn(
    params: flax.core.FrozenDict,
    apply_fn: callable,
    traj_batch_obs: jnp.ndarray,
    traj_batch_action: jnp.ndarray,
    traj_batch_log_prob_old: jnp.ndarray,
    traj_batch_value_old: jnp.ndarray,
    advantages: jnp.ndarray,
    targets: jnp.ndarray, 
    config: Dict
):
    pi, value_new = apply_fn({'params': params}, traj_batch_obs)
    log_prob_new = pi.log_prob(traj_batch_action)

    clip_vf_eps = config.get("CLIP_VF_EPS", config["CLIP_EPS"])
    value_pred_clipped = traj_batch_value_old + (
        value_new - traj_batch_value_old
    ).clip(-clip_vf_eps, clip_vf_eps)
    value_losses = jnp.square(value_new - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob_new - traj_batch_log_prob_old)
    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss_actor1 = ratio * norm_advantages
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * norm_advantages
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
    
    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    
    aux_info = {
        "pg_loss": loss_actor,
        "vf_loss": value_loss,
        "entropy": entropy,
        "approx_kl": (traj_batch_log_prob_old - log_prob_new).mean(),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])
    }
    return total_loss, aux_info

def ppo_update_minibatch(
    train_state: TrainState,
    obs_batch: jnp.ndarray,
    actions_batch: jnp.ndarray,
    log_probs_old_batch: jnp.ndarray,
    values_old_batch: jnp.ndarray,
    advantages_batch: jnp.ndarray,
    returns_batch: jnp.ndarray, 
    config: Dict
):
    grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)
    (loss, aux_info), grads = grad_fn(
        train_state.params,
        train_state.apply_fn,
        obs_batch,
        actions_batch,
        log_probs_old_batch,
        values_old_batch,
        advantages_batch,
        returns_batch,
        config
    )
    new_train_state = train_state.apply_gradients(grads=grads)
    return new_train_state, loss, aux_info 