import os
from typing import Dict, Union
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict

def video_callback(states, dones, step, renderer, mod=False):
    """
    Starting a new thread to render video so that it doesn't block training.
    """
    # video_thread = threading.Thread(target=video_renderer, args=(states, dones, step, renderer, mod))
    # video_thread.start()
    video_renderer(states, dones, step, renderer, mod)

def video_renderer(states, dones, step, renderer, mod):
    print("Rendering video...")
    video_folder = f"{wandb.run.dir}/media/videos/"
    os.makedirs(video_folder, exist_ok=True)

    while hasattr(states, 'atari_state'):
        states = states.atari_state

    if hasattr(states, 'env_state'):
        states = states.env_state

    # num_states is where the first done is True
    num_states = jnp.argmax(dones)
    # or len of the first array of the states pytree
    if num_states == 0:
        num_states = len(states[-1])

    # select every 4th frame (and only the first num_states)
    states_reduced = jax.tree_util.tree_map(lambda x: x[:num_states], states)
    rasters = jax.vmap(renderer.render)(states_reduced)
    frames = np.array(rasters, dtype=np.uint8)
    # shape currently is (N, W, H, 3)
    # but should be (N, 3, W, H)
    frames = np.transpose(frames, (0, 3, 1, 2))

    fps = 30 
    video = wandb.Video(frames, fps=fps, format="mp4")
    name = f"video_{step}"
    if mod:
        name = f"video_{step}_mod"
    wandb.log({name: video})
    print("Video done.")

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")
