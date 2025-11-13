import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxatari.environment import JAXAtariAction as Action
from upndown import JaxUpNDown, UpNDownConstants  # <-- your game file

def visualize_frame(frame: jnp.ndarray):
    """Render an RGB frame using matplotlib."""
    plt.imshow(frame.astype(jnp.uint8))
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.05)
    plt.clf()


def main():
    # Initialize environment
    env = JaxUpNDown(UpNDownConstants())

    # Reset environment
    obs, state = env.reset()
    print("Initial observation:", obs)

    # Display initial render
    frame = env.render(state)
    visualize_frame(frame)

    # Create a random key for sampling actions
    key = jax.random.PRNGKey(0)

    # Run for 50 steps
    for step in range(50):
        key, subkey = jax.random.split(key)
        # Choose a random action from action space
        action = jax.random.choice(subkey, jnp.arange(len(env.action_set)))

        obs, state, reward, done, info = env.step(state, action)

        # Render and display
        frame = env.render(state)
        visualize_frame(frame)

        print(f"Step {step}: action={env.action_set[int(action)]}, reward={reward}, done={done}")

        if bool(done):
            print("Game over — resetting environment.")
            obs, state = env.reset()

    plt.close()

if __name__ == "__main__":
    main()
