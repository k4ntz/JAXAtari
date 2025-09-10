import jax
import jax.numpy as jnp
import pygame

from jaxatari.games.jax_enduro import EnduroRenderer
import jaxatari.rendering.jax_rendering_utils as aj

from jax_enduro import JaxEnduro, EnduroGameState

from jaxatari.environment import JAXAtariAction as Action

DIRECTION_LABELS = {
    -1: "Left",
    0: "Straight",
    1: "Right"
}


def render_debug_overlay(screen, state: EnduroGameState, font, game_config):
    """Render debug information as pygame text overlay"""
    track_direction_starts_at = state.whole_track[:, 1]
    track_segment_index = int(jnp.searchsorted(track_direction_starts_at, state.distance, side='right') - 1)
    track_direction = state.whole_track[track_segment_index, 0]
    debug_info = [
        f"Speed: {float(state.player_speed):.2f}",  # Convert JAX arrays to Python floats
        f"Player X (abs): {state.player_x_abs_position}",
        f"Player Y (abs): {state.player_y_abs_position}",
        # f"Distance: {state.distance}",
        f"Level: {state.level}",
        f"Time: {state.total_time_elapsed}",
        f"Left Mountain x: {state.mountain_left_x}",
        f"Opponent Index: {state.opponent_index}",
        # f"Opponent window: {state.opponent_window}",
        # f"Opponents: {state.visible_opponent_positions}",
        # f"Cars overtaken: {state.cars_overtaken}",
        f"Opponent Collision: {state.is_collision}",
        # f"Cooldown Drift direction: {state.cooldown_drift_direction}"
        f"Weather: {state.weather_index}",
        # f"Track direction: {DIRECTION_LABELS.get(int(track_direction))} ({track_direction})",
        # f"Track top X: {state.track_top_x}",
        # f"Top X Offset: {state.track_top_x_curve_offset}",
    ]

    # Semi-transparent background for better readability
    overlay = pygame.Surface((250, len(debug_info) * 25 + 20))  # Made slightly larger
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (10, 10))

    # Render each debug line
    for i, text in enumerate(debug_info):
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (15, 15 + i * 25))


def play_enduro(debug_mode=True):
    """
    Plays the game with a renderer

    Args:
        debug_mode: If True, shows debug overlay. Set to False for production/optimized runs.
    """
    pygame.init()
    # Initialize game and renderer
    game = JaxEnduro()
    renderer = EnduroRenderer()
    scaling = 4

    screen = pygame.display.set_mode((160 * scaling, 210 * scaling))
    pygame.display.set_caption("Enduro" + (" - DEBUG MODE" if debug_mode else ""))

    font = pygame.font.Font(None, 20)  # You can adjust size as needed
    small_font = pygame.font.Font(None, 16)

    # Always JIT compile the core game functions
    # This ensures JIT compatibility is tested even during debugging
    step_fn = jax.jit(game.step)
    render_fn = jax.jit(renderer.render)
    reset_fn = jax.jit(game.reset)

    init_obs, state = reset_fn()

    # Setup game loop
    clock = pygame.time.Clock()
    running = True
    done = False

    print(f"Starting game in {'DEBUG' if debug_mode else 'PRODUCTION'} mode")
    print("Core game functions are JIT compiled for performance and compatibility testing")
    if debug_mode:
        print("Press 'D' to toggle debug overlay")

    show_debug = debug_mode  # Can be toggled during gameplay

    while running and not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d and debug_mode:
                    show_debug = not show_debug
                    print(f"Debug overlay: {'ON' if show_debug else 'OFF'}")

        # Handle input
        keys = pygame.key.get_pressed()
        # allow arrows and wsad
        if (keys[pygame.K_a] or keys[pygame.K_LEFT]) and keys[pygame.K_SPACE]:
            action = Action.LEFTFIRE
        elif (keys[pygame.K_d] or keys[pygame.K_RIGHT]) and keys[pygame.K_SPACE]:
            action = Action.RIGHTFIRE
        elif (keys[pygame.K_s] or keys[pygame.K_DOWN]) and keys[pygame.K_SPACE]:
            action = Action.DOWNFIRE
        elif keys[pygame.K_SPACE]:
            action = Action.FIRE
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = Action.LEFT
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = Action.RIGHT
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = Action.DOWN
        else:
            action = Action.NOOP

        # Update game state
        obs, state, reward, done, info = step_fn(state, action)

        # Render game frame
        frame = render_fn(state)
        aj.update_pygame(screen, frame, scaling, 160, 210)

        # Add debug overlay if enabled
        if debug_mode and show_debug:
            render_debug_overlay(screen, state, font, renderer.config)

            # Add controls help in corner
            help_text = small_font.render("Press 'D' to toggle debug", True, (200, 200, 200))
            screen.blit(help_text, (screen.get_width() - 180, screen.get_height() - 20))

        pygame.display.flip()

        # Cap at 60 FPS (or 30 for debug mode to make it easier to read)
        clock.tick(30 if debug_mode else 60)

    # If game over, wait before closing
    if done:
        pygame.time.wait(2000)


if __name__ == '__main__':
    # For debugging and development
    play_enduro(debug_mode=True)
