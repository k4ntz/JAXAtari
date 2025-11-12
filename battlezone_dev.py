# pip install "gymnasium[atari,accept-rom-license]" pygame ale-py
import pygame
import numpy as np
import time
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from typing import Dict, Tuple

# ------------------------------- Config ------------------------------------ #
ENV_ID = "ALE/BattleZone-v5"
FRAMESKIP = 1                  # keep 1 for ~60Hz updates
STICKY = 0.0                   # no sticky actions
GAME_SCALE = 3                 # scales the game surface (native ~160x210)
GRID_COLS = 16
GRID_ROWS = 8
CELL_W, CELL_H = 56, 50        # RAM cell size
# CELL_W, CELL_H = 20, 20        # RAM cell size
PADDING = 8
FONT_SIZE = 16
SMALL_FONT_SIZE = 14
NAMED_LINE_H = 18
FPS_CAP = 60

# Screenshot format: set to 'bmp' to save a bitmap image (default),
# or 'npy' to save a 3D numpy array of RGB values (height, width, 3).
SCREENSHOT_FORMAT = "bmp"  # one of: 'bmp', 'png', 'npy'

# Change-highlight parameters
STABLE_FRAMES_FOR_TRIGGER = 30
FLASH_FRAMES = 30
# Frames of steady increase/decrease required to show green trend border
TREND_FRAMES = 60


# Optional: fill with your known addresses. Toggle panel with 'N'.
NAMED: Dict[str, int] = {
    # example: "player_x": 0x10,
    # added addresses CA..CE
    # "CA": 0xCA,
    # "CB": 0xCB,
    # "CC": 0xCC,
    # "CD": 0xCD,
    # "CE": 0xCE,
    #
    "blue_tank_facing_direction": 0xAE,  # 46 + 0x80 ## probably actually the background scroll offset
    "blue_tank_size_y": 0xAF,            # 47
    "blue_tank_x": 0xB0,                 # 48
    "blue_tank2_facing_direction": 0xB4,# 52
    "blue_tank2_size_y": 0xB5,          # 53
    "blue_tank2_x": 0xB6,               # 54
    "num_lives": 0xBA,                  # 58
    "missile_y": 0xE9,                  # 105
    "compass_needles_angle": 0xD4,      # 84
    "angle_of_tank": 0x84,              # 4
    "left_tread_position": 0xBB,        # 59
    "right_tread_position": 0xBC,       # 60
    "crosshairs_color": 0xEC,  # 108 -> 0xEC; merged with previous
    "score": 0x9D,   
}
# Addresses to hide from the heatmap (use displayed addresses 0x80..0xFF)
# Example: BLACKLIST = {0x80, 0xCA}
BLACKLIST = {0xDD, 0xD7, 0xC0, 0xA6, 0xBF, 0xFC, 0xFD, 0xFE, 0xFF, 0xDE, 0xDF, 0xA4, 0xA5, 0xA7, 0xA8, 0xD9, 0xDA, 0xD8, 0xD6}

# Optional: custom names for heatmap boxes keyed by displayed address (0x80..0xFF).
# If an address is present here its value will be displayed instead of the hex address.
# Example: HEATMAP_NAMES = {0xCA: "player_x", 0xCB: "player_y"}
HEATMAP_NAMES: Dict[int, str] = {
    # from provided list, add 0x80 to each integer key
    0xAE: "blue_tank_facing_direction",  # 46 + 0x80
    0xAF: "blue_tank_size_y",            # 47
    0xB0: "blue_tank_x",                 # 48
    0xB4: "blue_tank2_facing_direction",# 52
    0xB5: "blue_tank2_size_y",          # 53
    0xB6: "blue_tank2_x",               # 54
    0xBA: "num_lives",                  # 58
    0xE9: "missile_y",                  # 105
    0xD4: "compass_needles_angle",      # 84
    0x84: "angle_of_tank",              # 4
    0xBB: "left_tread_position",        # 59
    0xBC: "right_tread_position",       # 60
    0xEC: "crosshairs_color",  # 108 -> 0xEC; merged with previous
    0x9D: "score",                      # 29
}

# Controls:
#  - Arrows = movement. Space = FIRE. (We auto-map to ALE combos like "UPFIRE" if present.)
#  - P = pause/resume stepping
#  - R = reset
#  - N = toggle named-address panel
#  - Q / ESC = quit
# --------------------------------------------------------------------------- #

def color_map(v: int) -> pygame.Color:
    v = int(np.clip(v, 0, 255))
    a = (v / 255.0) ** 0.8
    r = int(255 * min(1.0, a * 2.0))
    g = int(255 * a)
    b = int(255 * (0.3 + 0.7 * (1.0 - a)))
    return pygame.Color(r, g, b)

def build_action_mapper(env) -> Tuple[dict, int]:
    meanings = env.unwrapped.get_action_meanings()
    meaning_to_id = {m: i for i, m in enumerate(meanings)}
    noop_id = meaning_to_id.get("NOOP", 0)
    return meaning_to_id, noop_id

def resolve_action(keys: set, meaning_to_id: dict, noop_id: int) -> int:
    # Choose one direction deterministically if multiple are held
    base = None
    priority = ["UP", "DOWN", "LEFT", "RIGHT"]
    for p in priority:
        if p in keys:
            base = p
            break
    fire = "FIRE" in keys
    candidates = []
    if base and fire: candidates.append(base + "FIRE")
    if base: candidates.append(base)
    if fire: candidates.append("FIRE")
    candidates.append("NOOP")
    for name in candidates:
        if name in meaning_to_id:
            return meaning_to_id[name]
    return noop_id


def main():
    # ---- Gym env ----
    env = gym.make(
        ENV_ID,
        obs_type="rgb",   # we want the game frame for display
        frameskip=FRAMESKIP,
        repeat_action_probability=STICKY,
        render_mode="rgb_array",
    )
    obs, info = env.reset(seed=0)

    ale = env.unwrapped.ale
    meaning_to_id, noop_id = build_action_mapper(env)

    # ---- Pygame ----
    pygame.init()
    font = pygame.font.SysFont("consolas,menlo,monospace", FONT_SIZE)
    sfont = pygame.font.SysFont("consolas,menlo,monospace", SMALL_FONT_SIZE)

    show_named = len(NAMED) > 0
    # include a header row for the table when showing named addresses
    named_panel_h = ((len(NAMED) + 1) * NAMED_LINE_H + PADDING) if show_named else 0

    # Layout
    game_h, game_w = obs.shape[0], obs.shape[1]
    game_surface_size = (game_w * GAME_SCALE, game_h * GAME_SCALE)
    grid_w = GRID_COLS * CELL_W + (GRID_COLS + 1) * PADDING
    grid_h = GRID_ROWS * CELL_H + (GRID_ROWS + 1) * PADDING
    W = game_surface_size[0] + PADDING + grid_w + PADDING
    H = max(game_surface_size[1], grid_h + named_panel_h) + PADDING
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"{ENV_ID} — Play + RAM Viewer")

    clock = pygame.time.Clock()
    paused = False
    pressed = set()  # {"UP","DOWN","LEFT","RIGHT","FIRE"}

    # Precompute grid origin so event handling can reference it before drawing
    grid_x = PADDING + game_surface_size[0] + PADDING
    grid_y = PADDING

    # ---- RAM change tracking state ----
    last_vals = np.full(128, -1, dtype=int)          # -1 means "uninitialized"
    stable_counts = np.zeros(128, dtype=int)         # how long current value held
    flash_timers = np.zeros(128, dtype=int)          # frames remaining to show red border
    # trend trackers: consecutive increment / decrement counts
    inc_counts = np.zeros(128, dtype=int)
    dec_counts = np.zeros(128, dtype=int)

    # --- Editing UI state (click-to-edit) ---
    # editing_idx: index of the RAM cell currently being edited (or None)
    editing_idx = None
    edit_buffer = ""
    # last time the edit cursor blinked (ms)
    edit_cursor_ms = 0

    running = True

    # Print the available key commands once on startup
    print("Controls:")
    print("  Arrows: movement (mapped to ALE action meanings)")
    print("  Space: FIRE")
    print("  P: pause/resume stepping")
    print("  R: reset environment")
    print("  S: save a screenshot of the game frame")
    print("  N: toggle named-address panel (if any names configured)")
    print("  Q or ESC: quit")
    print("  Mouse left-click on a RAM cell: edit its value (enter hex). Enter to commit, Esc to cancel.")
    print(f"  Screenshot format: {SCREENSHOT_FORMAT} (one game px -> one saved px)")
    while running:
        # ---- Input ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                # If we're editing a RAM cell, capture hex input here
                if editing_idx is not None:
                    # Enter commits, Esc cancels, Backspace edits
                    if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        try:
                            s = edit_buffer.strip().lower()
                            if s.startswith("0x"):
                                s = s[2:]
                            if s == "":
                                val = 0
                            else:
                                val = int(s, 16)
                            val = int(np.clip(val, 0, 255))
                            ale.setRAM(editing_idx, val)
                            last_vals[editing_idx] = val
                            stable_counts[editing_idx] = 1
                            inc_counts[editing_idx] = 0
                            dec_counts[editing_idx] = 0
                        except Exception:
                            pass
                        editing_idx = None
                        edit_buffer = ""
                    elif k == pygame.K_ESCAPE:
                        editing_idx = None
                        edit_buffer = ""
                    elif k == pygame.K_BACKSPACE:
                        edit_buffer = edit_buffer[:-1]
                    else:
                        ch = event.unicode
                        if ch and ch.lower() in "0123456789abcdefx":
                            edit_buffer += ch
                            edit_cursor_ms = pygame.time.get_ticks()
                    continue

                # not editing -> handle gameplay/controls
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_p:
                    paused = not paused
                elif k == pygame.K_s:
                    ts = int(time.time())
                    fmt = SCREENSHOT_FORMAT.lower()
                    # Save a 1:1 screenshot (no scaling) so one game pixel == one saved pixel.
                    if fmt == "npy":
                        # obs is already HxWx3 (RGB); save directly
                        arr = np.array(obs, copy=True)
                        fname = f"screenshot_{ts}.npy"
                        try:
                            np.save(fname, arr)
                            print(f"Saved screenshot (npy): {fname} shape={arr.shape}")
                        except Exception as e:
                            print(f"Failed to save npy screenshot: {e}")
                    else:
                        # Create a surface from obs without scaling (1:1)
                        try:
                            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                            ext = fmt if fmt in ("bmp", "png", "jpg", "jpeg") else "bmp"
                            fname = f"screenshot_{ts}.{ext}"
                            pygame.image.save(surf, fname)
                            print(f"Saved screenshot: {fname}")
                        except Exception as e:
                            print(f"Failed to save image screenshot: {e}")
                elif k == pygame.K_r:
                    obs, info = env.reset()
                elif k == pygame.K_n:
                    show_named = not show_named
                elif k == pygame.K_SPACE:
                    pressed.add("FIRE")
                elif k == pygame.K_UP:
                    pressed.add("UP")
                elif k == pygame.K_DOWN:
                    pressed.add("DOWN")
                elif k == pygame.K_LEFT:
                    pressed.add("LEFT")
                elif k == pygame.K_RIGHT:
                    pressed.add("RIGHT")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # left-click to start editing a cell
                if event.button == 1:
                    mx, my = event.pos
                    # check if click inside RAM grid
                    if (mx >= grid_x and mx < grid_x + grid_w and
                            my >= grid_y and my < grid_y + grid_h):
                        rx = mx - (grid_x + PADDING)
                        ry = my - (grid_y + PADDING)
                        if rx >= 0 and ry >= 0:
                            c = rx // (CELL_W + PADDING)
                            r = ry // (CELL_H + PADDING)
                            if 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS:
                                idx = int(r * GRID_COLS + c)
                                disp_addr = idx + 0x80
                                if disp_addr in BLACKLIST:
                                    editing_idx = None
                                    edit_buffer = ""
                                else:
                                    editing_idx = idx
                                    edit_buffer = ""
                                    edit_cursor_ms = pygame.time.get_ticks()
                                continue
                    editing_idx = None
                    edit_buffer = ""
                    continue
            elif event.type == pygame.KEYUP:
                k = event.key
                if k == pygame.K_SPACE:
                    pressed.discard("FIRE")
                elif k == pygame.K_UP:
                    pressed.discard("UP")
                elif k == pygame.K_DOWN:
                    pressed.discard("DOWN")
                elif k == pygame.K_LEFT:
                    pressed.discard("LEFT")
                elif k == pygame.K_RIGHT:
                    pressed.discard("RIGHT")

        # ---- Step env ----
        if not paused:
            action = resolve_action(pressed, meaning_to_id, noop_id)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                last_vals[:] = -1
                stable_counts[:] = 0
                flash_timers[:] = 0
                inc_counts[:] = 0
                dec_counts[:] = 0

        # ---- Pull RAM & update change detectors ----
        ram = np.zeros(128, dtype=np.uint8)
        ale.getRAM(ram)

        # Update stable counts + trigger flash on qualified change
        for i in range(128):
            v = int(ram[i])
            prev = last_vals[i]
            if prev == -1:
                # first time seeing a value — initialize
                last_vals[i] = v
                stable_counts[i] = 1
                # initialize trend trackers
                inc_counts[i] = 0
                dec_counts[i] = 0
                continue
            if v == prev:
                stable_counts[i] += 1
                # no change -> reset trend counts
                inc_counts[i] = 0
                dec_counts[i] = 0
            else:
                # value changed
                if stable_counts[i] >= STABLE_FRAMES_FOR_TRIGGER:
                    flash_timers[i] = FLASH_FRAMES
                # update trend counters
                if v > prev:
                    inc_counts[i] += 1
                    dec_counts[i] = 0
                elif v < prev:
                    dec_counts[i] += 1
                    inc_counts[i] = 0

                last_vals[i] = v
                stable_counts[i] = 1
        # Decay flash timers
        flash_timers[flash_timers > 0] -= 1

        # ---- Draw ----
        screen.fill((12, 12, 12))

        # Game frame
        frame = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        frame = pygame.transform.scale(frame, game_surface_size)
        screen.blit(frame, (PADDING, PADDING))

    # (Save-screen button removed; use 'S' keyboard shortcut to save game frame)

        # RAM grid
        grid_x = PADDING + game_surface_size[0] + PADDING
        grid_y = PADDING

        for idx in range(128):
            r = idx // GRID_COLS
            c = idx % GRID_COLS
            x = grid_x + PADDING + c * (CELL_W + PADDING)
            y = grid_y + PADDING + r * (CELL_H + PADDING)
            disp_addr = idx + 0x80
            rect = pygame.Rect(x, y, CELL_W, CELL_H)

            # If address is blacklisted, draw a muted box and label it "BL"
            if disp_addr in BLACKLIST:
                pygame.draw.rect(screen, (20, 20, 20), rect, border_radius=6)
                pygame.draw.rect(screen, (35, 35, 35), rect, width=2, border_radius=6)
                bl_text = sfont.render("BL", True, (150, 150, 150))
                bt_rect = bl_text.get_rect(center=(x + CELL_W // 2, y + CELL_H // 2))
                screen.blit(bl_text, bt_rect)
                # show custom name if provided, otherwise show address
                label = HEATMAP_NAMES.get(disp_addr, f"0x{disp_addr:02X}")
                # support multiline names
                lines = str(label).splitlines()
                line_h = SMALL_FONT_SIZE + 2
                max_lines = CELL_H // line_h
                for li, line in enumerate(lines[:max_lines]):
                    addr_text = sfont.render(line, True, (120, 120, 120))
                    screen.blit(addr_text, (x + 5, y + 4 + li * line_h))
                continue

            v = int(ram[idx])
            col = color_map(v)

            # base cell
            pygame.draw.rect(screen, col, rect, border_radius=6)
            # normal border
            pygame.draw.rect(screen, (35, 35, 35), rect, width=2, border_radius=6)
            # green trend border (steady increase or decrease)
            if inc_counts[idx] >= TREND_FRAMES or dec_counts[idx] >= TREND_FRAMES:
                pygame.draw.rect(screen, (40, 220, 40), rect, width=3, border_radius=6)
            # flash border (on top) if active
            if flash_timers[idx] > 0:
                pygame.draw.rect(screen, (220, 40, 40), rect, width=3, border_radius=6)

            # Value bottom-right with small padding
            val_text = font.render(f"{v:3d}", True, (0, 0, 0) if v > 180 else (235, 235, 235))
            padding = 6
            vt_rect = val_text.get_rect(bottomright=(x + CELL_W - padding, y + CELL_H - padding))
            screen.blit(val_text, vt_rect)

            # Address top-left (display addresses starting at 0x80)
            label = HEATMAP_NAMES.get(disp_addr, f"0x{disp_addr:02X}")
            # support multiline names
            lines = str(label).splitlines()
            line_h = SMALL_FONT_SIZE + 2
            max_lines = CELL_H // line_h
            for li, line in enumerate(lines[:max_lines]):
                addr_text = sfont.render(line, True, (220, 220, 220))
                screen.blit(addr_text, (x + 5, y + 4 + li * line_h))

        # Named addresses panel (under grid) - display as a simple table (omit the address)
        if show_named and len(NAMED) > 0:
            panel_top = grid_y + grid_h
            pygame.draw.line(screen, (70, 70, 70),
                             (grid_x + PADDING, panel_top),
                             (grid_x + grid_w - PADDING, panel_top), 2)
            y = panel_top + int(PADDING * 0.75)
            # Header row
            h_name = sfont.render("NAME", True, (180, 180, 180))
            screen.blit(h_name, (grid_x + PADDING, y))
            h_val = sfont.render("VALUE", True, (180, 180, 180))
            h_val_rect = h_val.get_rect()
            h_val_rect.topright = (grid_x + grid_w - PADDING, y)
            screen.blit(h_val, h_val_rect)
            y += NAMED_LINE_H
            # Rows
            for name, addr in NAMED.items():
                a = addr & 0x7F
                v = int(ram[a])
                name_txt = sfont.render(f"{name}", True, (240, 240, 240))
                screen.blit(name_txt, (grid_x + PADDING, y))
                val_txt = sfont.render(f"{v:3d}", True, (240, 240, 240))
                val_rect = val_txt.get_rect()
                val_rect.topright = (grid_x + grid_w - PADDING, y)
                screen.blit(val_txt, val_rect)
                y += NAMED_LINE_H

        pygame.display.flip()
        clock.tick(FPS_CAP)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
