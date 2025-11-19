import os
import pickle
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
SAVESTATE_PATH = "battlezone_state.pkl"

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
    # "blue_tank_facing_direction": 0xAE,  # 46 + 0x80 
    # "blue_tank_size_y": 0xAF,            # 47
    # "blue_tank_x": 0xB0,                 # 48
    # "blue_tank2_facing_direction": 0xB4,# 52
    # "blue_tank2_size_y": 0xB5,          # 53
    # "blue_tank2_x": 0xB6,               # 54
    "frame": 0x80,
    "num_lives": 0xBA,                  # 58
    "missile_y": 0xE9,                  # 105
    # "compass_needles_angle": 0xD4,      # 84
    "bg_scroll_offset": 0x84,              # 4
    # "left_tread_position": 0xBB,        # 59
    # "right_tread_position": 0xBC,       # 60
    "crosshairs_color": 0xEC,  # !!!
    "score": 0x9D,   
    "enemy_a_X_hi": 0xC3,  # !!!
    "enemy_a_X_lo": 0xC4,  # !!!
    "enemy_a_Z_hi": 0xC5,  # !!!
    "enemy_a_Z_lo": 0xC6,  # !!!
    "enemy_b_X_hi": 0xCB,  # !!!
    "enemy_b_X_lo": 0xCC,  # !!!
    "enemy_b_Z_hi": 0xCD,  # !!!
    "enemy_b_Z_lo": 0xCE,  # !!!
    "enemy_a_sector": 0xE2,
    "enemy_b_sector": 0xE3,
    "enemy_a_distance_bucket": 0xD2,
    "enemy_b_distance_bucket": 0xD3,
    "fire0_X": 0xDC,
    "fire0_Z": 0xDE,
    "fire1_X": 0xD6,
    "fire1_Z": 0xD8,
}
# Addresses to hide from the heatmap (use displayed addresses 0x80..0xFF)
# Example: BLACKLIST = {0x80, 0xCA}
BLACKLIST = {0xDD, 0xD7, 0xC0, 0xA6, 0xBF, 0xFC, 0xFD, 0xFE, 0xFF, 0xDE, 0xDF, 0xA7, 0xA8, 0xD9, 0xDA, 0xD8, 0xD6}

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
    0x84: "bg_scroll\noffset",              # 4
    0xBB: "left_tread_position",        # 59
    0xBC: "right_tread_position",       # 60
    0xEC: "crosshairs_color",  # 108 -> 0xEC; merged with previous
    0x9D: "score",                      # 29
}

# Per-frame logging: fill with addresses (display addresses 0x80..0xFF) to log
# Example: LOG_ADDRESSES = {"player_x": 0xCA, "player_y": 0xCB}
LOG_ADDRESSES: Dict[str, int] = {
    "frame": 0x80,
    "enemy_a_X_hi": 0xC3,  # !!!
    "enemy_a_X_lo": 0xC4,  # !!!
    "enemy_a_Z_hi": 0xC5,  # !!!
    "enemy_a_Z_lo": 0xC6,  # !!!
    "enemy_b_X_hi": 0xCB,  # !!!
    "enemy_b_X_lo": 0xCC,  # !!!
    "enemy_b_Z_hi": 0xCD,  # !!!
    "enemy_b_Z_lo": 0xCE,  # !!!
    "fire0_X": 0xDC,
    "fire0_Z": 0xDE,
    "fire1_X": 0xD6,
    "fire1_Z": 0xD8,
}

# Logfile path for per-frame logging. Set to None to disable file output.
LOGFILE_PATH = "ram_log.txt"

# Controls:
#  - Arrows = movement. Space = FIRE. (We auto-map to ALE combos like "UPFIRE" if present.)
#  - P = pause/resume stepping
#  - R = reset
#  - N = toggle named-address panel
#  - Q = save savestate
#  - W = load savestate (if one was saved)
#  - ESC = quit
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
    logging_enabled = False
    log_file = None
    pressed = set()  # {"UP","DOWN","LEFT","RIGHT","FIRE"}
    saved_state = None

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

    # Load savestate pointer from disk if present (but don't restore automatically)
    if os.path.exists(SAVESTATE_PATH):
        try:
            with open(SAVESTATE_PATH, "rb") as f:
                saved_state = pickle.load(f)
            print(f"Found savestate at {SAVESTATE_PATH}. Press 'W' to load it.")
        except Exception as e:
            saved_state = None
            print(f"Failed to load savestate from disk: {e}")

    running = True

    # Print the available key commands once on startup
    print("Controls:")
    print("  Arrows: movement (mapped to ALE action meanings)")
    print("  Space: FIRE")
    print("  P: pause/resume stepping")
    print("  +: advance one tick when paused")
    print("  R: reset environment")
    print("  S: save a screenshot of the game frame")
    print("  N: toggle named-address panel (if any names configured)")
    print("  Q: save a savestate (persisted on quit)")
    print("  W: load the saved savestate (if any)")
    print("  ESC: quit")
    print(f"  Screenshot format: {SCREENSHOT_FORMAT} (one game px -> one saved px)")
    print("  L: toggle per-frame logging of addresses in LOG_ADDRESSES")
    while running:
        # ---- Input ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_q:
                    try:
                        saved_state = ale.cloneState()
                        print("Saved savestate.")
                    except Exception as e:
                        print(f"Failed to save savestate: {e}")
                elif k == pygame.K_w:
                    if saved_state is None:
                        print("No savestate available to load.")
                    else:
                        try:
                            ale.restoreState(saved_state)
                            obs = env.render()
                            # Reset change tracking so the restored frame starts cleanly
                            last_vals[:] = -1
                            stable_counts[:] = 0
                            flash_timers[:] = 0
                            inc_counts[:] = 0
                            dec_counts[:] = 0
                            print("Loaded savestate.")
                        except Exception as e:
                            print(f"Failed to load savestate: {e}")
                elif k == pygame.K_p:
                    paused = not paused
                # Single-step advance when paused: press '+' to advance one tick
                elif getattr(event, "unicode", "") == "+" and paused:
                    action = resolve_action(pressed, meaning_to_id, noop_id)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset()
                        last_vals[:] = -1
                        stable_counts[:] = 0
                        flash_timers[:] = 0
                        inc_counts[:] = 0
                        dec_counts[:] = 0
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
                elif k == pygame.K_l:
                    logging_enabled = not logging_enabled
                    if logging_enabled:
                        # open logfile for append
                        if LOGFILE_PATH:
                            try:
                                log_file = open(LOGFILE_PATH, "a")
                                log_file.write(f"--- LOG START {time.asctime()} ---\n")
                                log_file.flush()
                            except Exception as e:
                                print(f"Failed to open logfile {LOGFILE_PATH}: {e}")
                                logging_enabled = False
                        print("Logging ENABLED")
                        if len(LOG_ADDRESSES) == 0:
                            print("LOG_ADDRESSES is empty — add addresses to LOG_ADDRESSES at top of file to start logging.")
                    else:
                        print("Logging DISABLED")
                        if log_file is not None:
                            try:
                                # log_file.write(f"--- LOG STOP {time.asctime()} ---\n")
                                log_file.close()
                            except Exception:
                                pass
                            log_file = None
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

        # pressed.add("UP")  # always move forward for testing
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

        # Per-frame logging of selected RAM addresses (if enabled)
        if logging_enabled and len(LOG_ADDRESSES) > 0 and LOGFILE_PATH:
            # write each selected address value (display address -> value) to logfile
            try:
                for name, addr in LOG_ADDRESSES.items():
                    disp_addr = int(addr) & 0xFF
                    idx = disp_addr & 0x7F
                    v = int(ram[idx])
                    if log_file:
                        log_file.write(f"{name} (0x{disp_addr:02X}): {v}\n")
                if log_file:
                    log_file.write("---\n")
                    log_file.flush()
            except Exception:
                # best-effort: don't crash on logging errors
                pass

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

        # Named addresses panel (under grid) - compact table that fits contents
        if show_named and len(NAMED) > 0:
            panel_top = grid_y + grid_h
            rows = len(NAMED) + 1  # header + entries
            row_h = NAMED_LINE_H

            # compute column widths based on rendered text sizes
            name_col_w = 0
            val_col_w = 0
            # include header widths
            nh_w, _ = sfont.size("NAME")
            vh_w, _ = sfont.size("VALUE")
            name_col_w = max(name_col_w, nh_w)
            val_col_w = max(val_col_w, vh_w)
            # include content widths
            for name, addr in NAMED.items():
                nw, _ = sfont.size(str(name))
                a = addr & 0x7F
                v = int(ram[a])
                vw, _ = sfont.size(f"{v:3d}")
                if nw > name_col_w:
                    name_col_w = nw
                if vw > val_col_w:
                    val_col_w = vw

            # padding inside cells and gap between columns
            cell_pad = 8
            col_gap = 12
            panel_w = PADDING + name_col_w + col_gap + val_col_w + PADDING
            panel_h = rows * row_h + int(PADDING * 0.5)

            panel_left = grid_x + PADDING
            panel_rect = pygame.Rect(panel_left, panel_top, panel_w, panel_h)

            # background of panel (subtle)
            pygame.draw.rect(screen, (18, 18, 18), panel_rect, border_radius=4)
            # outer border
            pygame.draw.rect(screen, (70, 70, 70), panel_rect, width=1, border_radius=4)

            # vertical separator x (between name and value)
            name_col_x = panel_left + PADDING
            sep_x = name_col_x + name_col_w + col_gap
            # draw vertical separator
            pygame.draw.line(screen, (70, 70, 70), (sep_x, panel_top), (sep_x, panel_top + panel_h), 1)

            # draw horizontal lines (header separator + row separators)
            for i in range(rows + 1):
                y0 = panel_top + i * row_h
                pygame.draw.line(screen, (70, 70, 70), (panel_left, y0), (panel_left + panel_w, y0), 1)

            # render header
            y = panel_top + 0
            h_name = sfont.render("NAME", True, (180, 180, 180))
            screen.blit(h_name, (name_col_x + cell_pad // 2, y + (row_h - h_name.get_height()) // 2))
            h_val = sfont.render("VALUE", True, (180, 180, 180))
            h_val_rect = h_val.get_rect()
            h_val_rect.topright = (panel_left + panel_w - PADDING - cell_pad // 2, y + (row_h - h_val.get_height()) // 2)
            screen.blit(h_val, h_val_rect)

            # render rows
            y += row_h
            for name, addr in NAMED.items():
                a = addr & 0x7F
                v = int(ram[a])
                name_txt = sfont.render(f"{name}", True, (240, 240, 240))
                screen.blit(name_txt, (name_col_x + cell_pad // 2, y + (row_h - name_txt.get_height()) // 2))
                val_txt = sfont.render(f"{v:3d}", True, (240, 240, 240))
                val_rect = val_txt.get_rect()
                val_rect.topright = (panel_left + panel_w - PADDING - cell_pad // 2, y + (row_h - val_txt.get_height()) // 2)
                screen.blit(val_txt, val_rect)
                y += row_h

        pygame.display.flip()
        clock.tick(FPS_CAP)

    # Persist savestate (if any) on quit
    if saved_state is not None:
        try:
            with open(SAVESTATE_PATH, "wb") as f:
                pickle.dump(saved_state, f)
            print(f"Saved savestate to {SAVESTATE_PATH}")
        except Exception as e:
            print(f"Failed to save savestate to disk: {e}")

    env.close()
    # ensure logfile closed on exit
    try:
        if log_file is not None:
            # log_file.write(f"--- LOG END {time.asctime()} ---\n")
            log_file.close()
    except Exception:
        pass
    pygame.quit()

if __name__ == "__main__":
    main()
