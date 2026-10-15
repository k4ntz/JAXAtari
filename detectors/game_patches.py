import cv2
import numpy as np

# Global registry for custom game patches
GAME_PATCHES = {}


def register_patch(game_name):
    """Decorator to register custom computer vision patches per game."""

    def decorator(func):
        GAME_PATCHES[game_name] = func
        return func

    return decorator


@register_patch("enduro")
def patch_enduro(pixels, yolo_lines, frame_bgr, to_yolo, draw_debug_box, debug_mode):
    y_min, y_max = 140, 155
    car_strip = pixels[y_min:y_max, :, :]
    brightness = np.sum(car_strip, axis=2)
    y_coords, x_coords = np.where(brightness > 400)

    if len(x_coords) > 0:
        car_x = int(np.min(x_coords))
        car_w = int(np.max(x_coords) - car_x)
        car_h = y_max - y_min
        patch_h, patch_w = pixels.shape[:2]

        yolo_lines.append(to_yolo(0, car_x, y_min, car_w, car_h, patch_h, patch_w))
        if debug_mode:
            draw_debug_box(frame_bgr, car_x, y_min, car_w, car_h, 0, "entity", "player_car")
    return None


@register_patch("mspacman")
def patch_mspacman(pixels, yolo_lines, frame_bgr, to_yolo, draw_debug_box, debug_mode):
    patch_h, patch_w = pixels.shape[:2]

    # Extract Pacman
    player_color = np.array([210, 164, 74])
    p_mask = cv2.inRange(pixels, np.clip(player_color - 25, 0, 255), np.clip(player_color + 25, 0, 255))
    p_contours, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in p_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 4 <= w <= 16 and 4 <= h <= 16 and y < 176:
            yolo_lines.append(to_yolo(0, x, y, w, h, patch_h, patch_w))
            if debug_mode:
                pad_x = 2
                pad_y = 3

                px = max(0, x - pad_x)
                py = max(0, y - pad_y)
                pw = min(patch_w - px, w + (pad_x * 2))
                ph = min(patch_h - py, h + (pad_y * 2))

                yolo_lines.append(to_yolo(0, px, py, pw, ph, patch_h, patch_w))
                if debug_mode:
                    draw_debug_box(frame_bgr, px, py, pw, ph, 0, "entity", "player_cv")

    # Extract Pellets
    palette_colors = [
        np.array([210, 164, 116]), np.array([228, 111, 111]),
        np.array([200, 200, 80]), np.array([214, 214, 214])
    ]
    combined_pellet_mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
    for color in palette_colors:
        mask = cv2.inRange(pixels, np.clip(color - 15, 0, 255), np.clip(color + 15, 0, 255))
        combined_pellet_mask = cv2.bitwise_or(combined_pellet_mask, mask)

    pel_contours, _ = cv2.findContours(combined_pellet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in pel_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 2 <= w <= 12 and 1 <= h <= 6 and y < 176:
            yolo_lines.append(to_yolo(3, x, y, w, h, patch_h, patch_w))
            if debug_mode:
                draw_debug_box(frame_bgr, x, y, w, h, 3, "entity", "pellet_cv")
    return None


@register_patch("phoenix")
def patch_phoenix(pixels, yolo_lines, frame_bgr, to_yolo, draw_debug_box, debug_mode):
    patch_h, patch_w = pixels.shape[:2]
    s_mask = cv2.inRange(pixels, np.array([200, 200, 200]), np.array([255, 255, 255]))

    s_contours, _ = cv2.findContours(s_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in s_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 14 <= w <= 40 and 14 <= h <= 40 and y > 150:
            yolo_lines.append(to_yolo(4, x, y, w, h, patch_h, patch_w))
            if debug_mode:
                draw_debug_box(frame_bgr, x, y, w, h, 4, "entity", "shield_cv")
        elif w <= 4 and 3 <= h <= 15 and y > 25:
            pad = 2
            px, py, pw, ph = max(0, x - pad), max(0, y - pad), w + (pad * 2), h + (pad * 2)
            yolo_lines.append(to_yolo(6, px, py, pw, ph, patch_h, patch_w))
            if debug_mode:
                draw_debug_box(frame_bgr, px, py, pw, ph, 6, "entity", "enemy_proj_cv")
    return None


@register_patch("namethisgame")
def patch_namethisgame(pixels, yolo_lines, frame_bgr, to_yolo, draw_debug_box, debug_mode):
    """
    1. Clusters raw tentacle segments into unified tentacle-tip bounding boxes.
    2. Keeps shark, player, boat, spear, and oxygen_line intact.
    """
    patch_h, patch_w = pixels.shape[:2]

    parsed_boxes = []
    non_tentacle_lines = []

    # Separate tentacle enemy boxes (class 1) in the tentacle zone (y between 70 and 130)
    for line in yolo_lines:
        parts = line.split()
        cls_id = int(parts[0])
        xc, yc, wn, hn = map(float, parts[1:])

        w = wn * patch_w
        h = hn * patch_h
        x = (xc * patch_w) - (w / 2.0)
        y = (yc * patch_h) - (h / 2.0)

        if cls_id == 1 and 70 <= y <= 130 and w < 12:
            parsed_boxes.append((x, y, w, h))
        else:
            non_tentacle_lines.append(line)

    yolo_lines[:] = non_tentacle_lines

    if not parsed_boxes:
        return None

    parsed_boxes.sort(key=lambda b: b[0])

    clusters = []
    current_cluster = [parsed_boxes[0]]

    for box in parsed_boxes[1:]:
        prev_x2 = current_cluster[-1][0] + current_cluster[-1][2]
        if box[0] - prev_x2 <= 8:
            current_cluster.append(box)
        else:
            clusters.append(current_cluster)
            current_cluster = [box]
    clusters.append(current_cluster)

    for cluster in clusters:
        min_x = min(b[0] for b in cluster)
        min_y = min(b[1] for b in cluster)
        max_x = max(b[0] + b[2] for b in cluster)
        max_y = max(b[1] + b[3] for b in cluster)

        merged_w = max_x - min_x
        merged_h = max_y - min_y

        yolo_lines.append(to_yolo(1, min_x, min_y, merged_w, merged_h, patch_h, patch_w))

        if debug_mode:
            draw_debug_box(frame_bgr, min_x, min_y, merged_w, merged_h, 1, "entity", "tentacles")

    return None


@register_patch("hauntedhouse")
def patch_hauntedhouse(pixels, yolo_lines, frame_bgr, to_yolo, draw_debug_box, debug_mode):
    patch_h, patch_w = pixels.shape[:2]
    yolo_lines[:] = [line for line in yolo_lines if not line.startswith("0 ")]

    is_white_bg = np.mean(pixels[5:20, 5:20]) > 200
    claimed_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    claimed_mask[135:, :] = 255  # Lock out bottom HUD

    # Extract Eyes
    lower = np.array([200, 200, 200]) if not is_white_bg else np.array([0, 0, 0])
    upper = np.array([255, 255, 255]) if not is_white_bg else np.array([30, 30, 30])
    eye_mask = cv2.inRange(pixels, lower, upper)
    eye_mask[135:, :] = 0
    y_idx, x_idx = np.where(eye_mask > 0)

    if len(x_idx) >= 4:
        x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
        y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
        ew, eh = (x_max - x_min) + 1, (y_max - y_min) + 1

        if 4 <= ew <= 18 and 3 <= eh <= 12:
            pad = 2
            px, py, pw, ph = max(0, x_min - pad), max(0, y_min - pad), ew + (pad * 2), eh + (pad * 2)
            yolo_lines.append(to_yolo(0, px, py, pw, ph, patch_h, patch_w))
            claimed_mask[py:py + ph, px:px + pw] = 255
            if debug_mode:
                draw_debug_box(frame_bgr, px, py, pw, ph, 0, "entity", "player_eyes_cv")

    # Extract Enemies
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    if not is_white_bg:
        blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([140, 255, 255]))
        black_mask = cv2.inRange(pixels, np.array([0, 0, 0]), np.array([15, 15, 15]))
        fg_mask = cv2.bitwise_not(cv2.bitwise_or(blue_mask, black_mask))
    else:
        fg_mask = cv2.bitwise_not(cv2.inRange(pixels, np.array([200, 200, 200]), np.array([255, 255, 255])))

    fg_mask[claimed_mask > 0] = 0
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_enemy_type = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 5 <= w <= 25 and 6 <= h <= 30 and not (w <= 12 and h <= 10 and x > 110):
            sprite_crop = pixels[y:y + h, x:x + w]
            detected_enemy_type = _classify_hauntedhouse_enemy(sprite_crop)
            yolo_lines.append(to_yolo(1, x, y, w, h, patch_h, patch_w))
            if debug_mode:
                draw_debug_box(frame_bgr, x, y, w, h, 1, "entity", "enemies_cv")

    return detected_enemy_type


def _classify_hauntedhouse_enemy(crop_rgb):
    if crop_rgb.size == 0:
        return 'unknown'
    green = np.sum((crop_rgb[:, :, 1] > 140) & (crop_rgb[:, :, 2] < 120))
    orange = np.sum((crop_rgb[:, :, 0] > 150) & (crop_rgb[:, :, 1] > 100) & (crop_rgb[:, :, 2] < 80))
    if green > 4 or orange > 4:
        return 'spider'
    if np.sum((crop_rgb[:, :, 0] > 140) & (crop_rgb[:, :, 1] < 80) & (crop_rgb[:, :, 2] < 80)) > 5:
        return 'bat'
    return 'ghost'