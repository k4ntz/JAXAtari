import numpy as np
import os

EXTRACTED_DIR = "extracted_sprites"
TARGET_DIR = "src/jaxatari/games/sprites/pacman"

# Map source files
PACMAN_SOURCE = "sprite_7_color_252_224_144_7x14.npy"
GHOST_SOURCE = "sprite_1_color_252_144_200_8x16.npy"
WAFER_SOURCE = "sprite_19_color_223_192_111_4x2.npy"

# Colors (Atari palette / User Request)
GHOST_COLOR_REGULAR = [50, 50, 255]    # Blue
GHOST_COLOR_FRIGHTENED = [252, 144, 200] # Pink
PACMAN_COLOR = [255, 255, 0] # Yellow
WAFER_COLOR = [200, 180, 150]

def save_npy(rel_path, arr):
    path = os.path.join(TARGET_DIR, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr.astype(np.uint8))
    print(f"Saved {rel_path} ({arr.shape})")

def clean_sprite(arr):
    """
    Removes background pixels (sets to 0,0,0).
    Assumes background is top-left pixel color.
    """
    if arr.ndim < 3: return arr
    bg_color = arr[0, 0] 
    out = arr.copy()
    mask = np.all(arr == bg_color, axis=-1)
    out[mask] = [0, 0, 0]
    return out

def recolor_sprite(arr, new_color):
    """
    Sets non-black pixels to new_color.
    """
    out = arr.copy()
    mask = np.any(arr != 0, axis=-1)
    out[mask] = new_color
    return out

def normalize_to_right(arr):
    """
    Ensures the sprite is facing right.
    Assumes mouth (gap) is the side with fewer pixels.
    """
    h, w, _ = arr.shape
    mid = w // 2
    
    # Count non-zero pixels on left and right
    left_side = arr[:, :mid]
    right_side = arr[:, mid:]
    
    left_mass = np.sum(np.any(left_side != 0, axis=-1))
    right_mass = np.sum(np.any(right_side != 0, axis=-1))
    
    # If Left has LESS mass than Right, it is facing LEFT (Mouth is on Left)
    # We want it facing RIGHT. So Flip.
    if left_mass < right_mass:
        return np.fliplr(arr)
        
    return arr

def process_sprites():
    # 1. PACMAN
    candidates = []
    for f in os.listdir(EXTRACTED_DIR):
        if "color_252_224_144_7x14" in f:
            candidates.append(f)
            
    if not candidates:
        print("No Pacman candidates found! Using fallback.")
        if os.path.exists(os.path.join(EXTRACTED_DIR, PACMAN_SOURCE)):
            candidates.append(PACMAN_SOURCE)
            
    unique_states = {}
    for c_file in candidates:
        path = os.path.join(EXTRACTED_DIR, c_file)
        if not os.path.exists(path): continue
        arr = np.load(path)
        clean = clean_sprite(arr)
        mask_clean = np.any(clean != 0, axis=-1)
        count_clean = np.sum(mask_clean)
        if count_clean not in unique_states:
            unique_states[count_clean] = clean # Temporarily store clean raw

    print(f"Found {len(unique_states)} unique Pacman states: {list(unique_states.keys())}")
    
    sorted_counts = sorted(unique_states.keys()) # Ascending (Open -> Closed)
    states_list = [unique_states[k] for k in sorted_counts] 
    
    # Normalize ALL states to Right Facing
    states_list = [normalize_to_right(s) for s in states_list]
    
    frames = []
    if not states_list:
        print("Error: No valid pacman states.")
    elif len(states_list) == 1:
        frames = [states_list[0]] * 3
    elif len(states_list) == 2:
        frames = [states_list[0], states_list[1], states_list[1]] 
    else:
        frames = [states_list[0], states_list[1], states_list[-1]] 
        
    for i, p_arr in enumerate(frames):
        p_small = p_arr[::2, :] # Resize 14->7
        p_final = np.zeros((8, 8, 3), dtype=np.uint8)
        p_final[0:7, 0:7] = p_small
        p_final = recolor_sprite(p_final, PACMAN_COLOR)
        
        for d_name in ['right', 'left', 'up', 'down']:
            final_oriented = p_final
            if d_name == 'left': final_oriented = np.fliplr(p_final)
            elif d_name == 'up': final_oriented = np.rot90(p_final)
            elif d_name == 'down': final_oriented = np.rot90(p_final, -1)
            save_npy(f"player/player_{d_name}_{i}.npy", final_oriented)
            
    # Death
    if frames:
        death_base = frames[0]
        p_small = death_base[::2, :]
        p_final = np.zeros((8, 8, 3), dtype=np.uint8)
        p_final[0:7, 0:7] = p_small
        p_final = recolor_sprite(p_final, PACMAN_COLOR)
        for i in range(12):
            pp = p_final.copy()
            if i > 0: pp[:min(i, 8), :] = 0
            save_npy(f"player/death_{i}.npy", pp)

    # 2. GHOST
    g_path = os.path.join(EXTRACTED_DIR, GHOST_SOURCE)
    if os.path.exists(g_path):
        g_arr = np.load(g_path)
        g_clean = clean_sprite(g_arr)
        g_final = g_clean[::2, :] # Resize 16->8
        
        g_regular = recolor_sprite(g_final, GHOST_COLOR_REGULAR)
        ghost_types = ['blinky', 'pinky', 'inky', 'clyde']
        for gtype in ghost_types:
            for d in ['right', 'left', 'up', 'down']:
                for f in [0, 1, 2]: 
                    save_npy(f"ghost_{gtype}/ghost_{d}_{f}.npy", g_regular)
        
        g_fright = recolor_sprite(g_final, GHOST_COLOR_FRIGHTENED)
        for f in [0, 1, 2]:
            save_npy(f"ghost_frightened/ghost_frightened_{f}.npy", g_fright)
            
        g_white = recolor_sprite(g_final, [230, 230, 250])
        for f in [0, 1, 2]:
            save_npy(f"ghost_frightened/ghost_frightened_white_{f}.npy", g_white)
    else:
         print(f"Ghost source {GHOST_SOURCE} not found!")

    # 3. WAFER
    w_path = os.path.join(EXTRACTED_DIR, WAFER_SOURCE)
    if os.path.exists(w_path):
        w_arr = np.load(w_path)
        w_clean = w_arr # clean_sprite(w_arr)
        w_final = np.zeros((8, 8, 3), dtype=np.uint8)
        h, w, _ = w_clean.shape
        y_off = (8 - h) // 2
        x_off = (8 - w) // 2
        w_final[y_off:y_off+h, x_off:x_off+w] = w_clean
        w_final = recolor_sprite(w_final, WAFER_COLOR)
        
        save_npy("pellet_dot.npy", w_final)
        
        p_power = np.zeros((8, 8, 3), dtype=np.uint8)
        p_power[2:6, 1:7] = WAFER_COLOR 
        save_npy("pellet_power/pellet_power_on.npy", p_power)
        save_npy("pellet_power/pellet_power_off.npy", np.zeros((8,8,3), dtype=np.uint8))
    else:
        print(f"Wafer source {WAFER_SOURCE} not found!")

if __name__ == "__main__":
    process_sprites()
