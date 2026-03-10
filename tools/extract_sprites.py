
import numpy as np
import os
import cv2 # Assuming cv2 is available or I can use PIL if not. ALE usually implies similar envs.
# If cv2 not available, I'll use simple numpy.
from PIL import Image

SCREENSHOTS_DIR = "Pacman_screenshots"
OUTPUT_DIR = "extracted_sprites"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_frames():
    files = [f for f in os.listdir(SCREENSHOTS_DIR) if f.endswith('.npy')]
    if not files:
        print("No .npy files found.")
        return

    print(f"Found {len(files)} frames.")
    
    unique_colors = set()
    
    for f in files:
        path = os.path.join(SCREENSHOTS_DIR, f)
        frame = np.load(path)
        print(f"File: {f}, Shape: {frame.shape}, Dtype: {frame.dtype}, Size: {frame.size}")
        
        # Determine reshaping strategy
        if frame.ndim == 3 and frame.shape[2] == 4:
             print("  Converting RGBA to RGB...")
             frame = frame[:, :, :3] # Drop Alpha
             
        if frame.ndim == 3 and frame.shape[2] == 3:
             colors = frame.reshape(-1, 3)
        elif frame.ndim == 2:
             # Grayscale or palette?
             print("  Warning: 2D array found. Treating as grayscale/indexed.")
             continue # skip for now or treat as 1D indexes
        else:
             print(f"  Warning: Unexpected shape {frame.shape}. Skipping.")
             continue

        # Convert to tuple to hash
        frame_colors = set(tuple(c) for c in colors)
        unique_colors.update(frame_colors)
        
    print(f"\nUnique Distinct Colors Found ({len(unique_colors)}):")
    sorted_colors = sorted(list(unique_colors))
    for i, c in enumerate(sorted_colors):
        count = 0
        # Count occurrence in first frame just for info
        if len(files) > 0:
            f0 = np.load(os.path.join(SCREENSHOTS_DIR, files[0]))
            count = np.sum(np.all(f0 == c, axis=-1))
            
        print(f"Color {i}: {c} - Approx count in frame 1: {count}")
        
    # Attempt extraction
    # Strategy: For each non-background color, find connected components or bounding boxes.
    # Assume most frequent color is background.
    
    if not files: return
    
    first_frame = np.load(os.path.join(SCREENSHOTS_DIR, files[0]))
    # Find background color (most frequent)
    vals, counts = np.unique(first_frame.reshape(-1, 3), axis=0, return_counts=True)
    bg_color = vals[np.argmax(counts)]
    print(f"\nBackground Color seemingly: {bg_color}")
    
    sprite_counter = 0
    
    for f in files:
        frame = np.load(os.path.join(SCREENSHOTS_DIR, f))
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
            
        # Mask valid pixels (not background)
        mask = ~np.all(frame == bg_color, axis=-1)
        
        # We need to find connected components or blobs.
        # Since I might not have cv2, let's use a simple scanning approach or scipy if available?
        # Let's assume standard grid alignment? Atari has specific grid:
        # standard char mode is 8px wide.
        
        # Naive "Blob" extraction using simple bounding box of connected colors?
        # Actually, separate by color first.
        
        frame_unique_colors = set(tuple(c) for c in frame.reshape(-1, 3))
        
        for col in frame_unique_colors:
            col_arr = np.array(col)
            if np.array_equal(col_arr, bg_color):
                continue
            
            # Create binary mask for this color
            color_mask = np.all(frame == col_arr, axis=-1)
            
            # Find bounds
            rows = np.any(color_mask, axis=1)
            cols = np.any(color_mask, axis=0)
            
            if not np.any(rows): continue
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # This bounding box might contain multiple sprites of same color (e.g. dots)
            # We want to extract small distinct regions.
            
            # Simple grid search: verify if there are disconnected regions.
            # Using skimage or scipy usually best.
            # Let's try basic "find separated islands" algorithm for the mask.
            
            visited = np.zeros_like(color_mask, dtype=bool)
            
            h, w = color_mask.shape
            
            for y in range(h):
                for x in range(w):
                    if color_mask[y, x] and not visited[y, x]:
                        # Flood fill to find component
                        stack = [(y, x)]
                        visited[y, x] = True
                        min_r, max_r = y, y
                        min_c, max_c = x, x
                        
                        component_pixels = []
                        
                        while stack:
                            cy, cx = stack.pop()
                            component_pixels.append((cy, cx))
                            min_r = min(min_r, cy)
                            max_r = max(max_r, cy)
                            min_c = min(min_c, cx)
                            max_c = max(max_c, cx)
                            
                            # 4-neighbor check
                            neighbors = []
                            if cy > 0: neighbors.append((cy-1, cx))
                            if cy < h-1: neighbors.append((cy+1, cx))
                            if cx > 0: neighbors.append((cy, cx-1))
                            if cx < w-1: neighbors.append((cy, cx+1))
                            
                            for ny, nx in neighbors:
                                if color_mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))
                        
                        # Extract sprite
                        sprite_h = max_r - min_r + 1
                        sprite_w = max_c - min_c + 1
                        
                        # Filter likely noise
                        if sprite_h < 2 or sprite_w < 2: continue
                        
                        sprite_img = np.zeros((sprite_h, sprite_w, 3), dtype=np.uint8)
                        # Fill background
                        # Actually for extraction, we might want transparent background (0)
                        # or fill with the sprite color.
                        
                        # We copy from original frame
                        y_slice = frame[min_r:max_r+1, min_c:max_c+1]
                        
                        # Mask out parts that are not part of THIS component (e.g. intertwined same-color sprites?)
                        # For simple flood fill on single color mask, the slice contains only that color or background intervals.
                        # But wait, if donut shape?
                        # Let's just crop the rect for now, assuming usually convex or isolated.
                        
                        # Ensure we only keep the color we want?
                        # Or just save the raw patch.
                        
                        # Save
                        s_name = f"sprite_{sprite_counter}_color_{col[0]}_{col[1]}_{col[2]}_{sprite_w}x{sprite_h}.npy"
                        np.save(os.path.join(OUTPUT_DIR, s_name), y_slice)
                        sprite_counter += 1
                        print(f"Extracted {s_name}")

if __name__ == "__main__":
    analyze_frames()
