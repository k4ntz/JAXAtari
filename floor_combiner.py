import os
import numpy as np
import re

def combine_floor_npy_files(input_dir: str, output_path: str):
    """
    Combines a set of .npy files from the input directory vertically.
    The files are expected to be named in a format like 'floor_1.npy', 'floor_2.npy', etc.
    They will be sorted numerically so that floor_1 is on top of floor_2.
    """
    # Get all .npy files in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    # Sort files numerically based on the number in the filename
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')
        
    files.sort(key=extract_number)
    
    if not files:
        print(f"No .npy files found in {input_dir}")
        return
        
    print(f"Found {len(files)} files to combine in order:")
    for f in files:
        print(f" - {f}")
    
    # Load and combine the arrays
    arrays = []
    for file in files:
        file_path = os.path.join(input_dir, file)
        arr = np.load(file_path)
        print(f"Loaded {file} with shape {arr.shape}")
        arrays.append(arr)
        
    # Concatenate vertically (along axis 0)
    # vstack stacks them vertically, assuming height is the first dimension
    combined_array = np.vstack(arrays)
    
    print(f"Combined array shape: {combined_array.shape}")
    
    # Save the combined array
    np.save(output_path, combined_array)
    print(f"Saved combined array to {output_path}")

if __name__ == "__main__":
    # We use the directory name that actually exists on disk
    input_directory = "Tutankham_screensots_combiner"
    output_file = "floor_map4.npy"
    
    # Resolve absolute paths based on script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_directory)
    output_path = os.path.join(base_dir, "Tutankham_screensots_combiner", output_file)
    
    if os.path.exists(input_path):
        combine_floor_npy_files(input_path, output_path)
    else:
        print(f"Directory not found: {input_path}")

    #MAP1 834,160,4
    #MAP2 834,160,4
    #MAP3 834,160,4
    #MAP4 834,160,4
