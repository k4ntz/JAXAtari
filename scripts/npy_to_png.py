from PIL import Image, ImageOps
import numpy as np
import os

def npy_to_png(npy_file, scale_factor, output_dir, output_name):
    # Load the .npy file
    data = np.load(npy_file)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.fromarray(data)
    img = ImageOps.scale(img, scale_factor, resample=Image.NEAREST)
    img.save(os.path.join(output_dir, f'{output_name}.png'))
    
    print(f"Converted file to PNG format in '{output_dir}'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert .npy file to .png format")
    parser.add_argument("npy_file", type=str, help="Path to the input .npy file")
    parser.add_argument("--output_dir", "-o", type=str, help="Directory to save the output .png file", default=".")
    parser.add_argument("--output_name", "-n", type=str, help="Name of the output .png file (without extension)", default="output")
    parser.add_argument("--scale_factor", "-s", type=float, help="Scaling factor for the image", default=1.0)
    
    args = parser.parse_args()

    npy_to_png(args.npy_file, args.scale_factor, args.output_dir, args.output_name)