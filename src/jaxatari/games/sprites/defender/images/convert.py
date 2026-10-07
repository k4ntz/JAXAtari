import os
import numpy as np

def jpg_into_npy(png_file, npy_file):
    from PIL import Image
    img = Image.open(png_file)
    # Erzwinge 4 Kanäle (RGBA), damit Alpha erhalten bleibt
    img = img.convert('RGBA')
    img_array = np.array(img)
    print(img_array.shape)
    print(img_array[0, 0])
    np.save(npy_file, img_array)
    print(f"Saved {png_file} as {npy_file}")



if __name__ == "__main__":
    print("Hallo")
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    png_files = os.listdir(".")
    print(png_files)
    for png_file in png_files:
        if png_file.endswith(".png"):
            npy_file = png_file.replace(".png", ".npy")
            jpg_into_npy(png_file, "../" + npy_file)

