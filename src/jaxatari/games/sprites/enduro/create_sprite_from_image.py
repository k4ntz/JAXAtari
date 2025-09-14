import os
import numpy as np
from PIL import Image, ImageSequence



def convert_image_to_npy(image_path, output_dir=None, mode="RGBA", save_png=False):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = output_dir or "."
    os.makedirs(output_dir, exist_ok=True)

    try:
        if img.format == "GIF" and convert_all_frames:
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                frame_converted = frame.convert(mode)
                frame_array = np.array(frame_converted)

                # either save as PNG or npy
                if save_png:
                    frame_png_path = os.path.join(output_dir, f"{base_name}_frame{i:03d}.png")
                    frame_converted.save(frame_png_path)
                    print(f"Saved: {frame_png_path}")
                else:
                    frame_filename = os.path.join(output_dir, f"{base_name}_frame{i:03d}.npy")
                    np.save(frame_filename, frame_array)
                    print(f"Saved: {frame_filename} | Shape: {frame_array.shape}")
        else:
            img_array = np.array(img.convert(mode))
            out_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(out_path, img_array)
            print(f"Saved: {out_path} | Shape: {img_array.shape}")
    except Exception as e:
        print(f"Error saving array: {e}")


def make_white_pixels_transparent(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            try:
                arr = np.load(file_path)

                if arr.shape[-1] != 4:
                    print(f"Skipping {filename}: not an RGBA array")
                    continue

                # Mask where R=255, G=255, B=255
                white_mask = np.all(arr[..., :3] == 255, axis=-1)

                # Set alpha to 0 where the pixel is white
                arr[white_mask, 3] = 0

                # Save it back
                np.save(file_path, arr)
                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def load_images_from_dir(folder_path):
    png_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            png_images.append(os.path.join(folder_path, filename))
    return png_images


# 🔧 Set your image filename and output folder here
output_path = "backgrounds/"
convert_all_frames = True  # Set to False to convert only the first frame


if __name__ == '__main__':
    # images = load_images_from_dir('enduro/original_images/')
    # print(images)

    single_path = 'original_images/background_colors/green_level_background.png'
    images = [single_path]

    for img_path in images:
        # Run the conversion
        convert_image_to_npy(img_path, output_path)

    make_white_pixels_transparent(output_path)


