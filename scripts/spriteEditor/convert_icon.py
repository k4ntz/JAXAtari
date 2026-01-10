import base64
import io
from PIL import Image
import sys

# --- Configuration ---
image_file = sys.argv[1] 
target_size = (32, 32)    # The desired dimensions
output_format = "png"       # Format to save as (GIF or PNG are best for Tkinter)
# ---------------------

# 1. Open the image file
with Image.open(image_file) as img:
    
    # 2. Resize the image with a high-quality antialiasing filter
    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 3. Save the resized image to an in-memory buffer
    with io.BytesIO() as buffer:
        # We save it in the specified format (GIF is great for Tkinter)
        resized_img.save(buffer, format=output_format)
        
        # 4. Get the binary data from the buffer
        image_data = buffer.getvalue()

# 5. Encode the *new* binary data into a Base64 string
base64_string = base64.b64encode(image_data).decode('utf-8')

# 6. Print the string so you can copy-paste it
print(f"'{base64_string}'")
