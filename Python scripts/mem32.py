import cv2
import numpy as np
import os

# === CONFIG ===
image_path = 'not_fall1.jpg'    # Input .jpg file
output_path= 'not_fall1.mem' # Output .mem file, adjusted for 32x32 size
image_size = (32, 32)                   # Changed to 32x32
grayscale = True                        # Keep as grayscale

# === LOAD IMAGE ===
flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
img = cv2.imread(image_path, flag)

if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# === RESIZE & CONVERT TO UINT8 (0-255) ===
# This step ensures the image is 32x32 and its pixel values are within 0-255.
img = cv2.resize(img, image_size)
img = np.clip(img, 0, 255).astype(np.uint8)

# === FLATTEN & CONVERT TO SIGNED INT8 (-128 to 127) ===
# The conversion from unsigned 0-255 to signed -128 to 127
# For an 8-bit unsigned value 'u', the corresponding signed 8-bit value 's' is:
# if u <= 127: s = u
# if u >= 128: s = u - 256
# NumPy's astype(np.int8) on a uint8 array handles this conversion automatically.
# However, the common practice in CNNs is to center data around zero.
# This transformation maps:
#    0 (uint8) -> 0 - 128 = -128 (int8)
#  127 (uint8) -> 127 - 128 = -1 (int8)
#  128 (uint8) -> 128 - 128 = 0 (int8)
#  255 (uint8) -> 255 - 128 = 127 (int8)
# We use int16 temporarily to avoid overflow during the subtraction before casting to int8.
flat_unsigned = img.flatten() # Get 1D array of 0-255 values
flat_signed = (flat_unsigned.astype(np.int16) - 128).astype(np.int8)

# === SAVE AS .mem (hex format representing 8-bit signed two's complement) ===
with open(output_path, 'w') as f:
    for val in flat_signed:
        # To get the 8-bit two's complement hex representation of a signed integer:
        # 1. Convert the signed int8 to its equivalent unsigned int8 representation.
        #    NumPy handles this correctly when casting `np.int8` to `np.uint8`.
        unsigned_val_for_hex = np.uint8(val)
        # 2. Format this unsigned 8-bit value as a two-digit hexadecimal string.
        f.write(f"{unsigned_val_for_hex:02x}\n")

print(f"✅ Image converted to {output_path} with size {image_size} ({'grayscale' if grayscale else 'RGB'})")
print(f"The .mem file now contains 8-bit two's complement signed hexadecimal values.")
