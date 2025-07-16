import os
import numpy as np
import json
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
model_path = "fall_detection_32x32_gray.keras"  # Updated model path to 32x32 model
output_folder = "mem_output_32x32_signed"       # Separate output folder for 32x32 signed weights
quant_bits = 8  # 8-bit quantization

# === CREATE OUTPUT FOLDER ===
os.makedirs(output_folder, exist_ok=True)

# === LOAD MODEL ===
model = load_model(model_path)

# === QUANTIZATION FUNCTION (signed 8-bit: -128 to 127) ===
def quantize_weights_to_int8(w):
    max_val = np.max(np.abs(w))
    # Scale to the range of -127 to 127 to avoid overflow with -128
    scale = 127 / max_val if max_val != 0 else 1
    # Round and clip to the signed 8-bit range
    w_q = np.clip(np.round(w * scale), -128, 127).astype(np.int8)
    return w_q

# === WRITE TO .mem FILE (2-digit hex) ===
# For signed 8-bit values, we'll convert to unsigned for hex representation (0-255)
# This is common for memory files where each byte is represented as 00-FF.
# When reading, you'll need to interpret these as signed.
def save_mem_file(name, array):
    flat_array = array.flatten()
    with open(os.path.join(output_folder, f"{name}.mem"), 'w') as f:
        for val in flat_array:
            # Convert int8 to its unsigned byte representation for hex output
            # Cast val to a larger integer type (e.g., int) before arithmetic to avoid overflow
            unsigned_val = int(val) % 256
            f.write(f"{unsigned_val:02x}\n")

# === PROCESS EACH LAYER ===
model_summary = {
    "input_shape": str(model.input_shape),   # Expected: (None, 32, 32, 1) for 32x32 model
    "output_shape": str(model.output_shape),
    "layers": []
}

for i, layer in enumerate(model.layers):
    try:
        output_shape = str(layer.output.shape)
    except AttributeError:
        output_shape = "unknown"

    layer_info = {
        "name": layer.name,
        "type": layer.__class__.__name__,
        "output_shape": output_shape,
        "trainable": layer.trainable
    }

    weights = layer.get_weights()
    if weights:
        for j, param in enumerate(weights):
            param_type = "weights" if j % 2 == 0 else "biases"
            file_prefix = f"layer_{i}_{layer.name}_{param_type}"

            param_q = quantize_weights_to_int8(param) # Use the signed quantization function
            save_mem_file(file_prefix, param_q)

            layer_info[param_type] = {
                "shape": str(param.shape),
                "file": f"{file_prefix}.mem"
            }

    model_summary["layers"].append(layer_info)

# === SAVE MODEL SUMMARY ===
with open(os.path.join(output_folder, "model_summary.json"), "w") as f:
    json.dump(model_summary, f, indent=4)

print("✅ Conversion complete. All .mem files (2-digit hex, representing signed 8-bit) are saved in:", output_folder)
