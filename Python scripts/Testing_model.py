import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model('fall_detection_32x32_gray.keras')
print("Model input shape:", model.input_shape)

class_labels = ['fall', 'not_fall']
image_path = 'IMG20250528125255.jpg'

# Read image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image at path: {image_path}")

# Resize to 64x64 if model input is (64, 64, 1)
img = cv2.resize(img, (32, 32))
img = img / 255.0
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
predicted_class = np.argmax(pred, axis=1)[0]
confidence = np.max(pred)

print(f"Prediction: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")
