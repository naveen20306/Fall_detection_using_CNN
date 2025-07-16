import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === STEP 1: Set Parameters ===
dataset_path = 'fall_new/images'  # Path to your dataset
img_height, img_width = 32, 32  # Changed from 64, 64 to 32, 32
batch_size = 32
epochs = 30

# === STEP 2: Create Data Generators (Grayscale) ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',  # Grayscale mode
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',  # Grayscale mode
    class_mode='categorical',
    subset='validation'
)

# === STEP 3: Define CNN Model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),  # input_shape changed to (32, 32, 1)
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: fall and not_fall
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === STEP 4: Train Model ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# === STEP 5: Save Model ===
model.save('fall_detection_32x32_gray.keras') # Changed filename to reflect 32x32
print("✅ Model saved as fall_detection_32x32_gray.keras")
