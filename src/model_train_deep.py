import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load parameters from param.yaml
import yaml

with open("deep_params.yaml", "r") as f:
    params = yaml.safe_load(f)

img_width, img_height = params["model"]["image_size"]
epochs = params["model"]["epochs"]
batch_size = params["img_augment"]["batch_size"]

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    params["model"]["train_path"],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    params["model"]["test_path"],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Building the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer=params["model"]["optimizer"],
              loss=params["model"]["loss"],
              metrics=params["model"]["metrics"])

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=test_generator)

# Save the model
model.save('trained1.h5')