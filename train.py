import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)

# Mapping kode -> label
label_map = {
    "01": "Akhal-Teke",
    "02": "Appaloosa",
    "03": "Orlov Trotter",
    "04": "Vladimir Heavy Draft",
    "05": "Percheron",
    "06": "Arabian",
    "07": "Friesian"
}

# Load dataset
images = []
labels = []

for filename in os.listdir(DATASET_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):

        label_code = filename.split("_")[0]
        label_index = list(label_map.keys()).index(label_code)

        filepath = os.path.join(DATASET_DIR, filename)
        img = load_img(filepath, target_size=IMG_SIZE)
        img = img_to_array(img) / 255.0

        images.append(img)
        labels.append(label_index)

X = np.array(images)
y = np.array(labels)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

model.save("horse_model.h5")

print("Model trained & saved!")
