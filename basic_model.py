import tensorflow

from tensorflow import keras
from tensorflow.keras import Sequential, layers

def build_model(num_classes):
    model=Sequential([
        layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(180, 180, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    
    return model