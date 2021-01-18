"""
Data Augmentation
"""
import tensorflow as tf
from tensorflow.keras import layers

def augmentate_data(ds):
    AUTOTUNE=tf.data.experimental.AUTOTUNE
    
    augmentate=tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    
    augmented_data=ds.map(lambda x, y: (augmentate(x), y), num_parallel_calls=AUTOTUNE)
    
    return augmented_data
    