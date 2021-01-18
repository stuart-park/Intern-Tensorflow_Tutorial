"""
1. resample unbalanced data
2. normalize data
"""
import tensorflow as tf
from tensorflow.keras import layers


def rescale_dataset(ds):
    norm_layer = layers.experimental.preprocessing.Rescaling(1./255)\

    norm_ds = ds.map(lambda x, y: (norm_layer(x), y),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return norm_ds
