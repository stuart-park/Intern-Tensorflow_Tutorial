import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers


class data_preprocessor:
    def __init__(self, class_names,
                 batch_size=32,
                 buffer_size=1000,
                 img_width=180,
                 img_height=180):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.class_names = class_names
        self.img_width = img_width
        self.img_height = img_height
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def _process_path(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names
        label = tf.argmax(one_hot)

        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.img_width, self.img_height])

        return img, label

    def _convert_dataset(self, ds):
        converted_ds = ds.map(
            self._process_path, num_parallel_calls=self.AUTOTUNE)

        return converted_ds

    def _configure_data(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def _rescale_dataset(self, ds):
        normalization_layer = layers.experimental.preprocessing.Rescaling(
            1./255)
        normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))

        return normalized_ds

    def build_data(self, ds):
        convert_ds = self._convert_dataset(ds)
        configure_ds = self._configure_data(convert_ds)
        final_ds = self._rescale_dataset(configure_ds)

        return final_ds
