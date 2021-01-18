import tensorflow as tf

from functools import partial
from albumentations import (Compose, Transpose, HorizontalFlip,
                            VerticalFlip, RandomRotate90, RandomBrightnessContrast)

augmentate = Compose([
    Transpose(p=0.6),
    HorizontalFlip(p=0.6),
    VerticalFlip(p=0.6),
    RandomRotate90(p=0.6),
    RandomBrightnessContrast(p=0.6)
])


class augmentate_data():
    def __init__(self, ds, img_size):
        self.ds = ds
        self.input_shape=(img_size, img_size, 3)

    def _aug_fn(self, image):
        data = {"image": image}
        aug_data = augmentate(**data)
        aug_img = aug_data["image"]

        return aug_img

    def _process_img(self, image, label):
        aug_img = tf.numpy_function(
            func=self._aug_fn, inp=[image], Tout=tf.float32)

        return aug_img, label

    def _set_shapes(self, img, label):
        img.set_shape((self.input_shape))
        label.set_shape([])

        return img, label

    def data_aug(self):
        ds = self.ds.map(self._process_img,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = self.ds.map(self._set_shapes,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds


"""
def data_augmentation(ds):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    data_aug = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2)
    ])

    aug_data = ds.map(lambda x, y: (data_aug(x), y),
                      num_parallel_calls=AUTOTUNE)
    
    return aug_data
"""
