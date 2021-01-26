import tensorflow

from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import EfficientNetB3, ResNet50


def efficientnetB3(num_classes, img_size):
    model = Sequential()
    pretrained_b3 = EfficientNetB3(
        weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    model.add(pretrained_b3)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def resnet50(num_classes):
    model = Sequential()
    pretrained_resnet = ResNet50(include_top=False, weights="imagenet",
                                 input_shape=(img_size, img_size, 3))
    model.add(pretrained_resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
