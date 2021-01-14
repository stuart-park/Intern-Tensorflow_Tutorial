import tensorflow

from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import VGG16

def build_basic_model(num_classes):
    model = Sequential([
        layers.Conv2D(16, 3, padding="same", activation="relu",
                      input_shape=(180, 180, 3)),
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

def pretrained_model(num_classes, img_width, img_height):
    model=Sequential()
    pretrained_vgg=VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    pretrained_vgg.trainable=False
    model.add(pretrained_vgg)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


class build_model():
    def __init__(self, num_classes, img_width, img_height):
        self.num_classes = num_classes
        self.params = {'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                       'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
        self.model=Sequential()
        self.img_width = img_width
        self.img_height = img_height

    def _make_layers(self, params):
        input_channel = 3
        self.model.add(layers.Conv2D(64, 3,
                                activation='relu', padding='same', input_shape=(self.img_width, self.img_height, input_channel)))
        for v in params:
            if v == 'M':
                self.model.add(layers.MaxPooling2D(strides=2))
            else:
                self.model.add(layers.Conv2D(v, 3,
                                        activation='relu', padding='same'))
    
    def _make_classifier_layer(self):
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(4096, activation='relu'))
        self.model.add(layers.Dense(4096, activation='relu'))
        self.model.add(layers.Dense(1000, activation='relu'))
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))

    def vgg16(self):
        self._make_layers(self.params["VGG16"])
        self._make_classifier_layer()
        return self.model

    def vgg19(self):
        self._make_layers(self.params["VGG19"])
        self._make_classifier_layer()
        return self.model
