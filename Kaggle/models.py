import tensorflow 

from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB6, VGG16, ResNet50

def vgg16(num_classes, img_size):
    model=Sequential()
    pretrained_vgg=VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    pretrained_vgg.trainable=False
    model.add(pretrained_vgg)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def efficientnetB3(num_classes, img_size):
    model=Sequential()
    pretrained_b3=EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    pretrained_b3.trainable=False
    model.add(pretrained_b3)
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
        
    return model