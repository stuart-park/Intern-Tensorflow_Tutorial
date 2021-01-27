import tensorflow as tf
import time
import yaml

from tensorflow import keras
from tensorflow.keras import layers

import data_augmentation
import data_preprocessing
import data_generator
import models
#import train_func
import train_class
import validate
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="define hyperparameter")

    parser.add_argument('--yaml_file',
                        type=str,
                        required=True,
                        help="get hyperparameter values")

    """
    parser.add_argument('--val_ratio', 
                        required=True, 
                        type=float, 
                        help='validation ratio')
    parser.add_argument('--epochs', 
                        required=True, 
                        type=int, 
                        help="Define training epochs")
    parser.add_argument('--batch_size',
                        required=True,
                        type=int, 
                        help="Define size of batch")
    parser.add_argument('--buffer_size',
                        required=True,
                        type=int, 
                        help="Define size of buffer")
    parser.add_argument('--img_width',
                        required=True,
                        type=int,
                        help="Define Image width")
    parser.add_argument('--img_height',
                        required=True,
                        type=int,
                        help="Define Image height")
    parser.add_argument('--lr',
                        required=True, 
                        type=float,
                        help="learning rate")
    """

    config = parser.parse_args()

    return config


def main():

    ### Parse hyperparameter ###
    args = parse_args()

    with open(args.yaml_file) as file:
        hyper_param = yaml.load(file, Loader=yaml.FullLoader)

    preprocess_param = hyper_param["PreProcess"]
    train_param = hyper_param['Train']

    ### Get Data ###
    data_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    val_ratio = preprocess_param['val_ratio']

    train_ds, val_ds, class_names = data_generator.data_generator(
        data_url, val_ratio).generate_data()

    print("Cardinality of Train Dataset: %d" %
          (tf.data.experimental.cardinality(train_ds).numpy(), ))
    print("Cardinality of Validation Dataset: %d" %
          (tf.data.experimental.cardinality(val_ds).numpy(), ))

    ### Preprocess Data ###
    preprocess = data_preprocessing.data_preprocessor(class_names,
                                                      preprocess_param['batch_size'],
                                                      preprocess_param['buffer_size'],
                                                      preprocess_param['img_width'],
                                                      preprocess_param['img_height'])
    train_ds = preprocess.build_data(train_ds)
    val_ds = preprocess.build_data(val_ds)

    ### Data augmentation ###
    train_ds = data_augmentation.augmentate_data(train_ds)

    ### Build model ###
    num_classes = len(class_names)

    """
    # Load Basic Model
    model = models.basic_model(num_classes)
    """
    """
    # Load Model with class
    model = models.build_model(num_classes, preprocess_param['img_width'],
                                 preprocess_param['img_height']).vgg16()
    """

    model = models.vgg16(
        num_classes, preprocess_param["img_width"], preprocess_param["img_height"])
    model.summary()

    ### Train and Validate model ###
    epochs = train_param['epochs']

    # Train model by fit
    adam = keras.optimizers.Adam(learning_rate=float(train_param['lr']))
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    """
    # Train model by class
    model_train = train_class.train_model(build_model,
                                          preprocess_param['batch_size'],
                                          float(train_param['lr']))

    for epoch in range(epochs):
        print("\nStart of epoch %d " % (epoch, ))
        start_time = time.time()
        model_train.train_one_epoch(aug_train_ds)
        validate.validate_one_epoch(val_ds, build_model)
        print("Time taken %.2fs" % (time.time()-start_time))
    
    """

    """
    #Trian model by function
    
    optimizer=keras.optimizers.Adam(learning_rate=float(train_param['lr']))
    loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for epoch in range(epochs):
        print("\nStart of epoch %d " % (epoch, ))
        start_time=time.time()
        train_func.train_one_epoch(aug_train_ds, build_model, preprocess_param['batch_size'], optimizer, loss_fn)
        validate.validate_one_epoch(val_ds, build_model)
        print("Time taken %.2fs" % (time.time()-start_time))
    """


if __name__ == "__main__":
    main()
