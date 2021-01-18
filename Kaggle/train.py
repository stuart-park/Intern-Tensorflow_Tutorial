import models
import data_augmentation
import data_preprocessor
import generate_train_data
import pathlib
import argparse
import yaml
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_arges():
    parser = argparse.ArgumentParser(description="define hyper params")

    parser.add_argument("--yaml",
                        type=str,
                        required=True,
                        help="hyper param values")

    config = parser.parse_args()

    return config


def main():
    base_dir = "C:/Users/hnn02/cassava-leaf-disease-classification"

    args = parse_arges()

    with open(args.yaml) as file:
        hyper_params = yaml.load(file, Loader=yaml.FullLoader)

    preprocess_params = hyper_params["PreProcess"]
    train_params = hyper_params["Train"]

    # Read CSV file
    train_csv = pd.read_csv(base_dir+"/train.csv")
    train_csv["file_path"] = base_dir + \
        "/train_images/"+train_csv["image_id"]

    # Generate Data
    train_ds, val_ds = generate_train_data.get_train_data(
        train_csv,
        pathlib.Path(base_dir),
        preprocess_params["batch_size"],
        preprocess_params["img_size"],
        preprocess_params["val_ratio"],
        preprocess_params["buffer_size"]).generate_data()

    # Preprocess Data
    train_ds = data_preprocessor.rescale_dataset(train_ds)
    val_ds = data_preprocessor.rescale_dataset(val_ds)

    # Augmentate Data
    train_ds = data_augmentation.augmentate_data(
        train_ds, preprocess_params["img_size"]).data_aug()

    train_ds = train_ds.shuffle(buffer_size=1000).batch(
        preprocess_params["batch_size"]).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.shuffle(buffer_size=1000).batch(
        preprocess_params["batch_size"]).prefetch(buffer_size=AUTOTUNE)

    # Build model
    model = models.vgg16(
        5, preprocess_params["img_size"])
    model.summary()

    # Train
    epochs = train_params["epochs"]

    Adam = keras.optimizers.Adam(learning_rate=float(train_params["lr"]))

    model.compile(optimizer=Adam,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False),
                  metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)


if __name__ == "__main__":
    main()
