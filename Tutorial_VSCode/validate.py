import tensorflow as tf

from tensorflow import keras

val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


@tf.function
def test_step(model, image, label):
    val_logits = model(image, training=False)
    val_acc_metric.update_state(label, val_logits)


def validate_one_epoch(ds, model):
    for image_val, label_val in ds:
        test_step(model, image_val, label_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()

    print("Validation acc: %.4f" % (float(val_acc),))
