import tensorflow as tf

from tensorflow import keras


train_acc_metric = keras.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step(model, image, label, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        logits = model(image, training=True)
        loss_value = loss_fn(label, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(label, logits)

    return loss_value


def train_one_epoch(ds, model, batch_size, optimizer, loss_fn):
    for step, (image, label) in enumerate(ds):
        loss_value = train_step(model, image, label, optimizer, loss_fn)

        if step % 10 == 0:
            print("Training loss (for one batch) at step %d: %.4f" %
                  (step, float(loss_value)))
            print("Seen so far: %d samples" % ((step+1)*batch_size))

        train_acc = train_acc_metric.result()
        train_acc_metric.reset_states()

    print("Training acc over epoch: %.4f" % (float(train_acc),))
