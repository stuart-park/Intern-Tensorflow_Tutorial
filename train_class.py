import tensorflow as tf

from tensorflow import keras

class train_model():
    def __init__(self, model, batch_size, lr):
        self.model=model
        self.batch_size=batch_size
        self.train_acc_metric=keras.metrics.SparseCategoricalAccuracy()
        self.optimizer=keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        
    @tf.function
    def _train_step(self, image, label):
        with tf.GradientTape() as tape:
            logits=self.model(image, training=True)
            loss_value=self.loss_fn(label, logits)
        grads=tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(label, logits)
    
        return loss_value

    def train_one_epoch(self, ds):
        for step, (image, label) in enumerate(ds):
            loss_value=self._train_step(image, label)
        
            if step % 10==0:
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                print("Seen so far: %d samples" % ((step+1)*self.batch_size))
            
            train_acc=self.train_acc_metric.result()
            self.train_acc_metric.reset_states()    
        
        print("Training acc over epoch: %.4f" % (float(train_acc),))