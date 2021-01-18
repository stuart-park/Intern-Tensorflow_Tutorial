import tensorflow as tf


class get_train_data():
    def __init__(self,
                 csv_file,
                 base_dir,
                 batch_size=32,
                 img_size=224,
                 val_ratio=0.3,
                 buffer_size=None):
        self.batch_size = batch_size
        self.base_dir = base_dir
        self.train_csv = csv_file
        self.img_size = img_size
        self.img_num = len(list(base_dir.glob("train_images/*.jpg")))
        self.val_ratio = val_ratio
        self.buffer_size = buffer_size
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def _create_dataset(self):
        list_ds = tf.data.Dataset.from_tensor_slices(
            (self.train_csv["file_path"], self.train_csv["label"]))

        if self.buffer_size == None:
            list_ds = list_ds.shuffle(
                self.img_num, reshuffle_each_iteration=False)
        else:
            list_ds = list_ds.shuffle(
                self.buffer_size, reshuffle_each_iteration=False)

        return list_ds

    def _train_val_split(self, list_ds):
        val_num = int(self.img_num*self.val_ratio)
        train_ds = list_ds.skip(val_num)
        val_ds = list_ds.take(val_num)

        return train_ds, val_ds

    def _resize_img(self, file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.img_size, self.img_size])

        return img, label

    def generate_data(self):
        list_ds = self._create_dataset()
        train_ds, val_ds = self._train_val_split(list_ds)

        train_ds = train_ds.map(
            self._resize_img, num_parallel_calls=self.AUTOTUNE)
        val_ds = val_ds.map(self._resize_img, num_parallel_calls=self.AUTOTUNE)

        return train_ds, val_ds
