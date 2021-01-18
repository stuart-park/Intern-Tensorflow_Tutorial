import tensorflow as tf


class get_test_data():
    def __init__(self, test_csv, base_dir, img_size):
        self.test_csv = test_csv
        self.base_dir = base_dir
        self.test_num = len(list(base_dir.glob("test_images/*.jpg")))
        self.img_size = img_size
        
    def _create_dataset(self):
        self.test_csv["file_path"] = self.base_dir + \
            "/test_images/"+self.test_csv["image_id"]

        test_ds = tf.data.Dataset.from_tensor_slices(
            self.test_csv["file_path"])

        return test_ds

    def _resize_img(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.img_size, self.img_size])

        return img

    def generate_data(self):
        AUTOTUNE=tf.data.experimental.AUTOTUNE
        
        test_ds = self._create_dataset()
        test_ds = test_ds.map(
            self._resize_img, num_parallel_calls=AUTOTUNE)
        
        test_ds=test_ds.batch(self.test_num)
        test_ds=test_ds.prefetch(buffer_size=AUTOTUNE)
        
        return test_ds
