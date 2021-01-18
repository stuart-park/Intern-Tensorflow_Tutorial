"""
Data Generator 
"""
import numpy as np
import pathlib
import tensorflow as tf

class data_generator:
    def __init__(self, data_url, val_ratio):
        self.data_dir=pathlib.Path(
            tf.keras.utils.get_file(origin=data_url,
                                    fname="flower_photos",
                                    untar=True)
        )
        self.image_count=len(list(self.data_dir.glob("*/*.jpg")))
        self.val_ratio=val_ratio

    def _data_label_split(self):
        list_ds=tf.data.Dataset.list_files(str(self.data_dir/"*/*"), shuffle=False)
        list_ds=list_ds.shuffle(self.image_count, reshuffle_each_iteration=False)
    
        class_names=np.array(sorted([item.name for item in self.data_dir.glob("*") if item.name!="LICENSE.txt"]))
    
        return list_ds, class_names
    
    def _train_val_split(self, list_ds):
        val_size=int(self.image_count*self.val_ratio)
        train_ds=list_ds.skip(val_size)
        val_ds=list_ds.take(val_size)
    
        return train_ds, val_ds

    def generate_data(self):
        list_ds, class_names=self._data_label_split()
        train_ds, val_ds=self._train_val_split(list_ds)
        
        return train_ds, val_ds, class_names
    