import numpy as np
import tensorflow as tf
from datasets import decode


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, batch_size=32, contig_len=1000):
        self.batch_size = batch_size
        self.x = x
        self.on_epoch_end()
        self.current_index = 0
        self.contig_len = contig_len

    def __len__(self):
        len_ = int(np.floor(len(self.x) / self.batch_size))
        return len_

    def __getitem__(self, index):
        index = self.current_index
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_temp = [self.x[k] for k in indexes]
        # print(f'\nDataGenerator processes {len(indexes)} from {index * self.batch_size}:{(index + 1) * self.batch_size} of index {index}, total {len(self.indexes)}')
        if self.current_index == self.__len__():
            self.current_index = 0
        else:
            self.current_index += 1
        X, y = self.__data_generation(x_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x))

    def __data_generation(self, x_temp):
        X = []
        for i, contig in enumerate(x_temp):
            X.append(decode(contig, contig_len=self.contig_len))
        X = np.array(X)
        X = X.reshape(-1, self.contig_len, 4, 1).astype('float32')
        return X, X
