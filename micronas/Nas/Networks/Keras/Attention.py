import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Dropout
import tensorflow.keras.backend as K

from micronas.Nas.Layers.Keras.SelfAttention import KerasSelfAttention

class Keras_SelfAttention(tf.keras.Model):
    def __init__(self, nb_conv_layers, filter_num, filter_size, num_classes, add_instead_of_concat=False) -> None:
        super(Keras_SelfAttention, self).__init__()

        self.conv_modules = keras.Sequential()
        self.nb_conv_layers = nb_conv_layers
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.add_instead_of_concat = add_instead_of_concat

        for _ in range(nb_conv_layers):
            self.conv_modules.add(
                Conv2D(filter_num, (filter_size, 1), (2, 1), activation="relu"))
            self.conv_modules.add(BatchNormalization())

        self.att = KerasSelfAttention(filter_num)

        self.dropout = Dropout(0.2)

        self.dense_modules = keras.Sequential()
        self.dense_modules.add(Dense(16, activation="relu"))
        self.dense_modules.add(Flatten())
        self.dense_modules.add(Dense(num_classes, activation="softmax"))

    def call(self, x):
        x = self.conv_modules(x)
        t_list = []
        for t in tf.unstack(x, axis=1):
            t = K.expand_dims(t, axis=1)
            t_list.append(self.att(t))
        x = K.concatenate(t_list, axis=1)
        x_shape = K.shape(x)
        x = K.reshape(x, (-1, x_shape[1], x_shape[2] * x_shape[3]))
        x = self.dropout(x)
        res = self.dense_modules(x)
        return res