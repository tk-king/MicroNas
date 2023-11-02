import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Softmax

class Keras_Standard(tf.keras.Model):
    def __init__(self, nb_conv_layers, nb_conv_reduce, filter_num, filter_size, num_classes, add_instead_of_concat=False) -> None:
        super(Keras_Standard, self).__init__()

        self.conv_modules = keras.Sequential()
        self.nb_conv_layers = nb_conv_layers
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.add_instead_of_concat = add_instead_of_concat
        self.reduce_conv = keras.Sequential()
        self.softmax = Softmax()

        self.conv1 = Conv2D(self.num_classes, (1, 1))
        self.gap = GlobalAveragePooling2D()

        for _ in range(nb_conv_layers):
            self.conv_modules.add(
                Conv2D(filter_num, (filter_size, 1), (2, 1), activation="relu"))
            self.conv_modules.add(BatchNormalization())

        for _ in range(nb_conv_reduce):
            self.reduce_conv.add(
                Conv2D(filter_num, (1, 4), (1, 1), activation="relu"))
            self.reduce_conv.add(BatchNormalization())
            self.reduce_conv.add(Dropout(0.2))

        self.dropout = Dropout(0.2)

    def call(self, x):
        x = self.conv_modules(x)
        x = self.dropout(x)
        x = self.reduce_conv(x)
        x = self.conv1(x)
        x = self.gap(x)
        x = self.softmax(x)
        return x