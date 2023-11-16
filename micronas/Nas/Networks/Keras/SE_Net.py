import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, GlobalAveragePooling2D

from micronas.Nas.Layers.Keras.SqueezeExtract import SqueezeExtraction
from micronas.Nas.Layers.Keras.ChannelWiseConv import ChannelWiseConv

from micronas.Nas.Layers.Keras.ProfiledOp import profiledOp


class Keras_SE_Net(tf.keras.Model):
    def __init__(self, nb_conv_layers, nb_conv_reduce, filter_num, filter_size, num_classes, weights=None, getPruned=False, keepIdx=1) -> None:
        super().__init__()

        self._getPruned = getPruned
        self._nas_weights = weights
        if weights is not None:
            assert weights.shape[0] == nb_conv_reduce

        self._nb_conv_layers = nb_conv_layers
        self._num_classes = num_classes
        self._filter_size = filter_size
        self._conv_modules = keras.Sequential()
        self._se_blocks = keras.Sequential()
        self._softmax = profiledOp(keras.layers.Softmax())

        self._conv1 = profiledOp(Conv2D(self._num_classes, (1, 1)))
        self._gap = profiledOp(GlobalAveragePooling2D())
        self._dropout = Dropout(0.2)

        for i in range(nb_conv_layers):
            self._conv_modules.add(ChannelWiseConv(filter_num, weights=self._nas_weights[i] if self._nas_weights is not None else None, getPruned=self._getPruned))
            self._conv_modules.add(profiledOp(BatchNormalization()))

        for _ in range(nb_conv_reduce):
            self._se_blocks.add(SqueezeExtraction(filter_num, 4))
            self._se_blocks.add(profiledOp(Conv2D(filter_num, (1, 3), (1, 1), activation="relu")))
            self._se_blocks.add(profiledOp(BatchNormalization()))

    def call(self, inputs):
        x = self._conv_modules(inputs)
        x = self._dropout(x)
        x = self._se_blocks(x)
        x = self._conv1(x)
        x = self._gap(x)
        x = self._softmax(x)
        return x
    
    def get_config(self):
        return {
            "_nb_conv_layers": self._nb_conv_layers,
            "_num_classes": self._num_classes,
            "_filter_size": self._filter_size,
            "_conv_modules": self._conv_modules.get_config(),
            "_se_blocks": self._se_blocks.get_config(),
            "_softmax": self._softmax.get_config(),
            "_conv1": self._conv1.get_config(),
            "_gap": self._gap.get_config(),
            "_dropout": self._dropout.get_config()
        }

    @classmethod
    def from_config(cls, config):
        se_net = cls()
        se_net._nb_conv_layers = config["_nb_conv_layers"]
        se_net._num_classes = config["_num_classes"]
        se_net._filter_size = config["_filter_size"]
        se_net._conv_modules = config["_conv_modules"].from_config()
        se_net._se_blocks = config["_se_blocks"].from_config()
        se_net._softmax = config["_softmax"].from_config()
        se_net._conv1 = config["_conv1"].from_config()
        se_net._gap = config["_gap"].from_config()
        se_net._dropout = config["_dropout"].from_config()
        return se_net