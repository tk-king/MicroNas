import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import MaxPool2D, Conv2D, add
import numpy as np

from src.Nas.Layers.Keras.ProfiledOp import profiledOp

class ChannelWiseConv(keras.Model):
    def __init__(self, filter_num, filter_sizes=[3, 5, 7, 10, 20], weights=None, getPruned=False, defaultLayer=1, padding="same", stride=1) -> None:
        super().__init__()
        self._default_layer = defaultLayer
        self._nas_weights = weights
        if getPruned and weights is not None:
            assert weights.shape[0] == len(filter_sizes)

        self._getPruned = getPruned
        self._conv_layers = [profiledOp(Conv2D(
            filter_num, (filter_size, 1), (1, 1), activation="relu", padding=padding, stride=stride)) for filter_size in filter_sizes]
        self._reduce = profiledOp(MaxPool2D((2, 1)))
        
        
    def call(self, inputs):
        if self._nas_weights is None and self._getPruned:
            conv = self._conv_layers[self._default_layer](inputs)
            return self._reduce(conv)
        conv = [x(inputs) for x in self._conv_layers]
        if self._nas_weights is not None and not self._getPruned:
            conv = [x * w for x, w in zip(conv, self._nas_weights)]    
        if self._nas_weights is not None and self._getPruned:
            w = np.argmax(self._nas_weights.detach().numpy())
            return self._reduce(conv[w])
        x = add(conv)
        return self._reduce(x)

    def get_config(self):
        config = {"num_layers": len(self._conv_layers)}
        for i, x in enumerate(self._conv_layers):
            config[f"layer_{i}"] =  x.get_config()
        config["reduce"] = self._reduce.get_config()
        return config

    @classmethod
    def from_config(cls, config):
        cwc = cls()
        cwc._conv_layers = []
        for i in range(config["num_layers"]):
            cwc._conv_layers.append(config[f"layer_{i}"].get_config())
        cwc._reduce = config["reduce"].get_config()
            
