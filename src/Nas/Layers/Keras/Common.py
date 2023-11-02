import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D


class Id(keras.layers.Layer):
    def __init__(self, ch_in, ch_out, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        if ch_in != ch_out:
            self._conv = Conv2D(ch_out, (1, 1))
    
    def call(self, x):
        if self._conv:
            return self._conv(x)
        return x