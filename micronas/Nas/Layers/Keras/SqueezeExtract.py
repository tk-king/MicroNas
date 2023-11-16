import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Multiply, Add

from micronas.Nas.Layers.Keras.ProfiledOp import profiledOp

class SqueezeExtraction(tf.keras.Model):
    def __init__(self, ch, ratio=16) -> None:
        super().__init__()

        self._cse = ChannelSqueezeExtraction(ch, ratio)
        self._sse = SpacialSqueezeExtraction(ch, ratio)
        self._add = profiledOp(Add())


    def call(self, inputs):
        cse = self._cse(inputs)
        sse = self._sse(inputs)
        return self._add([cse, sse])

    def get_config(self):
        return {
            "_cse": self._cse.get_config(),
            "_sse": self._sse.get_config()
        }
    
    @classmethod
    def from_config(cls, config):
        se = cls()
        se._cse = ChannelSqueezeExtraction.from_config(config["_cse"])
        se._sse = SpacialSqueezeExtraction.from_config(config["_sse"])
        return se
        


class ChannelSqueezeExtraction(tf.keras.Model):
    def __init__(self, ch, ratio) -> None:
        super().__init__()

        self._gap = profiledOp(GlobalAveragePooling2D())
        self._dense_001 = profiledOp(Dense(ch//ratio, activation="relu"))
        self._dense_002 = profiledOp(Dense(ch, activation='sigmoid'))
        # Manually add sigmoid to profiled ops
        self._multiply = profiledOp(Multiply())

    def call(self, inputs):
        x = self._gap(inputs)
        x = self._dense_001(x)
        x = self._dense_002(x)
        return self._multiply([inputs, x])

    def get_config(self):
        return {
            "gap": self._gap.get_config(),
            "dense_001": self._dense_001.get_config(),
            "dense_002": self._dense_002.get_config()
        }

    @classmethod
    def from_config(cls, config):
        cse = cls()
        cse._gap = GlobalAveragePooling2D.from_config(config["gap"])
        cse._dense_001 = Dense.from_config(config["dense_001"])
        cse._dense_002 = Dense.from_config(config["dense_002"])
        return cse





class SpacialSqueezeExtraction(tf.keras.Model):
    def __init__(self, ch, ratio) -> None:
        super().__init__()
        self._conv = profiledOp(Conv2D(1, (1, 1), activation="sigmoid", kernel_initializer="he_normal"))
        self._multiply = profiledOp(Multiply())


    def call(self, inputs):
        x = self._conv(inputs)
        return self._multiply([inputs, x])
    
    def get_config(self):
        return {
            "conv": self._conv.get_config()
        }
    
    @classmethod
    def from_config(cls, config):
        sse = cls()
        sse._conv = Conv2D.from_config(config["conv"])