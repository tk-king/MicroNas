import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, add
from micronas.Nas.Layers.Keras.Common import Id
from micronas.Nas.Layers.Keras.SqueezeExtract import SqueezeExtraction
from micronas.Nas.Layers.Keras.ProfiledOp import profiledOp

class MixedOp(keras.layer.Layer):
    def __init__(self, ch_in, ch_out, filter_sizes=[3, 5, 7], padding="same", stride=1) -> None:
        super().__init__()

        self.conv_layers = []

        for f in filter_sizes:
            self.conv_layers.append(profiledOp(Conv2D(ch_out, (f, 1), padding=padding, stride=stride)))
        
        for f in [5, 7, 9]:
            self.conv_layers.append(profiledOp(Conv2D(ch_out, (1, f), padding="same")))
        
        if ch_in == ch_out:
            self.conv_layers.append(Id(ch_in, ch_out))
            self.conv_layers.append(SqueezeExtraction(ch_in, ratio=4))
    
    def get_num_weights(self):
        return len(self.conv_layers)


    def forward(self, x, weights=None, getPruned=False):
        if getPruned:
            max_w = np.argmax(weights)
            return self.conv_layers[max_w](x)
        if weights is None:
            weights = np.ones(len(self.conv_layers), dtype=Config.tensor_dtype)
        
        conv = []
        for (l, w) in zip(self.conv_layers, weights):
            res= l(x)
            conv.append(res * w)

        x = add(conv)
        return x