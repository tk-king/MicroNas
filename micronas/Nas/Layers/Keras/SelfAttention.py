import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D
import tensorflow.keras.backend as K

class KerasSelfAttention(keras.layers.Layer):
    def __init__(self, n_channels) -> None:
        super().__init__()

        # conv1d: N L C
        self.n_channels = n_channels
        self.query = Conv1D(self.n_channels, 1)
        self.value = Conv1D(self.n_channels, 1)
        self.key = Conv1D(self.n_channels, 1)

        self.gamma = tf.Variable(0.)

    def call(self, inputs):
        x = K.squeeze(inputs, 1)
        query = self.query(x)
        value = self.value(x)
        key = self.key(x)
        query_new = K.permute_dimensions(query, (0, 2, 1))
        beta_001 = K.batch_dot(key, query_new)
        beta = K.softmax(beta_001)
        o = self.gamma * \
            K.expand_dims(K.batch_dot(beta, value), axis=1) + inputs
        return o
