from logging import getLevelName
from tensorflow.keras.layers import Dense
import numpy as np

from src.Layers.BaseLayer import BaseLayer, Layer

class DenseLayer(BaseLayer):

    def __init__(self, numNeurons=None) -> None:
        self.type = Layer.DENSE
        self._numNeurons = numNeurons
        self._activation = "relu"
        self._inputLen = None

    @property
    def config(self):
        return {"name": "Dense", "input": self._inputLen, "output": self._numNeurons, "activation": self._activation}


    def build(self, inputShape):
        self._inputLen = np.prod(inputShape)
        return self._numNeurons

    def configureRandom(self, randomConfig, inputSize):
        minNeurons, maxNeurons = randomConfig
        self._numNeurons = np.random.randint(minNeurons, maxNeurons + 1)
        self.build(inputSize)
        return self, self._numNeurons

    def getKerasLayer(self):
        return Dense(self._numNeurons, activation=self._activation)

    def getLatency(self):
        # n, _ = op.output.shape
        # in_dim, out_dim = op.inputs[1].shape
        # work = n * in_dim * out_dim
        # loads, compute = 2 * work, work
        # if op.use_bias:
        #     loads += n * out_dim
        return 0

    def toString(self):
        return f"Dense :: InputShape: {self._inputLen}, Outputshape: {self._numNeurons}"

    