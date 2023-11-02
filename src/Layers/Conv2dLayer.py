from distutils.command.build import build
from distutils.command.config import config
from unicodedata import name
from tensorflow.keras.layers import Conv2D
import numpy as np

from src.Layers.BaseLayer import BaseLayer, Layer


class Conv2Dlayer(BaseLayer):

    def __init__(self, num_kernels=None, filterSizeX=None, filterSizeY=None) -> None:
        self.type = Layer.CONV2D
        self._num_kernels = num_kernels
        self._filterSizeX = filterSizeX
        self._filterSizeY = filterSizeY
        self._activation = 'relu'  # This is fixed for now
        self._useBias = True

        self._inputSizeX = None
        self._inputSizeY = None
        self._inputSizeC = None

        self._outputSizeX = None
        self._outputSizeY = None

    @property
    def config(self):
        return {"name": "Conv2D", "input": (self._inputSizeX, self._inputSizeY, self._inputSizeC), "output": (self._outputSizeX, self._outputSizeY, self._num_kernels), "filter": (self._filterSizeX, self._filterSizeY, self._num_kernels), "activation": self._activation}

    def build(self, inputSize):
        self._inputSizeX, self._inputSizeY, self._inputSizeC = inputSize
        self._outputSizeX = (self._inputSizeX - self._filterSizeX) + 1
        self._outputSizeY = (self._inputSizeY - self._filterSizeY) + 1
        return (self._outputSizeX, self._outputSizeY, self._num_kernels)

    def getKerasLayer(self):
        return Conv2D(self._num_kernels, (self._filterSizeX, self._filterSizeY), activation=self._activation)

    def configureRandom(self, cnnRandomConfig, inputSize):
        self._inputSizeX, self._inputSizeY, self._inputSizeC = inputSize
        minKSize, maxKSize, minNumK, maxNumK = cnnRandomConfig

        self._filterSizeX = np.random.randint(
            minKSize, min(maxKSize, self._inputSizeX) + 1)
        self._filterSizeY = np.random.randint(
            minKSize, min(maxKSize, self._inputSizeY) + 1)
        self._num_kernels = np.random.randint(minNumK, maxNumK + 1)

        self.build(inputSize)
        return self, (self._outputSizeX, self._outputSizeY, self._num_kernels)

    def configureRandom1x1CNN(self, cnnRandomConfig, inputSize):
        minKSize, maxKSize, minNumK, maxNumK = cnnRandomConfig
        return self.configureRandom((1, 1, minNumK, maxNumK), inputSize)

    def getLatency(self):

        work = self._num_kernels * self._outputSizeX * self._outputSizeY * \
            self._num_kernels * self._inputSizeX * self._inputSizeY * self._inputSizeC

        loads, compute = 2 * work, work
        if self._useBias:
            loads += self._num_kernels * self._outputSizeX * \
                self._outputSizeY * self._num_kernels

        return loads

    def toString(self):
        return f"Conv2D :: InputShape: {(self._inputSizeX, self._inputSizeY, self._inputSizeC)}, kernelSize: {(self._filterSizeX, self._filterSizeY)}, outputShape: {(self._outputSizeX, self._outputSizeY, self._num_kernels)}"
