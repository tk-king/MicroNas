from distutils.command.build import build
from unicodedata import name
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
import numpy as np

from src.Layers.BaseLayer import BaseLayer, Layer


class DepthWiseCnnLayer(BaseLayer):

    def __init__(self, filterSizeX=None, filterSizeY=None) -> None:
        self.type = Layer.CONV2D
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
        return {"name": "DepthWiseConv2D", "input": (self._inputSizeX, self._inputSizeY, self._inputSizeC), "output": (self._outputSizeX, self._outputSizeY, self._inputSizeC), "filter": (self._filterSizeX, self._filterSizeY), "activation": self._activation}

    def build(self, inputSize):
        self._inputSizeX, self._inputSizeY, self._inputSizeC = inputSize
        self._outputSizeX = (self._inputSizeX - self._filterSizeX) + 1
        self._outputSizeY = (self._inputSizeY - self._filterSizeY) + 1
        return (self._outputSizeX, self._outputSizeY, self._inputSizeC)

    def getKerasLayer(self):
        return DepthwiseConv2D((self._filterSizeX, self._filterSizeY), activation='relu')

    def configureRandom(self, cnnRandomConfig, inputSize):
        self._inputSizeX, self._inputSizeY, self._inputSizeC = inputSize
        minKSize, maxKSize, minNumK, maxNumK = cnnRandomConfig

        self._filterSizeX = np.random.randint(
            minKSize, min(maxKSize, self._inputSizeX) + 1)
        self._filterSizeY = np.random.randint(
            minKSize, min(maxKSize, self._inputSizeY) + 1)

        self.build(inputSize)
        return self, (self._outputSizeX, self._outputSizeY, self._inputSizeC)

    def getLatency(self):
        pass

    def toString(self):
        return f"DepthWiseConv2D :: InputShape: {(self._inputSizeX, self._inputSizeY, self._inputSizeC)}, kernelSize: {(self._filterSizeX, self._filterSizeY)}, outputShape: {(self._outputSizeX, self._outputSizeY, self._inputSizeC)}"
