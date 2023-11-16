import tensorflow as tf
import numpy as np
import abc

from micronas.Utils.Tensor import Tensor

class BaseNetwork(metaclass=abc.ABCMeta):

    def __init__(self, inputShape, numClasses):
        self.graph = []
        self.tensors_tmp = [] # Tensors needed for execution on the MCU
        self.inputShape = inputShape
        self.numClasses = numClasses

    @property
    def config(self):
        return [{**{"id": i}, **x.config} for i, x in enumerate(self.graph)]

    @abc.abstractclassmethod
    def toKerasModel(self):
        pass

    def build(self):
        previousShape = self.inputShape
        for layer in self.graph:
            previousShape = layer.build(previousShape)

    def addtmpTensor(self, tensor: Tensor):
        self.tensors_tmp.append(tensor)

    @property
    def memoryRequirements(self):
        return sum([tensor.size for tensor in self.tensors_tmp])

    def printSummary(self):
        for layer in self.graph:
            print(layer.toString())
            
    def addLayer(self, layer):
        self.graph.append(layer)
