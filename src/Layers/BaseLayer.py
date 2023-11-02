from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum

class Layer(Enum):
    CONV2D, MAXPOOL2D, DENSE = range(3)

class BaseLayer(metaclass=ABCMeta):

    def __init__(self) -> None:
        self.type

    @abstractproperty
    def config(self):
        pass

    @abstractmethod
    def getKerasLayer(self):
        pass

    @abstractmethod
    def getLatency(self):
        pass

    @abstractmethod
    def toString(self):
        pass