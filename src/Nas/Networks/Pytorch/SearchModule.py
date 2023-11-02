from random import Random
import torch
import torch.nn as nn
from enum import Enum

class InferenceType(Enum):
    NORMAL = 0
    MIN = 1
    MAX = 2
    CUSTOM = 3
    SAMPLE = 4
    MAX_WEIGHT = 5
    RANDOM = 6

class NAS_Module(nn.Module):
    def __init__(self):
        super().__init__()

        self._weights = []
        self._t = 1
    
    def sub_t(self, sub):
        self._t = max(0.00001, self._t - sub)

    # def get_nas_weights(self):
    #     raise NotImplementedError("get_nas_weights is not implemented")

    def forward(self, x):
        raise NotImplementedError()

    def getKeras(self, x):
        raise NotImplementedError("getKeras is not implemented")