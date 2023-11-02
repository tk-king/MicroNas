from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Nas.Networks.Pytorch.SearchModule import InferenceType, NAS_Module

from src.Nas.Utils import weight_softmax
from src.config import Config

from math import ceil


import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

import numpy as np


class Sequential(NAS_Module):
    def __init__(self, layers):
        super().__init__()

        self._layers = nn.ModuleList(layers)

    def forward(self, x, eps, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        lat_acc = torch.tensor(0, dtype=float).to(Config.device)
        mem_stack = []
        for l in self._layers:
            layer_res = l(x, eps=eps, inf_type=inf_type, last_ch_weights=last_ch_weights)
            x, lat, mem = layer_res[0:3]
            if len(layer_res) == 4:
                last_ch_weights = layer_res[3]
            lat_acc += lat
            mem_stack.append(mem)
        if len(layer_res) == 4:
            return x, lat_acc, torch.max(torch.stack(mem_stack)), last_ch_weights
        return x, lat_acc, torch.max(torch.stack(mem_stack))

    def print_nas_weights(self, eps, gumbel=False, raw=False):
        for l in self._layers:
            if hasattr(l, "get_nas_weights"):
                l.print_nas_weights(eps, gumbel=gumbel, raw=raw);

    def get_nas_weights(self):
        res = []
        for l in self._layers:
            if hasattr(l, "get_nas_weights"):
                res.extend(l.get_nas_weights())
            else:
                print("not included: ", l._get_name())
        return res


    def getKeras(self, x, getPruned, inf_type=InferenceType.NORMAL):
        for l in self._layers:
            x = l.getKeras(x, getPruned=getPruned, inf_type=inf_type)
        return x