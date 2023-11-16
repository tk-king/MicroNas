from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from micronas.Nas.Layers.Pytorch.Common import Dyn_Add

from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType, NAS_Module

from micronas.Nas.Utils import random_one_hot_like, weight_softmax
from micronas.config import Config

from torch.autograd import Variable


import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

import numpy as np


class MakeChoice(NAS_Module):
    def __init__(self, num_choices, alsoWeights):
        super().__init__()

        self._num_choices = num_choices
        self._also_weights = alsoWeights
        self._weights = Variable(1e-3 * torch.randn(num_choices, dtype=float, device=Config.device), requires_grad=True)
        self._weights_last = None

    def get_nas_weights(self):
        return [self._weights]

    def forward(self, x, eps, inf_type=InferenceType.NORMAL, weights=[]):
        assert isinstance(x, list), "Input needs to be a list"
        assert len(x) == self._num_choices, "Wrong number of inputs"
        if self._also_weights:
            assert(len(x) == len(weights)), "Weights need to be same length as inputs"

        weights_softmax = weight_softmax(self._weights, eps)
        if inf_type != InferenceType.NORMAL:
            weights_softmax = torch.zeros_like(self._weights)
        if inf_type == InferenceType.MIN:
            weights_softmax[0] = 1
        if inf_type == InferenceType.MAX:
            weights_softmax[-1] = 1
        if inf_type == InferenceType.SAMPLE:
            self._weights_last = weight_softmax(self._weights, eps, hard=True)
            weights_softmax = self._weights_last
        if inf_type == InferenceType.RANDOM:
            self._weights = random_one_hot_like(self._weights)
            weights_softmax = self._weights
        if inf_type == InferenceType.MAX_WEIGHT:
            weights_softmax = weight_softmax(self._weights, eps, gumbel=False)
        


        out_stack = []
        lat_acc = torch.tensor(0, dtype=float)
        mem_acc = torch.tensor(0, dtype=float)
        weight_stack = []

        for i, (x_in, w)  in enumerate(zip(x, weights_softmax)):
            out, lat, mem = x_in
            out_stack.append(out * w)
            lat_acc += lat * w
            mem_acc += mem * w
            weight_stack.append(weights[i] * w)
        
        scaled_weights =torch.sum(torch.stack(weight_stack), dim=0)
        out = torch.sum(torch.stack(out_stack), dim=0)
        if self._also_weights:
            return out, lat_acc, mem_acc, scaled_weights
        return out, lat_acc, mem_acc
    
    def print_nas_weights(self, eps, gumbel, raw):
        print("Make_Choice: ", np.around(weight_softmax(self._weights, eps, gumbel=gumbel).cpu().detach().numpy(), 2))
        # print("Make_cHoice_last: ", self._weights_last)
            
    def getKeras(self, x, getPruned, inf_type):
        weights_softmax = weight_softmax(self._weights, 1e-9, gumbel=False)
        if inf_type != InferenceType.NORMAL:
            weights_softmax = torch.zeros_like(self._weights)
        if inf_type == InferenceType.MIN:
            weights_softmax[0] = 1
        if inf_type == InferenceType.MAX:
            weights_softmax[-1] = 1
        if inf_type == InferenceType.SAMPLE or inf_type == InferenceType.RANDOM:
            weights_softmax = self._weights_last
        if inf_type == InferenceType.MAX_WEIGHT:
            weights_softmax = weight_softmax(self._weights, 1e-9, gumbel=False)


        if getPruned:
            maxIdx = torch.argmax(weights_softmax)
            return x[maxIdx]
        else:
            return L.add(x)

# For residual blocks
class Parallel_Choice_Add(NAS_Module):
    def __init__(self, layers, channels, granularity):
        super().__init__()

        self._layers = nn.ModuleList(layers)
        self._num_channels = channels
        self._granularity = granularity
        # assert(layers[0]._get_name() == "Zero"), "First operation needs to be zero"

        self._num_choices = len(layers) + 1 # +1 for hte zero 

        self._add = Dyn_Add(channels, channels, granularity=granularity)

        # self._bn = nn.BatchNorm2d(channels)

        self._weights_choice = Variable(1e-3 * torch.randn(self._num_choices, dtype=float, device=Config.device), requires_grad=True)
        self._weights_choice_last = None

    def get_nas_weights(self):
        return [self._weights_choice]

    def forward(self, x1, x2, prob_no_op, eps, weights, last_ch_weights, inf_type=InferenceType.NORMAL):
        weights_choice_softmax = weight_softmax(self._weights_choice, eps)
        assert(abs(weights.sum() - 1) < 1e-5)
        if last_ch_weights is not None:
            assert(abs(last_ch_weights.sum() - 1) < 1e-5)

        if inf_type != InferenceType.NORMAL:
            weights_choice_softmax = torch.zeros_like(self._weights_choice)
        if inf_type == InferenceType.MIN:
            weights_choice_softmax[0] = 1
        if inf_type == InferenceType.MAX:
            weights_choice_softmax[-1] = 1
        if inf_type == InferenceType.SAMPLE:
            self._weights_choice_last = weight_softmax(self._weights_choice, eps, hard=True)
            weights_choice_softmax = self._weights_choice_last
        if inf_type == InferenceType.RANDOM:
            self._weights_choice = random_one_hot_like(self._weights_choice)
            weights_choice_softmax = self._weights_choice
        if inf_type == InferenceType.MAX_WEIGHT:
            weights_choice_softmax = weight_softmax(self._weights_choice, eps, gumbel=False)

        out_stack = []
        x1_in, lat_acc, mem_acc = x1 # Through the layers
        x2_in, lat2_acc, mem2_acc = x2

        B, C, H, W = x1_in.shape
        if last_ch_weights is not None:
            input_list = [np.prod([1, (i + 1) * self._granularity, H, W]) for i in range(self._num_channels // self._granularity)]
            input_mem = torch.sum(torch.tensor([in_mem * w for in_mem, w in zip(input_list, last_ch_weights)]))
        else:
            input_mem = C * H * W

        # print(len(self._layers), len(weights_choice_softmax[1:]))
        for layer, w  in zip(self._layers, weights_choice_softmax[1:]):
            out, lat, mem = layer(x1_in, eps, inf_type=inf_type, last_ch_weights=last_ch_weights, weights=weights)
            # print(layer._get_name(), w)
            out_stack.append(out * w)
            lat_acc += lat * w
            # print("Res_lat_layer: ", layer._get_name(), lat, mem, w)
            mem_acc += mem * w
        

        layers_out = torch.sum(torch.stack(out_stack), dim=0) if len(out_stack) > 1 else out_stack[0]
        
        mem_acc = torch.max(mem_acc, mem2_acc)
        out_add, out_lat, out_mem = self._add(layers_out, x2_in, weights=weights)
        # out_add = self._bn(out_add)
        lat_acc += out_lat * torch.max(weights_choice_softmax[1:]) * (1 - prob_no_op)
        mem_acc = torch.max(mem_acc, (out_mem + input_mem) * torch.max(weights_choice_softmax[1:]) * (1- prob_no_op))

        lat_acc += lat2_acc


        return out_add, lat_acc, mem_acc
    
    def print_nas_weights(self, eps, gumbel, raw):
        print("Parallel_Choice_Add: ", np.around(weight_softmax(self._weights_choice, eps, gumbel=gumbel).cpu().detach().numpy(), 2))
        # print("Parallle_Choice_Add_last: ", self._weights_choice_last)
            
    def getKeras(self, x1, x2, getPruned, weights, inf_type=InferenceType.NORMAL):
        weights_choice_softmax = weight_softmax(self._weights_choice, 1e-9, gumbel=False)
        weights_ch_choose = weights
        if inf_type != InferenceType.NORMAL:
            weights_choice_softmax = torch.zeros_like(self._weights_choice)
        if inf_type == InferenceType.MIN:
            weights_choice_softmax[0] = 1
        if inf_type == InferenceType.MAX:
            weights_choice_softmax[-1] = 1
        if inf_type == InferenceType.SAMPLE or inf_type == InferenceType.RANDOM:
            weights_choice_softmax = self._weights_choice_last
        if inf_type == InferenceType.MAX_WEIGHT:
            weights_choice_softmax = weight_softmax(self._weights_choice, 1e-9, gumbel=False)

        if getPruned:
            maxIdx = torch.argmax(weights_choice_softmax)
            if maxIdx == 0:
                return x2
            last_layer = self._layers[maxIdx - 1].getKeras(x1, getPruned=True, weights=weights_ch_choose)
            if last_layer is not None:
                print("last_layer_form x1: ", last_layer.shape)
            if last_layer is None:
                print("Return x2")
                return x2
            if x2 is None:
                print("Return last")
                return last_layer
            print("Return add")
            return L.add([last_layer, x2])
        else:
            add_list = [l.getKeras(x1, getPruned=False, weights=weights_ch_choose) for l in self._layers] + [x2]
            add_list = list(filter(lambda x: x is not None, add_list))
            return L.add(add_list)