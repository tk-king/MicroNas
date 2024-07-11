import torch
import torch.nn as nn
import torch.nn.functional as F

from micronas.Nas.Layers.Pytorch.Common import Add, Conv2D, Dyn_Add, Dyn_Conv2D, GlobalAveragePooling, LogSoftMax, Id, Zero
from micronas.Profiler.LatMemProfiler import set_ignore_latency
from micronas.Nas.Utils import calcNumLayers, random_one_hot_like
from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType, NAS_Module
from torch.autograd import Variable

from micronas.Nas.Utils import weight_softmax
from micronas.config import Config

from math import ceil


import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

import numpy as np



class ChooseNParallel(NAS_Module):
    def __init__(self, layers, parallel, granularity, num_channels):
        super().__init__()
        self._granularity = granularity
        self._layers = nn.ModuleList(layers)
        self._num_channels = num_channels
        self._has_zero = any([l._get_name() == "Zero" for l in layers])
        if parallel > 1:
            assert(self._has_zero), "Need Zero when parallel > 1"
        if self._has_zero:
            assert(layers[0]._get_name() == "Zero"), "Zero needs to be the first operation"

        self._bn = nn.ModuleList([nn.BatchNorm2d(num_channels) for _ in range(parallel)])
        self._add_parallel = Dyn_Add(num_channels, num_channels, granularity=granularity)
        
        # Weights for the architecture
        self._weights_op = Variable(1e-3 * torch.randn((parallel, len(self._layers)), dtype=Config.tensor_dtype, device=Config.compute_unit), requires_grad=True)

        self._weights_op_last = None

    def print_nas_weights(self, eps, gumbel, raw):
        print("Chose_OP: ", np.around(weight_softmax(self._weights_op, eps, gumbel=gumbel).cpu().detach().numpy(), 2))
        # print("CHose_OP_last: ", self._weights_op_last)

    def get_nas_weights(self):
        w = [self._weights_op]
        return w
        

    def forward(self, x, eps, weights, last_ch_weights, inf_type=InferenceType.NORMAL):
        weights_op_softmax = weight_softmax(self._weights_op, eps)

        # assert(weights.sum() == 1), f"Weights sum to {weights.sum()}"
        assert(abs(weights.sum() - 1) < 1e-5), f"Weights sum to {weights.sum()}"
        if last_ch_weights is not None:
            # assert(last_ch_weights.sum() == 1)
            assert(abs(last_ch_weights.sum() - 1) < 1e-5), f"last_ch_weights sum to {weights.sum()}"

        last_ch_weights_softmax = last_ch_weights
        weights_channels_softmax = weights

        if inf_type != InferenceType.NORMAL:
            with torch.no_grad():
                weights_op_softmax = torch.zeros_like(weights_op_softmax, dtype=Config.tensor_dtype)


        if inf_type == InferenceType.MIN:
            with torch.no_grad():
                weights_op_softmax[:,0] = 1
                weights_op_softmax[0,0] = 0
                weights_op_softmax[0,1 if self._has_zero else 0] = 1

        if inf_type == InferenceType.MAX:
            with torch.no_grad():
                weights_op_softmax[:,-1] = 1

        if inf_type == InferenceType.CUSTOM:
            with torch.no_grad():
                weights_op_softmax[:,-1] = 1
                weights_op_softmax[2,0] = 1
        if inf_type == InferenceType.SAMPLE:
            self._weights_op_last = weight_softmax(self._weights_op, eps, hard=True)
            weights_op_softmax = self._weights_op_last
        if inf_type == InferenceType.RANDOM:
            self._weights_op = random_one_hot_like(self._weights_op)
            weights_op_softmax = self._weights_op
        if inf_type == InferenceType.MAX_WEIGHT:
            weights_op_softmax = weight_softmax(self._weights_op, eps, gumbel=False)

        lat_acc = torch.tensor(0, dtype=Config.tensor_dtype).to(Config.compute_unit)

        B, C, H, W = x.shape
        input_mem = C * W * H
        if last_ch_weights is not None:
            input_list = torch.tensor([np.prod([1, (i + 1) * self._granularity, H, W]) for i in range(self._num_channels // self._granularity)], dtype=Config.tensor_dtype)
            # input_mem = torch.sum(torch.tensor([in_mem * w for in_mem, w in zip(input_list, last_ch_weights_softmax)]))
            input_mem = input_list @ last_ch_weights_softmax.T
        layer_stack = []
        for l in self._layers:
            l_out, l_lat, l_mem = l(x, weights=weights_channels_softmax, onlyOutputs=True, inf_type=inf_type, last_ch_weights=last_ch_weights_softmax)
            layer_stack.append((l_out, l_lat, l_mem))

        mem_stack = []
        parallel_out = None

        for i, w_op in enumerate(weights_op_softmax):
            mem_acc = torch.tensor(0, dtype=Config.tensor_dtype).to(Config.compute_unit)
            conv_out_stack = []
            for w, (out, lat, mem) in zip(w_op, layer_stack):
                conv_out_stack.append(out * w)
                lat_acc += lat * w
                mem_acc += mem * w
            conv_stack_out = torch.stack(conv_out_stack).sum(dim=0)
            conv_stack_out  = self._bn[i](conv_stack_out)
                
            if parallel_out is None:
                parallel_out = conv_stack_out
                mem_stack.append(mem_acc + input_mem)
            else: 
                # Add
                multiplier = w_op[1:].sum(dim=0) if self._has_zero else w_op.sum(dim=0)
                add_out, add_lat, add_mem = self._add_parallel(parallel_out, conv_stack_out, weights_channels_softmax)
                lat_acc += add_lat * multiplier
                discount_input_mem = torch.prod(1- weights_op_softmax[i+1, 0]) if i+1 < len(weights_op_softmax) else 0
                mem_stack.append(torch.max(mem_acc, add_mem * multiplier + input_mem * discount_input_mem))
                parallel_out = add_out
        mem_acc = torch.max(torch.stack(mem_stack))
        return parallel_out, lat_acc, mem_acc


    def _getKeras_pruned(self, x, weights, inf_type):
        weights_op_softmax = weight_softmax(self._weights_op, 1e-9, gumbel=False)
        # if inf_type != InferenceType.NORMAL:
        #     weights_op_softmax = torch.zeros_like(self._weights_op, dtype=Config.tensor_dtype)
        if inf_type == InferenceType.MIN:
            weights_op_softmax[:, 0] = 1
            # weights_op_softmax[0,1] = 1
            # weights_op_softmax[0,0] = 0
        if inf_type == InferenceType.MAX:
            weights_op_softmax[:, -1] = 1
        if inf_type == InferenceType.SAMPLE or inf_type == InferenceType.RANDOM:
            weights_op_softmax = self._weights_op_last
        if inf_type == InferenceType.MAX_WEIGHT:
            weights_op_softmax = weight_softmax(self._weights_op, 1e-9, gumbel=False)
        last_layer = None
        for weight in weights_op_softmax:
            maxIdx = torch.argmax(weight)
            l = self._layers[maxIdx].getKeras(x, getPruned=True, weights=weights)
            if last_layer is not None:
                last_layer = L.add([last_layer, l]) if l is not None else last_layer
            else:
                last_layer = l
        return last_layer

    def _getKeras_not_pruned(self, x, weights, inf_type):
        layer_stack = []
        for l in self._layers:
            layer_stack.append(l.getKeras(x, getPruned=False))
        res = L.add(layer_stack)
        return res


    def getKeras(self, x, getPruned, weights, inf_type):
        if getPruned:
            return self._getKeras_pruned(x, weights, inf_type)
        else:
            return self._getKeras_not_pruned(x, weights, inf_type)


class ChooseNParallel_v2(NAS_Module):
    def __init__(self, layers, granularity, num_channels):
        super().__init__()
        self._granularity = granularity
        self._layers = nn.ModuleList(layers)
        self._num_channels = num_channels
        assert all([l._get_name() != "Zero" for l in layers]), "Zero operation is not allowed"

        # self._bn = nn.ModuleList([nn.BatchNorm2d(num_channels) for _ in range(len(self._layers))])
        self._bn = nn.BatchNorm2d(num_channels)
        self._add_parallel = Dyn_Add(num_channels, num_channels, granularity=granularity)
        
        # Weights for the architecture
        self._weights_op = Variable(1e-3 * torch.randn((len(self._layers), 2), dtype=Config.tensor_dtype, device=Config.compute_unit), requires_grad=True)
        self._weights_op_last = None

    def print_nas_weights(self, eps, gumbel, raw):
        print("Chose_OP: ", np.around(weight_softmax(self._weights_op, eps, gumbel=gumbel).cpu().detach().numpy(), 2))
        # print("Choose_OP_last: ", self._weights_op_last)

    def get_nas_weights(self):
        w = [self._weights_op]
        return w
        


    def forward(self, x, eps, weights, last_ch_weights, inf_type=InferenceType.NORMAL):
        weights_op_softmax = weight_softmax(self._weights_op, eps)

        # assert(weights.sum() == 1), f"Weights sum to {weights.sum()}"
        assert(abs(weights.sum() - 1) <= 1e-5), f"Weights sum to {weights.sum()}"
        if last_ch_weights is not None:
            # assert(last_ch_weights.sum() == 1), f"last_ch_weights sum to {weights.sum()}"
            assert(abs(last_ch_weights.sum() - 1) <= 1e-5), f"last_ch_weights sum to {weights.sum()}"

        last_ch_weights_softmax = last_ch_weights
        weights_channels_softmax = weights

        if inf_type != InferenceType.NORMAL:
            with torch.no_grad():
                weights_op_softmax = torch.zeros_like(weights_op_softmax, dtype=Config.tensor_dtype)

        if inf_type == InferenceType.MIN:
            with torch.no_grad():
                weights_op_softmax[:,0] = 1
                weights_op_softmax[0,0] = 0
                weights_op_softmax[0,1] = 1

        if inf_type == InferenceType.MAX:
            with torch.no_grad():
                weights_op_softmax[:,-1] = 1

        if inf_type == InferenceType.CUSTOM:
            with torch.no_grad():
                weights_op_softmax[:,-1] = 1
                weights_op_softmax[2,0] = 1
        if inf_type == InferenceType.SAMPLE:
            self._weights_op_last = weight_softmax(self._weights_op, eps, hard=True)
        if inf_type == InferenceType.RANDOM:
            self._weights_op = random_one_hot_like(self._weights_op)
            weights_op_softmax = self._weights_op
        if inf_type == InferenceType.MAX_WEIGHT:
            weights_op_softmax = weight_softmax(self._weights_op, eps, gumbel=False)


        lat_acc = torch.tensor(0, dtype=Config.tensor_dtype).to(Config.compute_unit)

        B, C, H, W = x.shape
        input_mem = C * W * H
        if last_ch_weights is not None:
            input_list = [np.prod([1, (i + 1) * self._granularity, H, W]) for i in range(self._num_channels // self._granularity)]
            input_mem = torch.sum(torch.tensor([in_mem * w for in_mem, w in zip(input_list, last_ch_weights_softmax)]))
        layer_stack = []
        for l in self._layers:
            l_out, l_lat, l_mem = l(x, weights=weights_channels_softmax, onlyOutputs=True, inf_type=inf_type, last_ch_weights=last_ch_weights_softmax)
            layer_stack.append((l_out, l_lat, l_mem))

        mem_stack = []
        parallel_out = None

        _, _, H_out, W_out = l_out.shape
        output_mem = H_out * W_out * torch.argmax(last_ch_weights_softmax) * self._granularity

        for i, (w, (out, lat, mem)) in enumerate(zip(weights_op_softmax, layer_stack)): # w[0] => ZERO_OP, w[1] => LAYER_OP
            mem_acc = torch.tensor(0, dtype=Config.tensor_dtype).to(Config.compute_unit)
            op_out = out * w[1]
            lat_acc += lat * w[1]
            mem_acc += mem * w[1]
            
            if parallel_out is None:
                parallel_out = op_out
                mem_stack.append(mem_acc + input_mem)
                last_w = w[1]
            else:
                add_out, add_lat, add_mem = self._add_parallel(parallel_out, op_out, weights_channels_softmax)
                lat_acc += add_lat * last_w * w[1] # Don't need to add when Zero is selected
                last_w  = torch.max(w[1], last_w)
                # print("parallel_add_cost: ", add_lat * last_w * w[1])
                discount_input_mem = torch.prod(weights_op_softmax[i+1:, 1]) if i+1 < weights_op_softmax.shape[0] else 0
                mem_stack.append(torch.max(mem_acc + input_mem + output_mem * last_w, add_mem * w[1] + input_mem * discount_input_mem))
                parallel_out = add_out
        # parallel_out = self._bn(parallel_out)
        mem_acc = torch.max(torch.stack(mem_stack))
        prob_no_op = torch.prod(weights_op_softmax[:, 0])
        # print("prob_no_op: ", prob_no_op)
        return parallel_out, lat_acc, mem_acc, prob_no_op


    def _getKeras_pruned(self, x, weights, inf_type):
        weights_op_softmax = weight_softmax(self._weights_op, 1e-9, gumbel=False)
        weights_softmax = weights
        if inf_type != InferenceType.NORMAL:
            weights_op_softmax = torch.zeros_like(self._weights_op, dtype=Config.tensor_dtype)
        if inf_type == InferenceType.MIN:
            weights_op_softmax[:, 0] = 1
            weights_op_softmax[0,1] = 1
            weights_op_softmax[0,0] = 0
        if inf_type == InferenceType.MAX:
            weights_op_softmax[:, -1] = 1
        if inf_type == InferenceType.SAMPLE or inf_type == InferenceType.RANDOM:
            weights_op_softmax = self._weights_op_last
        if inf_type == InferenceType.MAX_WEIGHT:
            weights_op_softmax = weight_softmax(self._weights_op, 1e-9, gumbel=False)

        last_layer = None
        for weight, layer in zip(weights_op_softmax, self._layers):
            maxIdx = torch.argmax(weight)
            if maxIdx == 0:
                continue
            l = layer.getKeras(x, getPruned=True, weights=weights_softmax)
            if last_layer is not None:
                last_layer = L.add([last_layer, l]) if l is not None else last_layer
            else:
                last_layer = l
        return last_layer

    def _getKeras_not_pruned(self, x, weights, inf_type):
        weights_softmax = weight_softmax(weights, 1e-9)
        last_layer = None
        for l in self._layers:
            res = l.getKeras(x, getPruned=False)
            if last_layer is None:
                last_layer = res
            else:
                last_layer = self._add_parallel.getKeras(last_layer, res, getPruned=False, weights=weights_softmax)
        return last_layer


    def getKeras(self, x, getPruned, weights, inf_type):
        if getPruned:
            return self._getKeras_pruned(x, weights, inf_type)
        else:
            return self._getKeras_not_pruned(x, weights, inf_type)
