from ast import Global
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType
from micronas.Profiler.LatMemProfiler import lookup_torch

from micronas.Nas.Layers.Keras.ProfiledOp import profiledOp

from tensorflow.keras import layers as L

from math import ceil, floor
import numpy as np

import tensorflow as tf

from micronas.Utilities import torch_to_keras_shape
from micronas.config import Config

# def getPadding(i, k, s):
#     o = ceil(i / s)
#     p = (o - 1) * s - i + k
#     return ceil(p / 2) if i % 2 == 1 else floor(p / 2)

def getPadding(in_dims, filter, strides):
    in_height, in_width = in_dims[1], in_dims[0]
    filter_height, filter_width = filter[0], filter[1]
    strides=(None,1,1)
    # out_height = np.ceil(float(in_height) / float(strides[1]))
    # out_width  = np.ceil(float(in_width) / float(strides[2]))


    if (in_height % strides[1] == 0):
        pad_along_height = max(filter_height - strides[1], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[1]), 0)
    if (in_width % strides[2] == 0):
        pad_along_width = max(filter_width - strides[2], 0)
    else:
        pad_along_width = max(filter_width - (in_width % strides[2]), 0)


    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


class Conv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding="same", activation=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, 0)
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._activation = activation

        self._relu = nn.ReLU() if activation == "relu" else None
        self._sigmoid = nn.Sigmoid() if activation == "sigmoid" else None

    def forward(self, x, eps=None, onlyOutputs=False, inf_type=InferenceType.NORMAL, last_ch_weights=None):

        # Apply padding
        if self._padding == "same":
            b, c, w, h = x.shape
            # pad_x = getPadding(w, self._kernel_size[0], self._stride[0])
            # pad_y = getPadding(h, self._kernel_size[1], self._stride[0])
            pad_left, pad_right, pad_top, pad_bottom = getPadding((w, h), self._kernel_size, self._stride)
            pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        else:
            pad = x
        # Apply convolution
        res = super().forward(pad)
        lat, mem = lookup_torch(self, x.shape, res.shape, options={"padding": self._padding, "strides": list(self._stride), "activation": self._activation}, only_outputs=onlyOutputs)
        if self._activation == "relu":
            res = self._relu(res)
        if self._activation == "sigmoid":
            res = self._sigmoid(res)
        return res, lat, mem

    def getKeras(self, x):
        return profiledOp(L.Conv2D(self._out_channels, self._kernel_size, self._stride, padding="valid" if self._padding == 0 else self._padding, activation=self._activation))(x)




class Dyn_Conv2D_old(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding="same", activation=None, granularity=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, 0)

        assert(granularity is not None), "Set the granularity"
        assert(out_channels % granularity == 0), f"Problem: out_channels: {out_channels}, granularity: {granularity}"
        assert(activation in [None, "relu", "sigmoid"]), "Activation is relu, sigmoid or None"

        self._out_channels = out_channels
        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._activation = activation
        self._granularity = granularity
        self._old_granularity = -1

        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()
        
        self._bn = nn.BatchNorm2d(out_channels)

        self._masks = []
        print("old dynconv")


    def forward(self, x, eps=None, weights="full", onlyOutputs=False, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        b, c, w, h = x.shape
        if weights == "full":
            weights = torch.zeros(self._out_channels // self._granularity)
            weights[-1] = 1
        # else:
        #     assert(len(weights) == self._out_channels // self._granularity)

        if last_ch_weights is None:
            last_ch_weights = torch.zeros(self._out_channels // self._granularity, dtype=Config.tensor_dtype).to(Config.compute_unit)
            last_ch_weights[-1] = 1

        assert(abs(last_ch_weights.sum() -1) < 1e-5)
        assert(abs(weights.sum() -1) < 1e-5)


        old_granularity =  c // len(last_ch_weights) 
        self._old_granularity = old_granularity
        # Apply padding
        if self._padding == "same":
            # pad_x = getPadding(w, self._kernel_size[0], self._stride[0])
            # pad_y = getPadding(h, self._kernel_size[1], self._stride[0])
            # pad = F.pad(x, (pad_y, pad_y, pad_x, pad_x))
            pad_left, pad_right, pad_top, pad_bottom = getPadding((w, h), self._kernel_size, self._stride)
            pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        else:
            pad = x
        # Apply convolution
        res = super().forward(pad)
        # Generate the masks
        if len(self._masks) == 0:
            B, C, H, W = res.shape
            num_masks = len(range(self._granularity, self._out_channels + self._granularity, self._granularity))
            for i in range(self._granularity, self._out_channels + self._granularity, self._granularity):
                mask_ones = torch.ones((B, i, H, W)).to(Config.compute_unit)
                mask_zeros = torch.zeros((B, C - i, H, W)).to(Config.compute_unit)
                self._masks.append(torch.cat((mask_ones, mask_zeros), dim=1) * ( (num_masks -  i) / num_masks))

        # Apply masks with weights
        stack_res = []
        lat_acc = torch.tensor(0, dtype=Config.tensor_dtype).to(Config.compute_unit)
        mem_acc = torch.tensor(0, dtype=Config.tensor_dtype).to(Config.compute_unit)

        for i, (msk, w) in enumerate(zip(self._masks, weights)):
            stack_res.append(msk * w)
            for j in range(len(last_ch_weights)):
                output_shape_lookup = torch_to_keras_shape([1, (i+1) * self._granularity, *res.shape[-2:]])
                input_shape_lookup = torch_to_keras_shape([1, max(1, min((j+1) * self._old_granularity, self._in_channels)), *x.shape[-2:]])
                lat, mem = lookup_torch(self, x.shape, res.shape, options={"input_shape": input_shape_lookup,"output_shape": output_shape_lookup,"padding": self._padding, "strides": list(self._stride), "activation": self._activation}, only_outputs=onlyOutputs)
                lat_acc += lat * w * last_ch_weights[j]
                mem_acc += mem * w * last_ch_weights[j]
        res = torch.stack(stack_res).sum(dim=0) * res
        
        if self._activation == "relu":
            res = self._relu(res)
        if self._activation == "sigmoid":
            res = self._sigmoid(res)
        res = self._bn(res)
        return res, lat_acc, mem_acc

    def getKeras(self, x, getPruned=False, weights=None, inf_type=None):
        if getPruned:
            if weights is None:
                outChannels = self._out_channels
            else:
                maxW = int(torch.argmax(weights))
                outChannels = (maxW + 1) * self._granularity
            return L.Conv2D(outChannels, self._kernel_size, self._stride, padding=self._padding, activation=self._activation)(x)

        # Just for profiling
        for ch_1 in range(self._granularity, self._out_channels + self._granularity, self._granularity):
            if self._old_granularity != 0:
                for ch_2 in range(self._old_granularity, self._in_channels + self._old_granularity, self._old_granularity):
                    fake_input = tf.zeros((1, *x.shape[1:3], ch_2))
                    profiledOp(L.Conv2D(ch_1, self._kernel_size, self._stride, padding=self._padding, activation=self._activation))(fake_input)
            else:
                fake_input = tf.zeros((1, *x.shape[1:3], self._in_channels))
                profiledOp(L.Conv2D(ch_1, self._kernel_size, self._stride, padding=self._padding, activation=self._activation))(fake_input)
           

        return L.Conv2D(self._out_channels, self._kernel_size, self._stride, padding=self._padding, activation=self._activation)(x)



class Dyn_Conv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding="same", activation=None, granularity=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, 0)

        assert(granularity is not None), "Set the granularity"
        assert(out_channels % granularity == 0), f"Problem: out_channels: {out_channels}, granularity: {granularity}"
        assert(activation in [None, "relu", "sigmoid"]), "Activation is relu, sigmoid or None"

        self._out_channels = out_channels
        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._activation = activation
        self._granularity = granularity
        self._old_granularity = -1

        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()
        
        self._bn = nn.BatchNorm2d(out_channels)

        self._masks = []

        self._compiled = False

        self._latencies = None
        self._memories = None
        self._pads = None



    def forward(self, x, eps=None, weights="full", onlyOutputs=False, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        b, c, w, h = x.shape
        if weights == "full":
            weights = torch.zeros(self._out_channels // self._granularity, dtype=Config.tensor_dtype)
            weights[-1] = 1
        # else:
        #     assert(len(weights) == self._out_channels // self._granularity)

        if last_ch_weights is None:
            last_ch_weights = torch.zeros(self._out_channels // self._granularity, dtype=Config.tensor_dtype).to(Config.compute_unit)
            last_ch_weights[-1] = 1

        assert(abs(last_ch_weights.sum() -1) < 1e-5)
        assert(abs(weights.sum() -1) < 1e-5), f"The weights are {weights}"


        old_granularity =  c // len(last_ch_weights) 
        self._old_granularity = old_granularity
        # Apply padding
        if self._padding == "same":
            if not self._compiled:
                self._pads = getPadding((w, h), self._kernel_size, self._stride)
            pad = F.pad(x, self._pads)

        else:
            pad = x
        # Apply convolution
        res = super().forward(pad)
        # Generate the masks
        if not self._compiled:
            B, C, H, W = res.shape
            for i in range(self._granularity, self._out_channels + self._granularity, self._granularity):
                mask_ones = torch.ones((B, i, H, W)).to(Config.compute_unit)
                mask_zeros = torch.zeros((B, C - i, H, W)).to(Config.compute_unit)
                self._masks.append(torch.cat((mask_ones, mask_zeros), dim=1))

        stack_res = []
        for msk, w in zip(self._masks, weights):
            stack_res.append(msk * w)

        if not self._compiled:
            lat_matrix, mem_matrix = [], []
            for i, (msk, w) in enumerate(zip(self._masks, weights)):
                tmp_lat, tmp_mem = [], []
                for j in range(len(last_ch_weights)):
                    output_shape_lookup = torch_to_keras_shape([1, (i+1) * self._granularity, *res.shape[-2:]])
                    input_shape_lookup = torch_to_keras_shape([1, max(1, min((j+1) * self._old_granularity, self._in_channels)), *x.shape[-2:]])
                    lat, mem = lookup_torch(self, x.shape, res.shape, options={"input_shape": input_shape_lookup,"output_shape": output_shape_lookup,"padding": self._padding, "strides": list(self._stride), "activation": self._activation}, only_outputs=onlyOutputs)
                    tmp_lat.append(lat)
                    tmp_mem.append(mem)
                lat_matrix.append(tmp_lat)
                mem_matrix.append(tmp_mem)
            self._latencies = torch.tensor(lat_matrix, dtype=Config.tensor_dtype)
            self._memories = torch.tensor(mem_matrix, dtype=Config.tensor_dtype)


        # print("types: ", weights.type(), self._latencies.type(), last_ch_weights.type())
        lat_acc = weights @ self._latencies @ last_ch_weights
        mem_acc = weights @ self._memories @ last_ch_weights


        stack_sum = torch.stack(stack_res).sum(dim=0)
        res = stack_sum[:res.shape[0]] * res

        if self._activation == "relu":
            res = self._relu(res)
        if self._activation == "sigmoid":
            res = self._sigmoid(res)
        res = self._bn(res)
        self._compiled = True
        return res, lat_acc, mem_acc

    def getKeras(self, x, getPruned=False, weights=None, inf_type=None):
        # scale_factor = (3 if self._out_channels != 10 else 1)
        if getPruned:
            if weights is None:
                outChannels = self._out_channels
            else:
                maxW = int(torch.argmax(weights))
                outChannels = (maxW + 1) * self._granularity
            return L.Conv2D(outChannels, self._kernel_size, self._stride, padding=self._padding, activation=self._activation)(x)

        # Just for profiling
        for ch_1 in range(self._granularity, self._out_channels + self._granularity, self._granularity):
            if self._old_granularity != 0:
                for ch_2 in range(self._old_granularity, self._in_channels + self._old_granularity, self._old_granularity):
                    fake_input = tf.zeros((1, *x.shape[1:3], ch_2))
                    profiledOp(L.Conv2D(ch_1, self._kernel_size, self._stride, padding=self._padding, activation=self._activation))(fake_input)
            else:
                fake_input = tf.zeros((1, *x.shape[1:3], self._in_channels))
                profiledOp(L.Conv2D(ch_1, self._kernel_size, self._stride, padding=self._padding, activation=self._activation))(fake_input)
           
        return L.Conv2D(self._out_channels, self._kernel_size, self._stride, padding=self._padding, activation=self._activation)(x)




class Dyn_SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding="same", activation=None, granularity=None) -> None:
        super().__init__()

        assert(granularity is not None), "Set the granularity"
        assert(out_channels % granularity == 0), f"Problem: out_channels: {out_channels}, granularity: {granularity}"
        assert(activation in [None, "relu", "sigmoid"]), "Activation is relu, sigmoid or None"

        self._out_channels = out_channels
        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._activation = activation
        self._granularity = granularity
        self._old_granularity = -1

        self._depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels)
        self._point_conv = nn.Conv2d(in_channels, out_channels, 1)

        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()
        
        self._bn = nn.BatchNorm2d(out_channels)

        self._masks = []


    def forward(self, x, eps=None, weights="full", onlyOutputs=False, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        b, c, w, h = x.shape
        if weights == "full":
            weights = torch.zeros(self._out_channels // self._granularity)
            weights[-1] = 1
        # else:
        #     assert(len(weights) == self._out_channels // self._granularity)

        if last_ch_weights is None:
            last_ch_weights = torch.zeros(self._out_channels // self._granularity, dtype=Config.tensor_dtype).to(Config.compute_unit)
            last_ch_weights[-1] = 1

        assert(abs(last_ch_weights.sum() -1) < 1e-5)
        assert(abs(weights.sum() -1) < 1e-5)


        old_granularity =  c // len(last_ch_weights) 
        self._old_granularity = old_granularity
        # Apply padding
        if self._padding == "same":
            # pad_x = getPadding(w, self._kernel_size[0], self._stride[0])
            # pad_y = getPadding(h, self._kernel_size[1], self._stride[0])
            # pad = F.pad(x, (pad_y, pad_y, pad_x, pad_x))
            pad_left, pad_right, pad_top, pad_bottom = getPadding((w, h), self._kernel_size, self._stride)
            pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        else:
            pad = x
        # Apply convolution
        res = self._point_conv(self._depth_conv(pad))
        # Generate the masks
        if len(self._masks) == 0:
            B, C, H, W = res.shape
            for i in range(self._granularity, self._out_channels + self._granularity, self._granularity):
                mask_ones = torch.ones((B, i, H, W)).to(Config.compute_unit)
                mask_zeros = torch.zeros((B, C - i, H, W)).to(Config.compute_unit)
                self._masks.append(torch.cat((mask_ones, mask_zeros), dim=1))

        # Apply masks with weights
        stack_res = []
        lat_acc = torch.tensor(0, dtype=Config.tensor_dtype).to(Config.compute_unit)
        mem_acc = torch.tensor(0, dtype=Config.tensor_dtype).to(Config.compute_unit)

        for i, (msk, w) in enumerate(zip(self._masks, weights)):
            stack_res.append(msk * w)
            for j in range(len(last_ch_weights)):
                output_shape_lookup = torch_to_keras_shape([1, (i+1) * self._granularity, *res.shape[-2:]])
                input_shape_lookup = torch_to_keras_shape([1, max(1, min((j+1) * self._old_granularity, self._in_channels)), *x.shape[-2:]])
                lat, mem = lookup_torch("SeparableConv2D", x.shape, res.shape, options={"kernel_size": self._kernel_size, "input_shape": input_shape_lookup,"output_shape": output_shape_lookup,"padding": self._padding, "strides": list(self._stride), "activation": self._activation}, only_outputs=onlyOutputs)
                lat_acc += lat * w * last_ch_weights[j]
                mem_acc += mem * w * last_ch_weights[j]
        res = torch.stack(stack_res).sum(dim=0) * res
        
        if self._activation == "relu":
            res = self._relu(res)
        if self._activation == "sigmoid":
            res = self._sigmoid(res)
        res = self._bn(res)
        return res, lat_acc, mem_acc

    def getKeras(self, x, getPruned=False, weights=None, inf_type=None):
        if getPruned:
            if weights is None:
                outChannels = self._out_channels
            else:
                maxW = int(torch.argmax(weights))
                outChannels = (maxW + 1) * self._granularity
            return L.SeparableConv2D(outChannels, self._kernel_size, self._stride, padding=self._padding, activation=self._activation)(x)

        # Just for profiling
        for ch_1 in range(self._granularity, self._out_channels + self._granularity, self._granularity):
            if self._old_granularity != 0:
                for ch_2 in range(self._old_granularity, self._in_channels + self._old_granularity, self._old_granularity):
                    fake_input = tf.zeros((1, *x.shape[1:3], ch_2))
                    profiledOp(L.SeparableConv2D(ch_1, self._kernel_size, self._stride, padding=self._padding, activation=self._activation))(fake_input)
            else:
                fake_input = tf.zeros((1, *x.shape[1:3], self._in_channels))
                profiledOp(L.SeparableConv2D(ch_1, self._kernel_size, self._stride, padding=self._padding, activation=self._activation))(fake_input)
           

        return L.SeparableConv2D(self._out_channels, self._kernel_size, self._stride, padding=self._padding, activation=self._activation)(x)




class MaxPool2D(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0) -> None:
        super().__init__(kernel_size, stride, padding)
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def forward(self, x, eps=None, last_ch_weights=None):
        res = super().forward(x)
        lat, mem = lookup_torch(self, x.shape, res.shape)
        return res, lat, mem

    def getKeras(self, x):
        return profiledOp(L.MaxPool2D(self._kernel_size, self._stride, padding="valid" if self._padding == 0 else self._padding))(x)


class BatchNorm2D(nn.BatchNorm2d):
    def __init__(self, num_features) -> None:
        super().__init__(num_features)

    def forward(self, x, eps=None, last_ch_weights=None):
        res = super().forward(x)
        lat, mem = lookup_torch(self, x.shape, res.shape)
        return res, lat, mem


class GlobalAveragePooling(nn.Module):
    def __init__(self, dim=2) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, x, eps=None, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        batch_size, num_channels, H, W = x.size()
        res = x.view(batch_size, num_channels, -1).mean(dim=self._dim)
        lat, mem = lookup_torch("global_average_pooling2d", x.shape, res.shape)
        return res, lat, mem

    def getKeras(self, x, getPruned=None, inf_type=None):
        return profiledOp(L.GlobalAveragePooling2D())(x)


class LogSoftMax(nn.LogSoftmax):
    def __init__(self) -> None:
        super().__init__(dim=0)

    def forward(self, x, eps=None, last_ch_weights=None, inf_type=InferenceType.NORMAL):
        res = super().forward(x)
        lat, mem = lookup_torch(self, x.shape, res.shape)
        return res, lat, mem

    def getKeras(self, x, getPruned=None, inf_type=None):
        return profiledOp(L.Softmax())(x)


class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1, x2, eps=None, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        res = x1 + x2
        lat, mem = lookup_torch("add", x1.shape, x2.shape)
        return res, lat, mem

    def getKeras(self, x1, x2):
        return profiledOp(L.Add())([x1, x2])

class Dyn_Add(nn.Module):
    def __init__(self, in_channels, out_channels, granularity) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._granularity = granularity

        self._compiled = False
    
    def forward(self, x1, x2, weights="full", eps=None, onlyOutputs=False, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        if weights == "full":
            weights = torch.zeros(self._out_channels // self._granularity)
            weights[-1] = 1
        else:
            # assert(len(weights) == self._out_channels // self._granularity)
            self._granularity = self._out_channels // len(weights)
        res = x1 + x2
        # lat_acc, mem_acc = torch.tensor(0.0, dtype=Config.tensor_dtype).to(Config.compute_unit), torch.tensor(0.0, dtype=Config.tensor_dtype).to(Config.compute_unit)

        if not self._compiled:
            lats, mems = [], []
            for j, _ in enumerate(weights):
                add_shape = [1, (j+1) * self._granularity, *res.shape[-2:]]
                lat, mem = lookup_torch("add", add_shape, add_shape)
                lats.append(lat)
                mems.append(mem)
            self._latencies = torch.tensor(lats, dtype=Config.tensor_dtype)
            self._memories = torch.tensor(mems, dtype=Config.tensor_dtype)
        
        lat_acc = torch.sum(self._latencies * weights)
        mem_acc = torch.sum(self._memories * weights)

        
        self._compiled = True
        return res, lat_acc, mem_acc
    
    def getKeras(self, x1, x2, getPruned=False, weights=None):
        for ch_2 in range(self._granularity, self._out_channels + self._granularity, self._granularity):
            fake_input = x1[: ,: , :, 0:ch_2]
            profiledOp(L.Add())([fake_input, fake_input])

        return L.add([x1, x2])

class Mulitply(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1, x2,  eps=None, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        lat, mem = lookup_torch("multiply", x1.shape,
                                x1.shape, input_shape_002=x2.shape)
        res = torch.mul(x1, x2)
        return res, lat, mem

    def getKeras(self, x1, x2):
        return profiledOp(L.Multiply())([x1, x2])


class Dense(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, activation=None) -> None:
        super().__init__(in_features, out_features, bias)
        self._activation = activation
        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()
        self._out_features = out_features

    def forward(self, x, eps=None, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        res = super().forward(x)
        if self._activation == "relu":
            res = self._relu(res)
        if self._activation == "sigmoid":
            res = self._sigmoid(res)
        lat, mem = lookup_torch("dense", x.shape, res.shape)
        return res, lat, mem

    def getKeras(self, x):
        return profiledOp(L.Dense(self._out_features, activation=self._activation))(x)


class Id(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()
        self._conv = None
        if ch_in != ch_out:
            self._conv = Conv2D(ch_in, ch_out, (1, 1))

    def forward(self, x, eps=None, onlyOutputs=False, weights=None, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        if self._conv:
            return self._conv(x)
        return x, torch.tensor(0.0).to(Config.compute_unit), torch.tensor(0.0).to(Config.compute_unit)

    def getKeras(self, x, getPruned=None, weights=None):
        return x


class Zero(nn.Module):
    def __init__(self, stride=(1,1), channels=None) -> None:
        super().__init__()
        self._stride = stride
        self._channels = channels
        self._out_shape = None

    def forward(self, x, eps=None, onlyOutputs=False, weights=None, inf_type=InferenceType.NORMAL, last_ch_weights=None):
        B, C, H, W = x.shape
        ch = self._channels if self._channels is not None else C
        res = torch.zeros((B, ch, H, W)).to(Config.compute_unit)
        res = res[:,:,::self._stride[0],::self._stride[1]]
        self._out_shape = res
        return res, torch.tensor(0).to(Config.compute_unit), torch.tensor(0).to(Config.compute_unit)

    def getKeras(self, x, getPruned=None, weights=None):
        # return tf.zeros(torch_to_keras_shape(self._out_shape.detach().numpy().shape))
        return None