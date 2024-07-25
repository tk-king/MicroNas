import torch
import torch.nn as nn
import torch.nn.functional as F

from micronas.Nas.Layers.Pytorch.Common import Add, Conv2D, Dyn_Conv2D, GlobalAveragePooling, SoftMax, Id, Zero
from micronas.Nas.Utils import calcNumLayers
from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType, NAS_Module

from math import ceil


import tensorflow.keras as keras
import tensorflow.keras.layers as L

import numpy as np

def weight_softmax(weights, eps):
    return F.gumbel_softmax(weights, eps)

class Dyn_Input(NAS_Module):
    def __init__(self, num_sensors):
        super().__init__()
        self._weights = torch.autograd.Variable(torch.zeros(num_sensors), requires_grad=True)
    
    def get_nas_weights(self):
        return [self._weights]

    def print_nas_weights(self, eps):
        print("Input_weights_channels: ", weight_softmax(self._weights, eps))

    def forward(self, x, eps):
        weights = weight_softmax(self._weights, eps)
        res = x * weights[None, None, None, :]
        return res
    
    def print_nas_weights(self, eps):
        print(weight_softmax(self._weights, eps))


def getInputMem(shape, granularity, weight):
    mem = torch.tensor(0.0, dtype=Config.tensor_dtype)
    mem_shape = np.prod(shape)
    for i, w in enumerate(weight):
        mem += (mem_shape * w) * (i + 1) / len(weight)
    return mem



class TCModule(NAS_Module):
    def __init__(self, input_dims, channels=64, reduceTime=False, reduceCh=False, input=False, parallel=2, kernels=[3, 5, 7], bottleNeck=False, addZero=False, granularity=8):
        super().__init__()

        self._input_dims = input_dims
        self._kernels = kernels

        self._add = Add()
        self._bottleNeck = None
        if bottleNeck:
            self._bottleNeck = Dyn_Conv2D(1 if input else channels, channels, (1, input_dims[1]), activation="relu", padding="same", granularity=granularity, stride=(1, 2 if reduceCh else 1))

        self._layers = nn.ModuleList()

        for k_size in kernels:
            self._layers.append(Dyn_Conv2D(1 if input and not bottleNeck else channels, channels, (k_size, 1), activation="relu", padding="same", granularity=granularity, stride=(2 if reduceTime else 1, 1)))

        if not input and not reduceTime and not reduceCh:
            self._layers.append(Id(channels, channels))
            if addZero:
                self._layers.append(Zero())
                print("Addzero")
        self.bn = [nn.BatchNorm2d(channels) for _ in range(parallel)]
        self._bn_conv = [nn.BatchNorm2d(channels) for _ in self._layers]

        self._weights = torch.autograd.Variable(torch.zeros((parallel, len(self._layers)), dtype=Config.tensor_dtype), requires_grad=True)
        self._channel_weights_bt = torch.autograd.Variable(torch.zeros(channels // granularity, dtype=Config.tensor_dtype), requires_grad=True)
        self._channel_weights_conv = torch.autograd.Variable(torch.zeros(channels // granularity, dtype=Config.tensor_dtype), requires_grad=True)
        # with torch.no_grad():
        #     self._channel_weights[3] = 1
        #     self._weights[1] = 1

    def get_nas_weights(self):
        if self._bottleNeck is not None:
            return [self._weights, self._channel_weights_conv, self._channel_weights_bt]
        return [self._weights, self._channel_weights_conv]

    def print_nas_weights(self, eps):
        print("Choose_op: ", weight_softmax(self._weights, eps))
        print("Choose_channel_conv: ", weight_softmax(self._channel_weights_conv, eps))
        if self._bottleNeck is not None:
            print("Choose_channel_bt: ", weight_softmax(self._channel_weights_bt, eps))



    def forward(self, x, eps, onlyOutputs=False, last_ch_weights=None, inf_type=InferenceType.NORMAL):
        if inf_type == InferenceType.MIN:
            with torch.no_grad():
                self._weights[:,0 if len(self._kernels) == len(self._layers) else -1] = 1
                self._channel_weights_conv[0] = 1
                self._channel_weights_bt[0] = 1

        if inf_type == InferenceType.MAX:
            with torch.no_grad():
                self._weights[:,2] = 1
                self._channel_weights_conv[-1] = 1
                self._channel_weights_bt[-1] = 1

        lat_bot, mem_bot = torch.tensor(0.0, dtype=Config.tensor_dtype), torch.tensor(0.0, dtype=Config.tensor_dtype)
        if self._bottleNeck is not None:
            x, lat_bot, mem_bot = self._bottleNeck(x, weight_softmax(self._channel_weights_bt, eps), last_ch_weights=weight_softmax(last_ch_weights, eps) if last_ch_weights is not None else None)

        last_ch_w = last_ch_weights if self._bottleNeck is None else self._channel_weights_bt
        last_ch_w = weight_softmax(last_ch_w, eps) if last_ch_w is not None else None

        layer_stack = []
        for l, bn in zip(self._layers, self._bn_conv):
            l_out, l_lat, l_mem = l(x, weight_softmax(self._channel_weights_conv, eps), inf_type=inf_type, last_ch_weights=last_ch_w)
            layer_stack.append((bn(l_out), l_lat, l_mem))

        lat_acc = lat_bot
        out = None
        mem_acc = torch.tensor(0.0)
        input_size = np.prod(list(x.size()))
        for i, par_weights in enumerate(self._weights):
            x_stack = []
            for w, (res, lat, mem) in zip(weight_softmax(par_weights, eps), layer_stack):
                x_stack.append(res * w)
                lat_acc += lat * w
                mem_acc += F.relu(mem - (input_size if i != 0 else 0)) * w
            x = torch.stack(x_stack).sum(dim=0)
            x  = self.bn[i](x)
            if out is None:
                out = x
            else:
                out, lat_add, mem_add = self._add(out, x)
                mem_acc = torch.max(mem_add, mem_acc)
                lat_acc += lat_add


        return out, lat_acc, torch.max(torch.stack([mem_acc, mem_bot])), self._channel_weights_conv

    
    def getKeras(self, x, getPruned=False):
        if self._bottleNeck is not None:
            x = self._bottleNeck.getKeras(x, getPruned=getPruned, weights=weight_softmax(self._channel_weights_bt, 1e-9))
        if getPruned:
            last_layer = None
            for weight in self._weights:
                maxIdx = torch.argmax(weight_softmax(weight, 1e-9))
                l = self._layers[maxIdx].getKeras(x, getPruned=getPruned, weights=weight_softmax(self._channel_weights_conv, 1e-9))
                if last_layer is not None:
                    last_layer = self._add.getKeras(last_layer, l)
                else:
                    last_layer = l

        # Not pruned
        x_stack = []
        for mod in self._layers:
            x_stack.append(mod.getKeras(x, getPruned=getPruned, weights=weight_softmax(self._channel_weights_conv, 1e-9)))
        out = L.add(x_stack)
        if len(self._weights) != 1:
            self._add.getKeras(x_stack[0], x_stack[1])
        return out 


class Parallel_Con(NAS_Module):
    def __init__(self, input_dims, channels=64, reduceTime=False, reduceCh=False, input=False, parallel=1, bottleNeck=False):
        super().__init__()

        self._layers = nn.ModuleList()

        for i in range(parallel):
            self._layers.append(TCModule(input_dims, channels=channels, reduceTime=reduceTime, reduceCh=reduceCh, input=input, bottleNeck=bottleNeck, addZero=i==0 and parallel > 1))

        self._weights = torch.autograd.Variable(torch.zeros(len(self._layers)), requires_grad=True)

    def get_nas_weights(self):
        w  = []
        w.append(self._weights)
        for l in self._layers:
            w.extend(l.get_nas_weights())
        return w

    def print_nas_weights(self, eps):
        print("Parallel_con: ", weight_softmax(self._weights, eps))
        for l in self._layers:
            l.print_nas_weights(eps)
        

    def forward(self, x, eps, inf_type=InferenceType.NORMAL):
        lat_acc, mem_acc = torch.tensor(0.0), torch.tensor(0.0)
        stack = []
        for i, l in enumerate(self._layers):
            res, lat, mem = l(x, eps, onlyOutputs=i!=0, inf_type=inf_type)
            lat_acc += lat
            mem_acc += mem
            stack.append(res)
        x = torch.stack(stack).sum(dim=0)
        return x, lat_acc, mem_acc

    def getKeras(self, x):
        stack = []
        for l in self._layers:
            stack.append(l.getKeras(x))
        return L.add(stack) if len(stack) > 1 else stack[0]
         



class  Parallel_Net(NAS_Module):
    def __init__(self, input_dims, num_classes, dim_limits=[8, 3], channels=64, net_scale_factor=0):
        super().__init__()

        self._multiple_inputs = False
        self._input_channel_select = []
        if isinstance(input_dims[0], list):
            first_input = input_dims[0] 
            self._multiple_inputs = True

        else:
            first_input = input_dims
        
        for in_shape in input_dims:
            self._input_channel_select.append(Dyn_Input(in_shape[1]))

        self._input_dims = input_dims

        self._num_red_time = calcNumLayers(first_input[0], 2, dim_limits[0])
        self._num_red_ch = calcNumLayers(first_input[1], 2, dim_limits[1])

        self._num_min_layers = self._num_red_ch + self._num_red_time
        # self.continious_len_reduce = self._num_red_time // self._num_ch_time
        cur_shape_time = first_input[0]
        cur_shape_ch =  first_input[1]

        # Weights for search
        num_inputs = len(input_dims) if self._multiple_inputs else 1
        self._nas_input_weights = torch.autograd.Variable(torch.zeros(num_inputs, dtype=Config.tensor_dtype), requires_grad=True)
        # with torch.no_grad():
        #     self._nas_input_weights[0] = 1

        self._layers = nn.ModuleList()
        ctr = 0

        for i in range(self._num_red_time):
            # self._layers.append(Parallel_Con([cur_shape_time, cur_shape_ch], channels=channels, reduceTime=True, input=i==0))
            self._layers.append(TCModule([cur_shape_time, cur_shape_ch], channels=channels, reduceTime=True, input=i==0, parallel=1))
            cur_shape_time = ceil(cur_shape_time / 2)

        for _ in range(self._num_red_ch):
            # self._layers.append(Parallel_Con([cur_shape_time, cur_shape_ch], channels=channels, reduceCh=True, bottleNeck=False))
            self._layers.append(TCModule([cur_shape_time, cur_shape_ch], channels=channels, reduceCh=True, bottleNeck=True, parallel=3))
            cur_shape_ch = ceil(cur_shape_ch / 2)
            for _ in range(net_scale_factor):
                # self._layers.append(Parallel_Con([cur_shape_time, cur_shape_ch], channels=channels))
                self._layers.append(TCModule([cur_shape_time, cur_shape_ch], channels=channels, bottleNeck=True, parallel=3))

        # Classification at the end
        self.cls_conv = Conv2D(channels, num_classes, (1, 1), activation="relu")
        self.cls_gap = GlobalAveragePooling()
        self.cls_softmax = SoftMax()


    def get_nas_weights(self):
        w  = []
        w.append(self._nas_input_weights)
        for l in self._layers:
            w.extend(l.get_nas_weights())
        for l in self._input_channel_select:
            w.extend(l.get_nas_weights())
        return w

    def print_nas_weights(self):
        print("Input_weights: ", weight_softmax(self._nas_input_weights, self._t))
        for l in self._layers:
            l.print_nas_weights(self._t)
        for l in self._input_channel_select:
            l.print_nas_weights(self._t)
    

    def forward(self, x, inf_type=InferenceType.NORMAL):
        lat_acc = torch.tensor(0.0)
        mem_acc = torch.tensor(0.0)

        if inf_type == InferenceType.MIN:
            with torch.no_grad():
                self._nas_input_weights[-1] = 1
                self._t = 1e-9
        if inf_type == InferenceType.MAX:
            with torch.no_grad():
                self._nas_input_weights[0] = 1
                self._t = 1e-9

        input_weights = weight_softmax(self._nas_input_weights, self._t)
        inputs = x
        if self._multiple_inputs:
            # TODO: Multiple inputs
            # x = self._input_channel_select[0](x[0], self._t) * input_weights[0]
            x = x[0] 
        input_ctr = 1
        last_ch_weights = None
        for l in self._layers:
            x, lat, mem, last_ch_weights = l(x, eps=self._t, inf_type=inf_type, last_ch_weights=last_ch_weights)
            lat_acc += lat
            mem_acc = torch.max(mem_acc, mem)
            time_dim = x.shape[2]
            # TODO: Multiple inputs
            # if self._multiple_inputs and input_ctr < len(inputs) and  time_dim == self._input_dims[input_ctr][0]:
            #     x = torch.add(self._input_channel_select[input_ctr](inputs[input_ctr], self._t) * input_weights[input_ctr], x * (1- input_weights[input_ctr]))
            #     mem_acc = (1- input_weights[input_ctr]) * mem_acc
            #     lat_acc = (1- input_weights[input_ctr]) * lat_acc
            #     input_ctr += 1
        x, lat_cls_conv, mem_cls_conv = self.cls_conv(x)
        x, lat_gap, mem_gap = self.cls_gap(x)
        x, lat_softmax, mem_softmax = self.cls_softmax(x)

        lat_acc += lat_cls_conv + lat_gap + lat_softmax
        mem_acc = torch.max(torch.stack([mem_acc, mem_cls_conv, mem_gap, mem_softmax]))

        return x, lat_acc, mem_acc

    def getKeras(self, getPruned=False):
        inputs = [keras.Input([*i_shape, 1], batch_size=1) for i_shape in self._input_dims]

        x = inputs[0]

        input_ctr = 1
        for l in self._layers:
            x = l.getKeras(x, getPruned=getPruned)
            time_dim = x.shape[1]
            if self._multiple_inputs and input_ctr < len(inputs) and time_dim == self._input_dims[input_ctr][0]:
                print("ANOTHER INPUT")
                x = inputs[input_ctr] + x
                input_ctr += 1
        
        x = self.cls_conv.getKeras(x)
        x = self.cls_gap.getKeras(x)
        x = self.cls_softmax.getKeras(x)


        return keras.Model(inputs=inputs, outputs=x)


