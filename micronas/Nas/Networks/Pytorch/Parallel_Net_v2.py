import torch
import torch.nn as nn
import torch.nn.functional as F

from micronas.Nas.Layers.Pytorch.Common import Add, Conv2D, Dyn_Add, Dyn_Conv2D, GlobalAveragePooling, LogSoftMax, Id, Zero
from micronas.Nas.Utils import calcNumLayers
from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType, NAS_Module

from math import ceil


import tensorflow.keras as keras
import tensorflow.keras.layers as L

import numpy as np


def weight_softmax(weights, eps):
    return F.softmax(weights / eps)

class Dyn_Input(NAS_Module):
    def __init__(self, num_sensors):
        super().__init__()
        self._weights = torch.autograd.Variable(torch.zeros(num_sensors), requires_grad=True)
    
    def get_nas_weights(self):
        return [self._weights]

    def print_nas_weights(self, eps):
        print("Input_weights_channels: ", weight_softmax(self._weights, eps).cpu().detach().numpy())

    def forward(self, x, eps):
        weights = weight_softmax(self._weights, eps)
        res = x * weights[None, None, None, :]
        return res
    
    def print_nas_weights(self, eps):
        print(weight_softmax(self._weights, eps))

    def getKeras(self, x, getPruned=False):
        raise NotImplementedError("Dyn_Input not implemented for keras")

class TCN_CH_REDUCE(NAS_Module):
    def __init__(self, input_dims, channels=64, stride=(1, 1), parallel=2, kernels=[3, 5, 7], granularity=8):
        super().__init__()
        self._input_dims = input_dims
        self._kernel = kernels
        self._granularity = granularity
        self._stride = stride

        self._bottleNeck = Dyn_Conv2D(channels, channels, (1, input_dims[1]), activation="relu", padding="same", granularity=granularity, stride=stride)

        self._layers = nn.ModuleList()
        if parallel > 1:
            self._layers.append(Zero())
        for k_size in kernels:
            self._layers.append(Dyn_Conv2D(channels, channels, (k_size, 1), activation="relu", padding="same", granularity=granularity))

        # Skip connection
        self._skip_connect = nn.ModuleList([
            Zero(stride=stride),
            Dyn_Conv2D(channels, channels, stride, activation="relu", padding="same", granularity=granularity, stride=stride)
        ])
        self._skip_add = Dyn_Add(channels, channels, granularity=granularity)
        self._layer_skip_add = Dyn_Add(channels, channels, granularity=granularity)


        self.bn = [nn.BatchNorm2d(channels) for _ in range(parallel)]
        self._bn_conv = [nn.BatchNorm2d(channels) for _ in self._layers]

        # Identity for the whole layer
        self._skip_layer = None
        if stride == (1, 1):
            self._skip_layer = Id(channels, channels)

        # Can reuse the add => No internal state
        self._add_parallel = Dyn_Add(channels, channels, granularity=granularity)
        
        # Weights for the architecture
        self._weights_op = torch.autograd.Variable(1e-3 * torch.randn((parallel, len(self._layers)), dtype=float), requires_grad=True)
        self._weights_channel_bt = torch.autograd.Variable(torch.zeros(channels // granularity, dtype=float), requires_grad=True)
        self._weights_channel_conv = torch.autograd.Variable(torch.zeros(channels // granularity, dtype=float), requires_grad=True)
        self._weights_skip_connect = torch.autograd.Variable(torch.zeros(len(self._skip_connect), dtype=float), requires_grad=True)
        self._weights_skip_module = torch.autograd.Variable(torch.zeros(2, dtype=float), requires_grad=True)

    def get_nas_weights(self):
        w = [self._weights_op, self._weights_channel_bt, self._weights_channel_conv, self._weights_skip_connect]
        if self._skip_layer is not None:
            w.append(self._weights_skip_module)
        return w

    def print_nas_weights(self, eps):
        print("Choose_op: ", weight_softmax(self._weights_op, eps).detach().numpy())
        print("Choose_channel_bt: ", weight_softmax(self._weights_channel_bt, eps).detach().numpy())
        print("Choose_channel_conv: ", weight_softmax(self._weights_channel_conv, eps).detach().numpy())
        print("Skip_connect: ", weight_softmax(self._weights_skip_connect, eps).detach().numpy())
        print("Skip module: ", weight_softmax(self._weights_skip_module, eps).detach().numpy())
        


    def forward(self, x, eps, last_ch_weights=None, inf_type=InferenceType.NORMAL):

        if inf_type == InferenceType.MIN:
            with torch.no_grad():
                self._weights_op[:,0] = 1
                self._weights_op[0,1] = 1
                self._weights_op[0,0] = 0
                self._weights_channel_conv[0] = 1
                self._weights_channel_bt[0] = 1
                self._weights_skip_connect[0] = 1
                self._weights_skip_module[1] = 1

        if inf_type == InferenceType.MAX:
            with torch.no_grad():
                self._weights_op[:,-1] = 1
                self._weights_channel_conv[-1] = 1
                self._weights_channel_bt[-1] = 1
                self._weights_skip_connect[1] = 1
                self._weights_skip_module[1] = 1

        if inf_type == InferenceType.CUSTOM:
            with torch.no_grad():
                self._weights_op[0,0] = 1
                self._weights_op[1,2] = 1
                self._weights_op[2,3] = 1
                self._weights_channel_conv[-1] = 1
                self._weights_channel_bt[-1] = 1
                self._weights_skip_connect[1] = 1
                self._weights_skip_module[1] = 1

        last_ch_weights_softmax = weight_softmax(last_ch_weights, eps)

        # Compute bottleNeck layer
        bt_weights_softmax = weight_softmax(self._weights_channel_bt, eps)
        bt_out, bt_lat, bt_mem = self._bottleNeck(x, bt_weights_softmax, last_ch_weights=last_ch_weights_softmax)
        layer_stack = []
        ch_weights_conv = weight_softmax(self._weights_channel_conv, eps)
        for l, bn in zip(self._layers, self._bn_conv):
            l_out, l_lat, l_mem = l(bt_out, ch_weights_conv, inf_type=inf_type, last_ch_weights=bt_weights_softmax)
            layer_stack.append((bn(l_out), l_lat, l_mem))
        
        lat_acc = bt_lat
        mem_stack = []
        parallel_out = None
        for i, par_weights in enumerate(self._weights_op):
            mem_acc = torch.tensor(0.0, dtype=float)
            weights_conv_softmax = weight_softmax(par_weights, eps)
            conv_block_out_stack = []
            for w, k, (out, lat, mem) in zip(weights_conv_softmax, self._kernel, layer_stack):
                conv_block_out_stack.append(out * w)
                lat_acc += lat * w
                if i != 0 and i != len(layer_stack) - 1:
                    mem_acc += F.relu((mem - ((bt_out.shape[1] * k * 4) + bt_out.nelement() if i != 0 else 0)) * w)
            conv_stack_out = torch.stack(conv_block_out_stack).sum(dim=0)
            conv_stack_out  = self.bn[i](conv_stack_out)
            if parallel_out is None:
                parallel_out = conv_stack_out
                mem_stack.append(mem_acc)
            else:
                parallel_add_out, parallel_add_lat, parallel_add_mem = self._add_parallel(parallel_out, conv_stack_out, ch_weights_conv)
                lat_acc += parallel_add_lat * weights_conv_softmax[1:].sum(dim=0) # Discount with prob of zero_op
                mem_stack.append(mem_acc + parallel_add_mem * weights_conv_softmax[1:].sum(dim=0))
                parallel_out = parallel_add_out
        mem_acc = torch.max(torch.stack(mem_stack))
        mem_acc = torch.max(mem_acc, bt_mem) 
        block_out = parallel_out

        # Skip connection
        w_skip_connect = weight_softmax(self._weights_skip_connect, eps)
        out_id, lat_id, mem_id = self._skip_connect[1](x, ch_weights_conv, inf_type=inf_type, last_ch_weights=last_ch_weights_softmax)
        out_zero, lat_zero, mem_zero = self._skip_connect[0](x, ch_weights_conv, inf_type=inf_type, last_ch_weights=last_ch_weights_softmax)
        skip_out = out_id * w_skip_connect[1] + out_zero * w_skip_connect[0]
        skip_lat = lat_id * w_skip_connect[1] + lat_zero * w_skip_connect[0]
        skip_mem = mem_id * w_skip_connect[1] + mem_zero * w_skip_connect[0]
        
        lat_acc += skip_lat
        mem_acc += skip_mem
        out_skip_add, lat_skip_add, mem_skip_add = self._skip_add(block_out, skip_out, ch_weights_conv)
        mem_acc = torch.max(mem_acc, mem_skip_add * (w_skip_connect[1]))
        lat_acc += lat_skip_add * (1 - w_skip_connect[1])

        # Decide weather to skip the entire layer, only when stride == 1
        if self._skip_layer is not None:
            w_skip_layer = weight_softmax(self._weights_skip_module, eps)
            out_module_skip, lat_module_skip, mem_module_skip = self._skip_layer(x)
            mod_out = out_module_skip * (w_skip_layer[0]) + out_skip_add * (w_skip_layer[1])
            lat_acc = lat_module_skip * (w_skip_layer[0]) + lat_acc * (w_skip_layer[1])
            mem_acc = mem_module_skip * (w_skip_layer[0]) + mem_acc * (w_skip_layer[1])
            return mod_out, lat_acc, mem_acc, self._weights_channel_conv
        
        return out_skip_add, lat_acc, mem_acc, self._weights_channel_conv


    def getKeras(self, x, getPruned=False):
        if getPruned:#
            # Skip whole layer
            if self._stride == (1, 1):
                max_mod_skip = torch.argmax(weight_softmax(self._weights_skip_module, 1e-9))
                if max_mod_skip == 0:
                    return x
            
            # Do not skip whole layer
            bt_out = self._bottleNeck.getKeras(x, getPruned=getPruned, weights=weight_softmax(self._weights_channel_bt, 1e-9))
            last_layer = None
            for weight in self._weights_op:
                maxIdx = torch.argmax(weight_softmax(weight, 1e-9))
                l = self._layers[maxIdx].getKeras(bt_out, getPruned=getPruned, weights=weight_softmax(self._weights_channel_conv, 1e-9))
                if last_layer is not None:
                    last_layer = L.add([last_layer, l]) if l is not None else last_layer
                else:
                    last_layer = l
            maxIdx = torch.argmax(weight_softmax(self._weights_skip_connect, 1e-9))
            res = last_layer
            if maxIdx != 0:
                skip_out = self._skip_connect[maxIdx].getKeras(x, getPruned=getPruned, weights=weight_softmax(self._weights_channel_conv, 1e-9))
                res = self._skip_add.getKeras(skip_out, last_layer, getPruned=getPruned, weights=weight_softmax(self._weights_channel_conv, 1e-9))

            return res
            # if self._skip_layer is not None:
            #     maxIdx = torch.argmax(weight_softmax(self._weights_skip_module, 1e-9))
            #     if maxIdx != 0:
            #         print("skip_out: ", skip_out.shape, res.shape)
            #         skip_out = self._skip_layer.getKeras(x, getPruned=getPruned)
            #         res = skip_out + res
            # return res

        # Not pruned
        bt_out = self._bottleNeck.getKeras(x, getPruned=getPruned, weights=weight_softmax(self._weights_channel_bt, 1e-9))
        last_layer = None
        for layer in self._layers:
            l = layer.getKeras(bt_out, getPruned=getPruned, weights=weight_softmax(self._weights_channel_conv, 1e-9))
            if last_layer is not None:
                last_layer = L.add([last_layer, l]) if l is not None else last_layer
            else:
                last_layer = l

        skip_out = self._skip_connect[1].getKeras(x, getPruned=getPruned, weights=weight_softmax(self._weights_channel_conv, 1e-9))
        res = self._skip_add.getKeras(skip_out, last_layer, getPruned=getPruned, weights=weight_softmax(self._weights_channel_conv, 1e-9))

        if self._skip_layer is not None:
            res = self._skip_layer.getKeras(x, getPruned=getPruned) + res
        return res




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
        self.bn = [nn.BatchNorm2d(channels) for _ in range(parallel)]
        self._bn_conv = [nn.BatchNorm2d(channels) for _ in self._layers]

        self._weights = torch.autograd.Variable(torch.zeros((parallel, len(self._layers)), dtype=float), requires_grad=True)
        self._channel_weights_bt = torch.autograd.Variable(torch.zeros(channels // granularity, dtype=float), requires_grad=True)
        self._channel_weights_conv = torch.autograd.Variable(torch.zeros(channels // granularity, dtype=float), requires_grad=True)

    def get_nas_weights(self):
        if self._bottleNeck is not None:
            return [self._weights, self._channel_weights_conv, self._channel_weights_bt]
        return [self._weights, self._channel_weights_conv]

    def print_nas_weights(self, eps):
        print("Choose_op: ", weight_softmax(self._weights, eps).detach().numpy())
        print("Choose_channel_conv: ", weight_softmax(self._channel_weights_conv, eps).detach().numpy())
        if self._bottleNeck is not None:
            print("Choose_channel_bt: ", weight_softmax(self._channel_weights_bt, eps).detach().numpy())



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

        
        last_ch_weights=weight_softmax(last_ch_weights, eps) if last_ch_weights is not None else None
        channel_weights_bt = weight_softmax(self._channel_weights_bt, eps)

        lat_bot, mem_bot = torch.tensor(0.0, dtype=float), torch.tensor(0.0, dtype=float)
        if self._bottleNeck is not None:
            x, lat_bot, mem_bot = self._bottleNeck(x, channel_weights_bt, last_ch_weights=last_ch_weights if last_ch_weights is not None else None)

        last_ch_w = last_ch_weights if self._bottleNeck is None else channel_weights_bt
        last_ch_w = last_ch_weights if last_ch_w is not None else None

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
            return last_layer
        # Not pruned
        x_stack = []
        for mod in self._layers:
            x_stack.append(mod.getKeras(x, getPruned=getPruned, weights=weight_softmax(self._channel_weights_conv, 1e-9)))
        out = L.add(x_stack)
        if len(self._weights) != 1:
            self._add.getKeras(x_stack[0], x_stack[1])
        return out 


class  Parallel_Net_v2(NAS_Module):
    def __init__(self, input_dims, num_classes, dim_limits=[8, 3], channels=64, net_scale_factor=0):
        super().__init__()


        self._t = 1
        self._multiple_inputs = False
        if isinstance(input_dims[0], list):
            first_input = input_dims[0] 
            self._multiple_inputs = True

        else:
            first_input = input_dims

        self._input_dims = input_dims

        self._num_red_time = calcNumLayers(first_input[0], 2, dim_limits[0])
        self._num_red_ch = calcNumLayers(first_input[1], 2, dim_limits[1])

        self._num_min_layers = self._num_red_ch + self._num_red_time
        cur_shape_time = first_input[0]
        cur_shape_ch =  first_input[1]

        num_inputs = len(input_dims) if self._multiple_inputs else 1


        self._layers = nn.ModuleList()

        for i in range(self._num_red_time):
            self._layers.append(TCModule([cur_shape_time, cur_shape_ch], channels=channels, reduceTime=True, input=i==0, parallel=1))
            cur_shape_time = ceil(cur_shape_time / 2)

        for _ in range(self._num_red_ch):
            self._layers.append(TCN_CH_REDUCE([cur_shape_time, cur_shape_ch], channels=channels, stride=(1, 2), parallel=3))
            cur_shape_ch = ceil(cur_shape_ch / 2)
            for _ in range(net_scale_factor):
                self._layers.append(TCN_CH_REDUCE([cur_shape_time, cur_shape_ch], channels=channels, parallel=3))

        # Classification at the end
        self.cls_conv = Conv2D(channels, num_classes, (1, 1), activation="relu")
        self.cls_gap = GlobalAveragePooling()
        self.cls_softmax = LogSoftMax()


        self._nas_input_weights = torch.autograd.Variable(torch.zeros(num_inputs, dtype=float), requires_grad=True)


    def get_nas_weights(self):
        w  = []
        w.append(self._nas_input_weights)
        for l in self._layers:
            w.extend(l.get_nas_weights())
        for l in self._input_channel_select:
            w.extend(l.get_nas_weights())
        return w

    def print_nas_weights(self):
        print("Input_weights: ", weight_softmax(self._nas_input_weights, self._t).detach().numpy())
        for i, l in enumerate(self._layers):
            print(f"----Layer_{i}----")
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
            x = x[0] 
        input_ctr = 1
        last_ch_weights = None
        for l in self._layers:
            x, lat, mem, last_ch_weights = l(x, eps=self._t, inf_type=inf_type, last_ch_weights=last_ch_weights)
            lat_acc += lat
            mem_acc = torch.max(mem_acc, mem)
            time_dim = x.shape[2]
            if self._multiple_inputs and input_ctr < len(inputs) and  time_dim == self._input_dims[input_ctr][0]:
                x = torch.add(inputs[input_ctr] * input_weights[input_ctr], x * (1- input_weights[input_ctr]))
                mem_acc = (1- input_weights[input_ctr]) * mem_acc
                lat_acc = (1- input_weights[input_ctr]) * lat_acc
                input_ctr += 1
        x, lat_cls_conv, mem_cls_conv = self.cls_conv(x)
        x, lat_gap, mem_gap = self.cls_gap(x)
        x, lat_softmax, mem_softmax = self.cls_softmax(x)

        lat_acc += lat_cls_conv + lat_gap + lat_softmax
        mem_acc = torch.max(torch.stack([mem_acc, mem_cls_conv, mem_gap, mem_softmax]))

        return x, lat_acc, mem_acc

    def getKeras(self, getPruned=False, batch_size=1):
        inputs = [keras.Input([*i_shape, 1], batch_size=batch_size) for i_shape in self._input_dims]

        x = inputs[0]

        input_ctr = 1
        for i, l in enumerate(self._layers):
            x = l.getKeras(x, getPruned=getPruned)
            time_dim = x.shape[1]
            if self._multiple_inputs and input_ctr < len(inputs) and time_dim == self._input_dims[input_ctr][0]:
                x = inputs[input_ctr] + x
                input_ctr += 1
        
        x = self.cls_conv.getKeras(x)
        x = self.cls_gap.getKeras(x)
        x = self.cls_softmax.getKeras(x)


        return keras.Model(inputs=inputs, outputs=x)


