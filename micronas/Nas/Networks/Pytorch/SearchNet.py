import torch
import torch.nn as nn
from micronas.Nas.Layers.Pytorch import ChooseNParallel

from micronas.Nas.Layers.Pytorch.Common import Dyn_Conv2D, GlobalAveragePooling, LogSoftMax, Id
from micronas.Nas.Utils import calcNumLayers, random_one_hot_like, weight_softmax
from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType

from micronas.Nas.Layers.Pytorch import Sequential, Parallel_Choice_Add, ChooseNParallel_v2, MakeChoice
from micronas.config import Config

from math import ceil


import tensorflow.keras as keras
import tensorflow.keras.layers as L

import numpy as np


class SearchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def compile(self, ts_len, num_sensors, num_classes):
        input_dims = (ts_len, num_sensors)
        self._input_dims = input_dims
        self._num_classes = num_classes
        self._net_scale_factor = Config.net_scale_factor

        self._num_ch_redTime = Config.time_reduce_ch
        self._gran_redTime = Config.time_reduce_granularity

        self._num_ch_redCh = Config.ch_reduce_ch
        self._gran_redCh = Config.ch_reduce_granularity

        self._t = 1
        # self._layers = []
        self._num_red_time = max(1, calcNumLayers(input_dims[0], 2, Config.dim_limit_time))
        self._num_red_ch = max(1, calcNumLayers(input_dims[1], 2, Config.dim_limit_ch))


        cur_shape_time = input_dims[0]
        cur_shape_ch = input_dims[1]


        timeReduceLayers = []
        for i in range(self._num_red_time):
            timeReduceLayers.append(ReduceTime(self._num_ch_redTime, self._gran_redTime, isInput=i==0))
            cur_shape_time = ceil(cur_shape_time / 2)

        self._time_reduce = Sequential(timeReduceLayers)

        chReduceLayers = []
        for i in range(self._num_red_ch):
            chReduceLayers.append(ReduceCH(cur_shape_ch, self._num_ch_redTime if i == 0 else self._num_ch_redCh, self._num_ch_redCh, reduce=True, granularity=self._gran_redCh))
            cur_shape_ch = ceil(cur_shape_ch / 2)
            for _ in range(self._net_scale_factor):
                chReduceLayers.append(ReduceCH(cur_shape_ch, self._num_ch_redCh, self._num_ch_redCh, reduce=False, granularity=self._gran_redCh))
        self._ch_reduce = Sequential(chReduceLayers)

        self._cls_stack = Sequential([
            Dyn_Conv2D(self._num_ch_redCh, num_classes, (1, 1), activation="relu", granularity=num_classes),
            GlobalAveragePooling(),
            LogSoftMax()
        ])

        fake_input = torch.randn((1, 1, *input_dims))
        self(fake_input)


    def get_nas_weights(self):
        res = []
        res.extend(self._time_reduce.get_nas_weights())
        res.extend(self._ch_reduce.get_nas_weights())
        res.extend(self._cls_stack.get_nas_weights())
        return res
    
    def print_nas_weights(self, eps, gumbel=False, raw=False):
        self._time_reduce.print_nas_weights(eps, gumbel=gumbel, raw=raw)
        self._ch_reduce.print_nas_weights(eps, gumbel=gumbel, raw=raw)
        self._cls_stack.print_nas_weights(eps, gumbel=gumbel, raw=raw)


    def forward(self, x, inf_type=InferenceType.NORMAL):
        eps = self._t if inf_type == InferenceType.NORMAL else 1e-5
        time_out_x, time_out_lat, time_out_mem, last_ch_weights = self._time_reduce(x, eps=eps, inf_type=inf_type, last_ch_weights=None)
        ch_out_x, ch_out_lat, ch_out_mem, last_ch_weights = self._ch_reduce(time_out_x, eps, inf_type, last_ch_weights)
        # last_ch_weights = weight_softmax(last_ch_weights, eps, gumbel=inf_type is not InferenceType.MAX_WEIGHT)
        cls_out_x, cls_out_lat, cls_out_mem = self._cls_stack(ch_out_x, eps, inf_type, last_ch_weights)
        return cls_out_x, time_out_lat + ch_out_lat + cls_out_lat, torch.max(torch.stack([time_out_mem, ch_out_mem, cls_out_mem]))



    def getKeras(self, batch_size=1, getPruned=False, inf_type=InferenceType.NORMAL):
        k_input = keras.Input([*self._input_dims, 1], batch_size=batch_size)
        k_time_out = self._time_reduce.getKeras(k_input, getPruned=getPruned, inf_type=inf_type)
        k_ch_out = self._ch_reduce.getKeras(k_time_out, getPruned=getPruned, inf_type=inf_type)
        k_output = self._cls_stack.getKeras(k_ch_out, getPruned=getPruned)
        return keras.Model(k_input, k_output)
        # return keras.Model(k_input, k_time_out)




class ReduceCH(nn.Module):
    def __init__(self, ch_dim, in_ch, out_ch, reduce=False, granularity=8):
        super().__init__()

        self._ch_dim = ch_dim
        self._in_ch = in_ch
        self._out_ch = out_ch
        self._reduce = reduce
        self._granularity = granularity

        self.chs = ChooseNParallel_v2([
        Dyn_Conv2D(self._out_ch, self._out_ch, (3, 1), padding="same", granularity=granularity, activation="relu"),
        Dyn_Conv2D(self._out_ch, self._out_ch, (5, 1), padding="same", granularity=granularity, activation="relu"),
        Dyn_Conv2D(self._out_ch, self._out_ch, (7, 1), padding="same", granularity=granularity, activation="relu")
        ], granularity, self._out_ch)

        self._bottleNeck = Dyn_Conv2D(self._in_ch, self._out_ch, (1, self._ch_dim), stride=(1, 1 if not reduce else 2), granularity=granularity, activation="relu")
        
        self._res_con = Parallel_Choice_Add([
            # Zero(stride=(1, 1 if not reduce else 2), channels=self._out_ch), 
            Dyn_Conv2D(self._in_ch, self._out_ch, (1, 1 if not reduce else 2), stride=(1, 1 if not reduce else 2), activation="relu", granularity=granularity)],
            self._out_ch, granularity)

        if not reduce:
            self._skip_module = MakeChoice(2, alsoWeights=True)
            self._skip_zero = Id(self._in_ch, self._out_ch)

        self._dropout = nn.Dropout2d(0.3)

        # Weights for NAS
        self._weights_channels_bottleNeck = torch.zeros(self._out_ch // granularity, dtype=Config.tensor_dtype, requires_grad=True, device=Config.compute_unit)
        self._weights_channels_convBlock = torch.zeros(self._out_ch // granularity, dtype=Config.tensor_dtype, requires_grad=True, device=Config.compute_unit)

        self._weights_channels_bottleNeck_last = None
        self._weights_channels_convBlock_last = None

    def print_nas_weights(self, eps, gumbel, raw):
        print("----CH_Reduce----")
        print("Chose_Channels_bt: ", np.around(weight_softmax(self._weights_channels_bottleNeck, eps, gumbel=gumbel).cpu().detach().numpy(), 2))
        print("Chose_Channels_Conv: ", np.around(weight_softmax(self._weights_channels_convBlock, eps, gumbel=gumbel).cpu().detach().numpy(), 2))
        # print("weights_channels_bottleNeck_last: ", self._weights_channels_bottleNeck_last)
        # print("weights_channels_convBlock_last: ", self._weights_channels_convBlock_last)
        self.chs.print_nas_weights(eps, gumbel, raw)
        self._res_con.print_nas_weights(eps, gumbel, raw)
        if not self._reduce:
            self._skip_module.print_nas_weights(eps, gumbel, raw)


    def get_nas_weights(self):
        res = [self._weights_channels_bottleNeck, self._weights_channels_convBlock]
        res.extend(self.chs.get_nas_weights())
        res.extend(self._res_con.get_nas_weights())
        if not self._reduce:
            res.extend(self._skip_module.get_nas_weights())
        return res


    def forward(self, x, eps, last_ch_weights=None, inf_type=InferenceType.NORMAL):
        weights_ch_bt_softmax = weight_softmax(self._weights_channels_bottleNeck, eps)
        weights_ch_convBlock_softmax = weight_softmax(self._weights_channels_convBlock, eps)
        weights_lastCh_softmax = last_ch_weights

        if inf_type != inf_type.NORMAL:
            weights_ch_bt_softmax = torch.zeros_like(self._weights_channels_bottleNeck)
            weights_ch_convBlock_softmax = torch.zeros_like(self._weights_channels_convBlock)
            if inf_type == inf_type.MIN:
                weights_ch_bt_softmax[0] = 1
                weights_ch_convBlock_softmax[0] = 1
            if inf_type == inf_type.MAX:
                weights_ch_bt_softmax[-1] = 1
                weights_ch_convBlock_softmax[-1] = 1
            if inf_type == inf_type.SAMPLE:
                weights_ch_bt_softmax = weight_softmax(self._weights_channels_bottleNeck, eps, hard=True)
                self._weights_channels_bottleNeck_last = weights_ch_bt_softmax
                weights_ch_convBlock_softmax = weight_softmax(self._weights_channels_convBlock, eps, hard=True)
                self._weights_channels_convBlock_last = weights_ch_convBlock_softmax
            if inf_type == inf_type.RANDOM:
                self._weights_channels_bottleNeck = random_one_hot_like(self._weights_channels_bottleNeck)
                self._weights_channels_convBlock = random_one_hot_like(self._weights_channels_convBlock)
                weights_ch_bt_softmax = self._weights_channels_bottleNeck
                weights_ch_convBlock_softmax = self._weights_channels_convBlock
            if inf_type == InferenceType.MAX_WEIGHT:
                weights_ch_bt_softmax = weight_softmax(self._weights_channels_bottleNeck, eps, gumbel=False)
                weights_ch_convBlock_softmax = weight_softmax(self._weights_channels_convBlock, eps, gumbel=False)



        bt_out, bt_lat, bt_mem = self._bottleNeck(x, eps, weights=weights_ch_bt_softmax, last_ch_weights=weights_lastCh_softmax)
        #print("ch_weights: ", weights_ch_convBlock_softmax)

        conv_block_out, conv_block_lat, conv_block_mem, prob_no_op = self.chs(bt_out, eps, weights_ch_convBlock_softmax, weights_ch_bt_softmax, inf_type)
        conv_mem = torch.max(bt_mem, conv_block_mem)
        conv_lat = (bt_lat + conv_block_lat) * (1-prob_no_op)
        res_out = self._res_con((x, torch.tensor(0, dtype=Config.tensor_dtype) , torch.tensor(0, dtype=Config.tensor_dtype)), (conv_block_out, conv_lat, conv_mem), prob_no_op=prob_no_op,
        eps=eps, weights=weights_ch_convBlock_softmax, last_ch_weights=weights_lastCh_softmax, inf_type=inf_type)
        # Check if module should be skipped
        if not self._reduce:
            skip_zero = self._skip_zero(x)
            skip_out_res, skip_out_lat, skip_out_mem, skip_out_w = self._skip_module([skip_zero, res_out], eps, inf_type=inf_type, weights=[weights_lastCh_softmax, weights_ch_convBlock_softmax])
            # print(skip_out)
            return self._dropout(skip_out_res), skip_out_lat, skip_out_mem, skip_out_w
        res_out_res, res_out_lat, res_out_mem, res_out_w = (*res_out, weights_ch_convBlock_softmax)
        return self._dropout(res_out_res), res_out_lat, res_out_mem, res_out_w


    def getKeras(self, x, getPruned, inf_type=InferenceType.NORMAL):
        weights_ch_bt = weight_softmax(self._weights_channels_bottleNeck, 1e-9, gumbel=False)
        weights_ch_conv = weight_softmax(self._weights_channels_convBlock, 1e-9, gumbel=False)
        if inf_type != inf_type.NORMAL:
            weights_ch_bt = torch.zeros_like(self._weights_channels_bottleNeck)
            weights_ch_conv= torch.zeros_like(self._weights_channels_convBlock)
        if inf_type == inf_type.MIN:
            weights_ch_bt[0] = 1
            weights_ch_conv[0] = 1
        if inf_type == inf_type.MAX:
            weights_ch_bt[-1] = 1
            weights_ch_conv[-1] = 1
        if inf_type == inf_type.SAMPLE:
            weights_ch_bt = self._weights_channels_bottleNeck_last
            weights_ch_conv = self._weights_channels_convBlock_last
        if inf_type == inf_type.RANDOM:
            weights_ch_bt = self._weights_channels_bottleNeck_last
            weights_ch_conv = self._weights_channels_convBlock_last
        if inf_type == inf_type.MAX_WEIGHT:
            weights_ch_bt = weight_softmax(self._weights_channels_bottleNeck, 1e-9, gumbel=False)
            weights_ch_conv = weight_softmax(self._weights_channels_convBlock, 1e-9, gumbel=False)


        bt_out = self._bottleNeck.getKeras(x, getPruned, weights=weights_ch_bt, inf_type=inf_type)
        conv_block_out = self.chs.getKeras(bt_out, getPruned, weights=weights_ch_conv, inf_type=inf_type)
        res = self._res_con.getKeras(x, conv_block_out, getPruned=getPruned, weights=weights_ch_conv, inf_type=inf_type)
        if res is not None:
            res = L.Dropout(0.3)(res)
        if not self._reduce:
            res = self._skip_module.getKeras([x, res], getPruned=getPruned, inf_type=inf_type)
        return res
        


class ReduceTime(nn.Module):
    def __init__(self, channels, granularity, isInput=False) -> None:
        super().__init__()

        self._parallel_choice = ChooseNParallel([
            Dyn_Conv2D(1 if isInput else channels, channels, (3, 1), stride=(2, 1), activation="relu", granularity=granularity),
            Dyn_Conv2D(1 if isInput else channels, channels, (5, 1), stride=(2, 1), activation="relu", granularity=granularity),
            Dyn_Conv2D(1 if isInput else channels, channels, (7, 1), stride=(2, 1), activation="relu", granularity=granularity),
        ], 1, granularity, channels)

        self._dropout = nn.Dropout2d(0.3)

        self._weights_channels = torch.zeros(channels // granularity, dtype=Config.tensor_dtype, requires_grad=True, device=Config.compute_unit)

        self._weights_channels_last = None

    def print_nas_weights(self, eps, gumbel, raw):
        print("----Time_Reduce----")
        print("Choose_Channels: ", np.around(weight_softmax(self._weights_channels, eps, gumbel=gumbel).cpu().detach().numpy(), 2))
        # print("Weight_channel_last: ", np.around(weight_softmax(self._weights_channels_last, eps, gumbel=gumbel).cpu().detach().numpy(), 2))
        self._parallel_choice.print_nas_weights(eps, gumbel, raw)

    def get_nas_weights(self):
        res = [self._weights_channels]
        res.extend(self._parallel_choice.get_nas_weights())
        return res

        
    
    def forward(self, x, eps, inf_type, last_ch_weights):
        weight_channels_softmax = weight_softmax(self._weights_channels, eps)
        last_ch_weights_softmax = last_ch_weights
        if inf_type != inf_type.NORMAL:
            weight_channels_softmax = torch.zeros_like(self._weights_channels)
        if inf_type == inf_type.MIN:
            weight_channels_softmax[0] = 1
        if inf_type == inf_type.MAX:
            weight_channels_softmax[-1] = 1
        if inf_type == inf_type.SAMPLE:
            self._weights_channels_last = weight_softmax(self._weights_channels, eps, hard=True)
            weight_channels_softmax = self._weights_channels_last
        if inf_type == inf_type.RANDOM:
            self._weights_channels = random_one_hot_like(self._weights_channels)
            weight_channels_softmax = self._weights_channels
        if inf_type == inf_type.MAX_WEIGHT:
            weight_channels_softmax = weight_softmax(self._weights_channels, eps, gumbel=False)

        c_out, c_lat, c_mem = self._parallel_choice(x, eps, weight_channels_softmax, last_ch_weights_softmax, inf_type=inf_type)
        return self._dropout(c_out), c_lat, c_mem, weight_channels_softmax
    

    def getKeras(self, x, getPruned, inf_type=InferenceType.NORMAL):
        weight_channels_softmax = weight_softmax(self._weights_channels, 1e-9, gumbel=False)
        if inf_type != inf_type.NORMAL:
            weight_channels_softmax = torch.zeros_like(self._weights_channels)
        if inf_type == inf_type.MIN:
            weight_channels_softmax[0] = 1
        if inf_type == inf_type.MAX:
            weight_channels_softmax[-1] = 1
        if inf_type == inf_type.SAMPLE or inf_type == inf_type.RANDOM:
            weight_channels_softmax = self._weights_channels_last
        if inf_type == inf_type.MAX_WEIGHT:
            weight_channels_softmax = weight_softmax(self._weights_channels, 1e-9, gumbel=False)
        
        res = self._parallel_choice.getKeras(x, getPruned=getPruned, weights=weight_channels_softmax, inf_type=inf_type)

        res = L.Dropout(0.3)(res)
        
        return res