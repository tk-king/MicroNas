from ast import Global
from unicodedata import name
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Nas.Layers.Pytorch.ParallelOp import ParallelOp
from src.Nas.Layers.Pytorch.MixedOp import MixedOp

from src.Nas.Operations import GAP
from src.Nas.Layers.Pytorch.ChannelWiseConv import ChannelConv
from src.Nas.Layers.Pytorch.SqueezeExtract import SqueezeExtraction
from src.Nas.Layers.Pytorch.Common import Conv2D, BatchNorm2D, GlobalAveragePooling, MaxPool2D, SoftMax, Id


import tensorflow.keras as keras


class Pytorch_SE_Net_v2(nn.Module):
    def __init__(self, nb_conv_layers, num_inputs, filter_num, filter_size, num_classes, parallel) -> None:
        super().__init__()

        self._num_inputs = num_inputs
        self._nb_conv_layers = nb_conv_layers
        self._num_classes = num_classes
        self._filter_num = filter_num
        self._filter_size = filter_size

        self._t = 1

        self.conv_modules = nn.ModuleList()
        self.reduce_modules = nn.ModuleList()

        for i in range(nb_conv_layers):
            num_k = min(((i // 3) + 1) * 4, 32)
            # print("idx: ", num_k)
            self.conv_modules.append(MixedOp(1 if i == 0 else filter_num, filter_num, padding="same", parallel=parallel, reduce=i%3==0))
            # self.conv_modules.append(ParallelOp(MixedOp, i, 4, filter_num))

        # for i in range(nb_conv_layers):
        #     if i < nb_conv_layers // 2:
        #         self.reduce_modules.append(MaxPool2D((2, 1)))
        #     else:
        #         self.reduce_modules.append(Id(filter_num, filter_num))

        self.cls_conv = Conv2D(filter_num, num_classes, (1, 1), activation="relu")
        self.cls_gap = GlobalAveragePooling()
        self.cls_softmax = SoftMax()

        
        # self._nas_weights = torch.autograd.Variable(1e-3*torch.randn((nb_conv_layers, self.conv_modules[1].get_num_weights())))
        self._nas_input_weights = torch.autograd.Variable(1e-3*torch.randn((num_inputs, 4)), requires_grad=True)

    @property
    def nas_weights(self):
        mixed_op_weights = []
        for op in self.conv_modules:
            mixed_op_weights.extend(op.nas_weights)

        return [self._nas_input_weights, *mixed_op_weights]

    def printNasWeights(self):
        weights = self.nas_weights
        for w in weights:
            print(F.softmax(w))

    def forward(self, inputs, getPruned=False, useWeights=True):
        lat_acc, mem_acc = torch.tensor(0, dtype=torch.float), torch.tensor(0)

        if not isinstance(inputs, list):
            inputs = [inputs]

        x = inputs[0]
        ctr = 1
        for i, conv_layer in enumerate(self.conv_modules):
            
            if ctr < len(inputs) and inputs[ctr].shape[2:] == x.shape[2:]:
                w = self._nas_input_weights[ctr - 1]
                x = w[0] * x + w[1] * inputs[ctr]
                ctr += 1

            x, lat_conv_block, mem_conv_block = conv_layer(x, useWeights=useWeights, getPruned=getPruned)  # Conv
            # print("x_shape: ", x.shape)
            # if i < len(self.reduce_modules):
            #     x, lat_reduce, mem_reduce = red(x)
            # else:
            #     lat_reduce, mem_reduce = torch.tensor(0), torch.tensor(0)

            lat_acc += lat_conv_block
            mem_acc = torch.max(torch.stack((mem_conv_block, mem_acc)))

        x, lat_cls_conv, mem_cls_conv = self.cls_conv(x)
        x, lat_gap, mem_gap = self.cls_gap(x)
        x, lat_softmax, mem_softmax = self.cls_softmax(x)
        lat_acc += lat_cls_conv + lat_gap + lat_softmax
        return x, lat_acc, torch.max(torch.stack((mem_acc, mem_cls_conv, mem_gap, mem_softmax)))


    def getKeras(self, input_shape, getPruned=False):

        chosen = 0
        for i, elm in enumerate(self._nas_input_weights):
            if elm[0] < elm[1]:
                chosen = i
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        ctr = 1
        inputs = [keras.Input(shape=i_shape, batch_size=1) for i_shape in input_shape]
        if getPruned:
            inputs = [inputs[chosen]]
        for i, conv_layer in enumerate(self.conv_modules):
            x = x if i != 0 else inputs[0]
            if ctr < len(inputs) and inputs[ctr].shape[1:2] == x.shape[1:2]:
                w_input = self._nas_input_weights[ctr - 1]
                x = w_input[0] * x + w_input[1] * inputs[ctr]
                if getPruned and ctr == chosen:
                    x = inputs[ctr] 
                ctr += 1

            x = conv_layer.getKeras(x, useWeights=True, getPruned=getPruned)  # Conv

        x = self.cls_conv.getKeras(x)
        x = self.cls_gap.getKeras(x)
        x = self.cls_softmax.getKeras(x)

        model = keras.Model(inputs=inputs, outputs=x, name="mnist_model")
        return model