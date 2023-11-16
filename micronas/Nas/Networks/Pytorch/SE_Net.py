from ast import Global
import torch
import torch.nn as nn

from micronas.Nas.Operations import GAP
from micronas.Nas.Layers.Pytorch.ChannelWiseConv import ChannelConv
from micronas.Nas.Layers.Pytorch.SqueezeExtract import SqueezeExtraction
from micronas.Nas.Networks.Keras.SE_Net import Keras_SE_Net
from micronas.Nas.Layers.Pytorch.Common import Conv2D, BatchNorm2D, GlobalAveragePooling, SoftMax


import numpy as np

import torch.nn.functional as F


class Pytorch_SE_Net(nn.Module):
    def __init__(self, nb_conv_layers, nb_conv_reduce, filter_num, filter_size, num_classes) -> None:
        super().__init__()

        self._nb_conv_layers = nb_conv_layers
        self._nb_conv_reduce = nb_conv_reduce
        self._num_classes = num_classes
        self._filter_num = filter_num
        self._filter_size = filter_size
        self._t = 1

        self._nas_weights = torch.autograd.Variable(
            1e-3*torch.randn((nb_conv_layers, 5)))

        self._t = 1 # Temperature

        self.conv_modules = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        for i in range(nb_conv_layers):
            self.conv_modules.add_module(f"conv_block_{i}",
                                         nn.ModuleList([ChannelConv(1 if i == 0 else filter_num, filter_num),
                                                        BatchNorm2D(
                                                            filter_num)
                                                        ]))

        for i in range(nb_conv_reduce):
            self.se_blocks.add_module(f"se_block_{i}", nn.ModuleList([
                SqueezeExtraction(filter_num, 4),
                Conv2D(filter_num, filter_num, (1, 3), (1, 1)),
                BatchNorm2D(filter_num),
                nn.ReLU()
            ]))

        self.dropout = nn.Dropout(0.2)
        self.cls_conv = Conv2D(filter_num, num_classes, (1, 1))
        self.cls_gap =  GlobalAveragePooling()
        self.cls_relu = nn.ReLU()
        self.cls_softmax = SoftMax()

    def forward(self, x, getPruned=False):
        lat_acc, mem_acc = torch.tensor(0, dtype=torch.float), torch.tensor(0)

        for conv_layer, w in zip(self.conv_modules, self._nas_weights):
            w_softmax = F.gumbel_softmax(w, tau=self._t, dim=0)
            conv_out, lat_conv_block, mem_conv_block = conv_layer[0](
                x, weights=w_softmax, getPruned=getPruned)  # Conv
            x, lat_bn, mem_bn = conv_layer[1](conv_out)  # BN
            lat_acc += (lat_conv_block + lat_bn)
            mem_acc = torch.max(torch.stack((mem_conv_block, mem_bn, mem_acc)))

        x = self.dropout(x)
        for l in self.se_blocks:
            x, lat_se, mem_se = l[0](x)  # se
            x, lat_conv, mem_conv = l[1](x)  # conv
            b_norm, lat_bn, mem_bn = l[2](x)  # batchnorm
            x = l[3](b_norm)  # relu: is integrated into other operation in keras
            lat_acc += (lat_se + lat_conv + lat_bn)
            mem_acc = torch.max(torch.stack((mem_se, mem_conv, mem_bn, mem_acc)))

        x, lat_cls_conv, mem_cls_conv = self.cls_conv(x)
        x = self.cls_relu(x)
        x, lat_gap, mem_gap = self.cls_gap(x)
        x, lat_softmax, mem_softmax = self.cls_softmax(x)
        lat_acc += lat_cls_conv + lat_gap + lat_softmax
        return x, lat_acc, torch.max(torch.stack((mem_acc, mem_cls_conv, mem_gap, mem_softmax)))

    def set_alpha_grad(self, grad: bool):
        self._nas_weights.requires_grad = grad

    def _a_soft_tmp(self):
        return F.softmax(self._nas_weights / self._t, dim=-1)

    def getKeras(self, input_shape, getPruned=True):
        input_shape = (input_shape[0], input_shape[2],
                       input_shape[3], input_shape[1])
        k_model = Keras_SE_Net(self._nb_conv_layers, self._nb_conv_reduce, self._filter_num,
                               self._filter_size, self._num_classes, weights=self._nas_weights, getPruned=getPruned)
        k_model.compute_output_shape(input_shape=input_shape)
        return k_model
