from src.Utilities import count_parameters
from src.Nas.Operations import OPS, GAP, BottleNeck

import tensorflow.keras as keras
from tensorflow.keras.layers import concatenate, Add, Conv1D, Conv2D, BatchNormalization, Concatenate, Dropout, Flatten, Dense, GlobalAveragePooling2D, multiply, Concatenate
import tensorflow as tf
from tensorflow.keras import backend as K

import torch
from torch import nn, softmax
from torch.autograd import Variable
import torch.nn.functional as F
from math import factorial

import numpy as np




class KerasModelNoAtt(tf.keras.Model):
    def __init__(self, nb_conv_layers, nb_conv_reduce, filter_num, filter_size, num_classes, add_instead_of_concat=False) -> None:
        super(KerasModelNoAtt, self).__init__()

        self.conv_modules = keras.Sequential()
        self.nb_conv_layers = nb_conv_layers
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.add_instead_of_concat = add_instead_of_concat
        self.reduce_conv = keras.Sequential()
        self.se_blocks = keras.Sequential()
        self.softmax = keras.layers.Softmax()

        self.conv1 = Conv2D(self.num_classes, (1, 1))
        self.gap = GlobalAveragePooling2D()

        for _ in range(nb_conv_layers):
            self.conv_modules.add(
                Conv2D(filter_num, (filter_size, 1), (2, 1), activation="relu"))
            self.conv_modules.add(BatchNormalization())

        for _ in range(nb_conv_reduce):
            self.reduce_conv.add(
                Conv2D(filter_num, (1, 4), (1, 1), activation="relu"))
            self.reduce_conv.add(BatchNormalization())
            self.reduce_conv.add(Dropout(0.2))

        self.dropout = Dropout(0.2)

    def call(self, x):
        x = self.conv_modules(x)
        x = self.dropout(x)
        x = self.reduce_conv(x)
        x = self.conv1(x)
        x = self.gap(x)
        x = self.softmax(x)
        return x

class inceptionTime(nn.Module):
    def __init__(self, numLayers, intputSize, hiddenSize) -> None:
        super().__init__()

        # Architecture stuff
        self.maxPool_Bottle = OPS.maxpool_bottle(intputSize, hiddenSize, 0)
        self.bottle = OPS.bottle_neck(intputSize, hiddenSize, 0)
        self.conv10 = OPS.conv_10(hiddenSize, hiddenSize, 0)
        self.conv20 = OPS.conv_20(hiddenSize, hiddenSize, 0)
        self.conv40 = OPS.conv_40(hiddenSize, hiddenSize, 0)
        self.gap = GAP(27, 4)

    def forward(self, x):
        max_bottle = self.maxPool_Bottle(x)
        bottle = self.bottle(x)
        out_1 = self.conv10(bottle)
        out_2 = self.conv20(bottle)
        out_3 = self.conv40(bottle)
        res = torch.cat((max_bottle, out_1, out_2, out_3), dim=(1))
        return self.gap(res)


class Module(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, useChannelsConv=False) -> None:
        super().__init__()
        self._useChannelsConv = useChannelsConv
        if useChannelsConv:
            self._ops = nn.ModuleList([
                OPS.maxpool_bottle(inputSize, outputSize, 0),
                OPS.channel_conv_3(inputSize, outputSize, 0),
                OPS.channel_conv_5(inputSize, outputSize, 0),
                OPS.channel_conv_7(inputSize, outputSize, 0)
            ])
        else:
            self._ops = nn.ModuleList([
                OPS.maxpool_bottle(inputSize, hiddenSize, 0),
                # OPS.bottle_neck(inputSize, hiddenSize, 0),
                # OPS.conv_10(inputSize, hiddenSize, 0),
                # OPS.conv_20(inputSize, hiddenSize, 0),
                # OPS.conv_40(inputSize, hiddenSize, 0),
                OPS.conv_3(inputSize, hiddenSize, 0),
                OPS.conv_5(inputSize, hiddenSize, 0),
                OPS.conv_7(inputSize, hiddenSize, 0)
            ])

        self.id = OPS.id(inputSize, hiddenSize, 0)
        self.zero = OPS.none(inputSize, hiddenSize, 0)
        self.bottleNeckOutput = OPS.bottle_neck(
            hiddenSize, outputSize, 0)

    def forward(self, weigths, x):
        layers = [w * op(x) for w, op in zip(weigths, self._ops)]
        stack_out = torch.stack(layers).sum(dim=0)
        if not self._useChannelsConv:
            stack_out = self.bottleNeckOutput(stack_out)
        x = stack_out
        return x, 0

    @staticmethod
    def get_num_choices():
        return 4

    def getKeras(self, x, weights, prune):
        weights = weights.detach().numpy()

        outputs = [op.getKeras()(x) for op in self._ops]
        if prune:
            maxIdx = np.argmax(weights)
            tmp = outputs[maxIdx]
        else:
            tmp = Add()(outputs)

        if self._useChannelsConv:
            return tmp
        return self.bottleNeckOutput.getKeras()(tmp)


class Network001(nn.Module):
    def __init__(self, numLayers, inputSize, hiddenSize, numClasses, backsteps=3) -> None:
        super().__init__()
        # Stuff for training
        self.criterion = nn.CrossEntropyLoss()
        self._numLayers = numLayers

        self._a = Variable(1e-3*torch.randn(numLayers, Module.get_num_choices()),
                           requires_grad=True)  # Set to true for training
        self._t = 1

        self._mods = nn.ModuleList()
        for _ in range(numLayers):
            self._mods.append(Module(inputSize, inputSize, hiddenSize))

        # Global Average Pooling for classification
        self.gap = GAP(inputSize, numClasses)

    def forward(self, x):
        for i in range(self._numLayers):
            w = F.softmax(self._a[i] / self._t, dim=-1)
            x = self._mods[i](w, x)
        x = self.gap(x)
        return x

    def getKeras(self, input_shape, prune=False):
        x = keras.Input(shape=input_shape)
        tmp = self._mods[0].getKeras(x, F.softmax(
            self._a[0] / self._t, dim=-1), prune)
        tmp = self._mods[1].getKeras(tmp, F.softmax(
            self._a[1] / self._t, dim=-1), prune)
        tmp = self._mods[2].getKeras(tmp, F.softmax(
            self._a[2] / self._t, dim=-1), prune)
        tmp = self.gap.getKeras()(tmp)
        model = keras.Model(inputs=x, outputs=tmp, name="NasNet")
        return model

    def _loss(self, input, target):
        logits = self(input)
        return self.criterion(logits, target)

    def _a_soft_tmp(self):
        return F.softmax(self._a / self._t, dim=-1)

    def set_alpha_grad(self, grad: bool):
        self.a.requires_grad = grad


class Network002(nn.Module):
    def __init__(self, numModules, inputSize, hiddenSize, numClasses, criterion, backsteps=3, numChannelsConv=None) -> None:
        super().__init__()
        self.criterion = criterion
        self._numModules = numModules
        self._numChannelsConv = numChannelsConv
        self._backsteps = backsteps

        num_ch_conv = 0 if numChannelsConv is None else numChannelsConv
        self._a = Variable(1e-3*torch.randn(self._numModules + num_ch_conv, Module.get_num_choices()),
                           requires_grad=True)  # Set to true for training
        self._t = 1

        self._mods = nn.ModuleList()
        self._mods_channels = nn.ModuleList()
        mods_output = hiddenSize if numChannelsConv else inputSize
        for _ in range(numModules):
            self._mods.append(
                Module(inputSize, mods_output, hiddenSize, False))

        self.scale_down = BottleNeck(inputSize, hiddenSize)

        if numChannelsConv:
            for _ in range(numChannelsConv):
                self._mods_channels.append(
                    Module(inputSize, inputSize, hiddenSize, True))

        # Global Average Pooling for classification
        self.gap = GAP(inputSize, numClasses)

    def forward(self, x):
        moduleCtr = 0
        if self._numChannelsConv:
            outputs = [x]
            for i in range(self._numChannelsConv):
                if (len(outputs) > 1):
                    out, mod_latency = self._mods_channels[i](
                        F.softmax(self._a[moduleCtr] / self._t, dim=-1), sum(outputs[-self._backsteps:]))
                else:
                    out, mod_latency = self._mods_channels[i](
                        F.softmax(self._a[moduleCtr] / self._t, dim=-1), outputs[0])
                outputs.append(out)
                moduleCtr += 1
            x = sum(outputs[-3:])

            x = self.scale_down(x)

        outputs = [x]
        for i in range(self._numModules):
            if (len(outputs) > 1):
                out, mod_latency = self._mods[i](
                    F.softmax(self._a[moduleCtr] / self._t, dim=-1), sum(outputs[-self._backsteps:]))
            else:
                out, mod_latency = self._mods[i](
                    F.softmax(self._a[moduleCtr] / self._t, dim=-1), outputs[0])
            outputs.append(out)
            moduleCtr += 1
        x = sum(outputs[-3:])

        x = self.gap(x)
        return x

    def getKeras(self, input_shape, prune=False):
        moduleCtr = 0
        x = keras.Input(shape=input_shape)
        outputs = [x]
        tmp_001 = None
        if self._numChannelsConv:
            for _ in range(self._numModules):
                if len(outputs) > 1:
                    tmp_input = Add()(outputs[-self._backsteps:])
                else:
                    tmp_input = outputs[0]
                outputs.append(self._mods_channels[moduleCtr].getKeras(
                    tmp_input, F.softmax(self._a[moduleCtr] / self._t, dim=-1), prune=prune))

            tmp_001 = Add()(outputs[-self._backsteps:])
            tmp_001 = self.scale_down.getKeras()(tmp_001)

        outputs = [tmp_001 if tmp_001 else x]
        for _ in range(self._numModules):
            if len(outputs) > 1:
                tmp_input = Add()(outputs[-self._backsteps:])
            else:
                tmp_input = outputs[0]
            outputs.append(self._mods[moduleCtr].getKeras(
                tmp_input, F.softmax(self._a[moduleCtr] / self._t, dim=-1), prune=prune))
            moduleCtr += 1
        tmp = Add()(outputs[-self._backsteps:])

        tmp = self.gap.getKeras()(tmp)
        model = keras.Model(inputs=x, outputs=tmp, name="NasNet")
        return model

    def _loss(self, input, target):
        logits = self(input)
        return self.criterion(logits, target, torch.tensor(1), torch.tensor(1))

    def _a_soft_tmp(self):
        return F.softmax(self._a / self._t, dim=-1)

    def set_alpha_grad(self, grad: bool):
        self._a.requires_grad = grad


if __name__ == "__main__":
    pass
    # # net = Network002(4, 18, 4, 6, 3)
    # # fake_input = torch.randn(1, 18, 128)
    # # print(net(fake_input).shape)
    # # net.getKeras((128, 18), prune=True).save(
    # #     "models/network_002_keras_prune.h5")
    # # # torch.onnx.export(net, fake_input, "models/torch_test_002.onnx")

    # net = KerasModel(4, 16, 5, 6)
    # net.build((1, 128,  9, 1))
    # fake_input = torch.randn((1, 128, 18, 1)).numpy()
    # print(net.summary())
    # print(net.predict(fake_input))
