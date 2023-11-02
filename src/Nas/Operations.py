from tokenize import group

from numpy import dtype
from src.Utilities import DotDict
import torch
from torch import max_pool2d, nn
from math import ceil, floor
import torch.nn.functional as F
from torch.autograd import Variable
from tensorflow.keras.layers import Conv2D

from src.Nas.Utils import getPadding


# Tensorflow stuff
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D, MaxPool1D, concatenate, GlobalAveragePooling1D, Softmax, DepthwiseConv1D


class Id(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

    def getKeras(self):
        return keras.layers.Layer()  # Default is id in keras


class Zero(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class MaxPoolBottle(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.maxPool(x)
        x = self.conv(x)
        return self.activation(x)

    def getKeras(self):
        return keras.Sequential([
            MaxPool1D(3, 1, "same"),
            Conv1D(self._out_channels, 1, activation="relu")
        ])


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self._out_channels = out_channels
        self.bottle = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.bottle(x)
        return self.activation(x)

    def getKeras(self):
        return Conv1D(self._out_channels, 1, activation="relu")


class GAP(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self._out_channels = out_channels
        self.conv = nn.Conv1d(
            kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = x.mean(dim=(-1))
        return self.softmax(x)

    def getKeras(self):
        return keras.Sequential([
            Conv1D(self._out_channels, 1, strides=1, activation="relu"),
            GlobalAveragePooling1D(),
            Softmax()
        ])


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = F.pad(x, (*self.padding, 0, 0))
        x = self.conv(x)
        return self.activation(x)

    def getKeras(self):
        return Conv1D(self._out_channels, self._kernel_size, activation="relu", padding="same")


class DepSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.depthConv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, groups=in_channels, padding='same')
        self.reduceConv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.depthConv(x)
        x = self.reduceConv(x)
        return self.activation(x)




OPS = {
    "bottle_neck": lambda in_channels, out_channels, stride: BottleNeck(in_channels, out_channels),
    "maxpool_bottle": lambda in_channels, out_channels, stride: MaxPoolBottle(in_channels, out_channels, stride),

    "id": lambda in_channels, out_channels, stride: Id(),
    "none": lambda in_channels, out_channels, stride: Zero(in_channels, out_channels, stride),

    "conv_3": lambda in_channels, out_channels, stride: Conv(in_channels, out_channels, 3),
    "conv_5": lambda in_channels, out_channels, stride: Conv(in_channels, out_channels, 5),
    "conv_7": lambda in_channels, out_channels, stride: Conv(in_channels, out_channels, 7),
    "conv_10": lambda in_channels, out_channels, stride: Conv(in_channels, out_channels, 10),
    "conv_20": lambda in_channels, out_channels, stride: Conv(in_channels, out_channels, 20),
    "conv_40": lambda in_channels, out_channels, stride: Conv(in_channels, out_channels, 40),

    "channel_conv_3": lambda in_channels, out_channels, stride: ChannelConv(in_channels, in_channels, 3),
    "channel_conv_5": lambda in_channels, out_channels, stride: ChannelConv(in_channels, in_channels, 5),
    "channel_conv_7": lambda in_channels, out_channels, stride: ChannelConv(in_channels, in_channels, 7),


    "sep_conv_10": lambda in_channels, out_channels, stride: DepSepConv(in_channels, out_channels, 10),
    "sep_conv_20": lambda in_channels, out_channels, stride: DepSepConv(in_channels, out_channels, 20),
    "sep_conv_40": lambda in_channels, out_channels, stride: DepSepConv(in_channels, out_channels, 40),

    "max_pool_10": lambda in_channels, out_channels, stride: nn.MaxPool1d(kernel_size=10),
    "max_pool_20": lambda in_channels, out_channels, stride: nn.MaxPool1d(kernel_size=20),
    "max_pool_30": lambda in_channels, out_channels, stride: nn.MaxPool1d(kernel_size=30)
}
OPS = DotDict(OPS)
