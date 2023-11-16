import torch
import torch.nn as nn

from micronas.Nas.Layers.Pytorch.Common import Add, Conv2D, Dense, GlobalAveragePooling, Mulitply


class SqueezeExtraction(nn.Module):
    def __init__(self, ch, ratio=16) -> None:
        super().__init__()
        self.channel_se = Channel_Squeeze_Extraction(ch, ratio)
        self.spacial_se = Spacial_Squeeze_Extraction(ch, ratio)
        self.add = Add()

    def forward(self, inputs, onlyOutputs=False):
        channel_se_x, channel_se_lat, channel_se_mem = self.channel_se(inputs)
        spacial_se_x, spacial_se_lat, spacial_se_mem = self.spacial_se(inputs)
        res, lat_add, mem_add = self.add(channel_se_x, spacial_se_x)
        lat_sum = channel_se_lat + spacial_se_lat + lat_add
        mem_sum = torch.max(torch.stack((channel_se_mem, spacial_se_mem, mem_add)))
        return res, lat_sum, mem_sum
    
    def getKeras(self, x):
        x1 = self.channel_se.getKeras(x)
        x2 = self.spacial_se.getKeras(x)
        return self.add.getKeras(x1, x2)

class Channel_Squeeze_Extraction(nn.Module):
    def __init__(self, ch, ratio) -> None:
        super().__init__()
        self.dense_001 = Dense(ch, ch // ratio, activation="relu")
        self.dense_002 = Dense(ch // ratio, ch, activation="sigmoid")
        self.gap = GlobalAveragePooling()
        self.mul = Mulitply()


    def forward(self, inputs):
        squeeze_tensor, lat_squeeze, mem_squeeze = self.gap(inputs)
        dense_001, lat_dense_001, mem_dense_001 = self.dense_001(squeeze_tensor)
        x, lat_dense_002, mem_dense_002 = self.dense_002(dense_001)
        a, b = squeeze_tensor.size()
        mul_input = x.view(a, b, 1, 1)
        x, lat_multiply, mem_multiply = self.mul(inputs, mul_input)
        lat_sum = lat_squeeze + lat_dense_001 + lat_dense_002 + lat_multiply
        mem_sum = torch.max(torch.stack((mem_squeeze, mem_dense_001, mem_dense_002, mem_multiply)))
        return x, lat_sum, mem_sum

    def getKeras(self, inputs):
        x = self.gap.getKeras(inputs)
        x = self.dense_001.getKeras(x)
        x = self.dense_002.getKeras(x)
        return self.mul.getKeras(x, inputs)



class Spacial_Squeeze_Extraction(nn.Module):
    def __init__(self, ch, ratio) -> None:
        super().__init__()
        self.conv = Conv2D(ch, 1, (1, 1), activation="sigmoid")
        self.mul = Mulitply()

    def forward(self, inputs):
        x_conv, lat_conv, mem_conv = self.conv(inputs)
        x, lat_multiply, mem_multiply = self.mul(inputs, x_conv)
        return x, lat_conv + lat_multiply, torch.max(mem_conv, mem_multiply)

    def getKeras(self, x):
        x_conv = self.conv.getKeras(x)
        x = self.mul.getKeras(x, x_conv)
        return x