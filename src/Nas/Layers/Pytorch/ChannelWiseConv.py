import torch
import torch.nn as nn

from src.Nas.Layers.Pytorch.Common import Conv2D, MaxPool2D

class ChannelConv(nn.Module):
    def __init__(self, ch_in, ch_out, filter_sizes=[3, 5, 7, 10, 20], padding="same", stride=1) -> None:
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.reduce = MaxPool2D((2, 1))

        for f in filter_sizes:
            self.conv_layers.append(Conv2D(ch_in, ch_out, (f, 1), padding=padding, stride=stride))

    def forward(self, x, weights=None, getPruned=False):
        if getPruned:
            max_w = torch.argmax(weights)
            conv_out, lat_conv, mem_conv = self.conv_layers[max_w](x)
            reduce_out, lat_reduce, mem_reduce = self.reduce(conv_out)
            return reduce_out, lat_conv + lat_reduce, torch.max(mem_conv, mem_reduce)
        if weights is None:
            weights = torch.ones(len(self.conv_layers), dtype=float)
        
        lat_acc, mem_acc = torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float)
        conv = []
        for i, (l, w) in enumerate(zip(self.conv_layers, weights)):
            res, lat, mem = l(x)
            conv.append(res * w)
            mem_acc += mem * w
            lat_acc += lat * w
        x = torch.stack(conv).sum(dim=0)
        red_out, lat_reduce, mem_reduce = self.reduce(x)
        lat_acc += lat_reduce
        mem_acc = torch.max(mem_reduce, mem_acc)

        return red_out, lat_acc, mem_acc