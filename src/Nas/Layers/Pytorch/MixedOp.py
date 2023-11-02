import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

import torch.nn.functional as F

from src.Nas.Layers.Pytorch.SqueezeExtract import SqueezeExtraction
from src.Nas.Layers.Pytorch.Common import Conv2D, MaxPool2D, Id, Zero

from tensorflow.keras.layers import add

class MixedOp(nn.Module):
    def __init__(self, ch_in, ch_out, parallel=1, filter_sizes=[3, 5, 7], padding="same", reduce=False) -> None:
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self._parallel = parallel
        self._reduce_factor = 2 if reduce else 1

        for f in filter_sizes:
            self.conv_layers.append(Conv2D(ch_in, ch_out, (f, 1), padding=padding, stride=(self._reduce_factor, 1), activation="relu"))
        
        for f in [5, 7, 9]:
            self.conv_layers.append(Conv2D(ch_in, ch_out, (self._reduce_factor, f), padding=padding, stride=(self._reduce_factor, 1), activation="relu"))
        
        if ch_in == ch_out and not reduce:
            self.conv_layers.append(Id(ch_in, ch_out))
            self.conv_layers.append(Zero())
            self.conv_layers.append(SqueezeExtraction(ch_in, ratio=4))

        self.bn = nn.BatchNorm2d(ch_out)

        self._weights = torch.autograd.Variable(1e-3*torch.randn(self._parallel, len(self.conv_layers)), requires_grad=True)
        self._weights.retain_grad()
    
    @property
    def nas_weights(self):
        return [self._weights]

    def get_num_weights(self):
        return len(self.conv_layers)


    def forward(self, x, useWeights=True, getPruned=False):
        
        # Get the one connection with the largest weight
        if getPruned:
            out_paths = []
            out_lat = []
            out_mem = []
            weights = self._weights if useWeights else torch.ones(self._parallel, len(self.conv_layers), dtype=float)
            max_w = torch.argmax(weights, dim=1)
            for idx, w in enumerate(max_w):
                res, lat, mem = self.conv_layers[w](x, onlyOutputs=idx != 0)
                out_paths.append(res)
                out_lat.append(lat)
                out_mem.append(mem)
                
            return torch.stack(out_paths).sum(dim=0), torch.stack(out_lat).sum(), torch.stack(out_mem).sum()
        
        # Turn on / off the weights
        weights = F.softmax(self._weights, dim=0) if useWeights else torch.ones(self._parallel, len(self.conv_layers), dtype=float)
        lat_acc, mem_acc = torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float)
        conv = []
        for l in self.conv_layers:
            conv.append(l(x))
        
        out_paths = []
        for weight in weights:
            stack = []
            lat_stack = []
            mem_stack = []
            for l, w in zip(conv, weight):
                res, lat, mem = l
                stack.append(res * w)
                mem_acc = torch.max(mem * w, mem_acc)
                lat_acc += lat * w
                lat_stack.append(lat * w)
                mem_stack.append(mem * w)
            # print([round(x) for x in torch.tensor(lat_stack).detach().numpy()])
            # print([round(x) for x in torch.tensor(mem_stack).detach().numpy()])
            out_paths.append(torch.stack(stack).sum(dim=0))
        x = torch.stack(out_paths).sum(dim=0)
        x = self.bn(x)
        return x, lat_acc, mem_acc

    def getKeras(self, x, useWeights=True, getPruned=False):
        weights = self._weights if useWeights else torch.ones(self._parallel, len(self.conv_layers), dtype=float)

        conv = []
        weights = weights.detach().numpy()

        for l in self.conv_layers:
            conv.append(l.getKeras(x))

        out_paths = []
        for weight in weights:
            stack = []
            if getPruned:
                max_w = np.argmax(weight)
                if not isinstance(self.conv_layers[max_w], Zero):
                    r = self.conv_layers[max_w].getKeras(x)
                    out_paths.append(r)
                continue
            for l, w in zip(conv, weight):
                stack.append(l * w)
            out_paths.append(add(stack))
        return add(out_paths) if len(out_paths) > 1 else out_paths[0]