import torch.nn as nn
import torch
from tensorflow.keras.layers import add

class ParallelOp(nn.Module):
    def __init__(self, num_parallel, num_connections):
        super().__init__()
        self._num_parallel = num_parallel
        self._num_connections = num_connections

        self._weights = Variable(1e-3*torch.randn((num_parallel, num_connections)))


    def forward(self, x, useWeights=True, getPruned=False):
        assert isinstance(x, list), "Input needs to be a list"
        assert len(x) == self._num_parallel
        stack = []
        for weight in self._weights:
            stack.append(sum([l * w for l, w in zip(x, weight)]))
        
        

    
