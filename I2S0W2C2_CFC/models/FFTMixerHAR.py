from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class MLP(nn.Module):
  def __init__(self, num_features, expansion_factor=1, dropout=0):
    super().__init__()
      
    num_hidden = num_features * expansion_factor
    self.fc1 = nn.Linear(num_features, num_hidden)


    self.dropout1 = nn.Dropout(dropout)
    self.fc2 = nn.Linear(num_hidden, num_features)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x):

      # 激活函数有问题
      x = self.dropout1(F.gelu(self.fc1(x)))
      x = self.dropout2(self.fc2(x))

      return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, segments_nr, sensor_channel, expansion_factor, dropout):
        super().__init__()
        # along temporal dimension
        self.num_features = num_features
        self.segments_nr  = segments_nr
        self.sensor_channel = sensor_channel
        self.norm1 = nn.LayerNorm(num_features*segments_nr)
        self.mlp = MLP(num_features*segments_nr, expansion_factor, dropout)
        self.norm2 = nn.LayerNorm(num_features*segments_nr)

    def forward(self, x):

        batch_size = x.shape[0]
        
        # residual.shape == (batch_size, num_features, temporal_length, sensor_channel)


        x = x.permute(0,3,2,1).reshape(batch_size,self.sensor_channel,-1)
        residual = x
        # x.shape == (batch_size, sensor_channel, temporal_length, num_features)
        # x.shape == (batch_size, sensor_channel, temporal_length*num_features)

        x = self.norm1(x)
        #x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        #x = x.transpose(1, 2)
        # # x.shape == (batch_size, sensor_channel, num_features*temporal_length)
        # x.shape == (batch_size, num_features, temporal_length, sensor_channel)
  
        
        out = x + residual
        
        out = self.norm2(out)

        out = out.reshape(batch_size,self.sensor_channel,self.segments_nr,self.num_features).permute(0,3,2,1)
        return out

class ChannelMixer(nn.Module):
    def __init__(self, num_features, segments_nr, sensor_channel, expansion_factor, dropout):
        super().__init__()
        #self.norm = nn.LayerNorm(num_features)
        self.num_features = num_features
        self.segments_nr  = segments_nr
        self.sensor_channel = sensor_channel
        self.mlp = MLP(num_features*sensor_channel, expansion_factor, dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        residual = x
        # residual.shape == (batch_size, num_features, temporal_length, sensor_channel)


        x = x.permute(0,2,3,1).reshape(batch_size,self.segments_nr,-1)
        # x.shape == (batch_size, temporal_length, num_features, sensor_channel)
        # x.shape == (batch_size, temporal_length, num_features*sensor_channel)

        #x = self.norm(x)

        x = self.mlp(x)
        # x.shape == (batch_size, temporal_length, num_features*sensor_channel)
        x = x.reshape(batch_size,self.segments_nr,self.sensor_channel,self.num_features).permute(0,3,1,2)


        out = x + residual
        return out