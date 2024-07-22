import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from einops import rearrange, repeat

class SE_Block(nn.Module):
  "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
  def __init__(self, c, r=16):
      super().__init__()
      self.squeeze = nn.AdaptiveAvgPool2d(1)
      self.excitation = nn.Sequential(
          nn.Linear(c, c // r, bias=False),
          nn.ReLU(inplace=True),
          nn.Linear(c // r, c, bias=False),
          nn.Sigmoid()
      )

  def forward(self, x):
      bs, c, _, _ = x.shape
      y = self.squeeze(x).view(bs, c)
      y = self.excitation(y).view(bs, c, 1, 1)
      return x * y.expand_as(x)


class ALAE_TAE(nn.Module):
  def __init__(
      self,
      input_shape,
      nb_classes,
      individual_kernel_size=5,
      individual_conv_layer_nr=4,
      filter_nr = 64,


      activation="ReLU"):
    super(ALAE_TAE, self).__init__()
    self.input_shape = input_shape
    self.nb_classes = nb_classes
    self.sensor_channel = input_shape[3]
    self.individual_kernel_size = individual_kernel_size
    self.individual_conv_layer_nr = individual_conv_layer_nr

    feature_dim_list_1 = [1]
    for i in range(self.individual_conv_layer_nr):
      feature_dim_list_1.append(filter_nr)

    layers_conv_1 = []
    for i in range(self.individual_conv_layer_nr):
      layers_conv_1.append(
          nn.Sequential(nn.Conv2d(
              in_channels = feature_dim_list_1[i],
              out_channels = feature_dim_list_1[i+1],
              kernel_size = (individual_kernel_size, 1),
              stride = (1,1),
              padding = (int(individual_kernel_size/2),0)
              ),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(feature_dim_list_1[i+1])))
    self.layers_conv_1 = nn.ModuleList(layers_conv_1)

    self.SE_attention = SE_Block(filter_nr)

    #self.project = nn.Linear(self.sensor_channel * filter_nr, 3*filter_nr)

    # define lstm layers
    self.lstm_layers = []
    for i in range(2):
        if i == 0:
            self.lstm_layers.append(nn.LSTM(self.sensor_channel * filter_nr, filter_nr, batch_first =True))
        else:
            self.lstm_layers.append(nn.LSTM(filter_nr, filter_nr, batch_first =True))
    self.lstm_layers = nn.ModuleList(self.lstm_layers)

    # define dropout layer
    #self.dropout = nn.Dropout(self.drop_prob)

    # attention
    self.linear_1 = nn.Linear(filter_nr, filter_nr)
    self.tanh = nn.Tanh()
    self.dropout_2 = nn.Dropout(0.2)
    self.linear_2 = nn.Linear(filter_nr, 1, bias=False)
    # define classifier
    self.fc = nn.Linear(filter_nr, self.nb_classes)

  def forward(self,x):
    #x.shape batch 1 window_length, sensor_channel
    window_length = x.shape[2]
    batch_size = x.shape[0]

    for layer in self.layers_conv_1:
      x = layer(x)
      #print("x",x.shape)


    x = self.SE_attention(x)
    #print(x.shape)


    x = x.permute(0,2,1,3).reshape(batch_size, window_length, -1)
    for lstm_layer in self.lstm_layers:
        x, _ = lstm_layer(x)

    context = x[:, :-1, :]
    out = x[:, -1, :]

    uit = self.linear_1(context)
    uit = self.tanh(uit)
    uit = self.dropout_2(uit)
    ait = self.linear_2(uit)
    attn = torch.matmul(F.softmax(ait, dim=1).transpose(-1, -2),context).squeeze(-2)

    out = self.fc(out+attn)

    return out