import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from einops import rearrange, repeat


def extract_blocks_with_overlap_torch(input_segs, T_size):

  B, F, T, C = input_segs.size()
  overlap = int(T_size)

  overlap = 0
  blocks = []
  step = T_size - overlap

  for b in range(B):  # 遍历每张图片
    seg_blocks = []
    for j in range(0, C, 1):
      for i in range(0, T - T_size + 1, step):
        block = input_segs[b:b+1, :, i:i+T_size, j:j+1]


        seg_blocks.append(block)
    blocks.append(torch.cat(seg_blocks ,dim=-1 ))


  return torch.cat(blocks)

class DeepSense(nn.Module):
  def __init__(
      self,
      input_shape,
      fft_segments_length,
      k_number_sensors_group,
      nb_classes,
      d_sensor_channel = 3,
      kernel_size_1 = 5,
      kernel_size_2 = 5,
      layer_nr_1 = 3,
      layer_nr_2 = 3,
      filter_nr = 64,


      activation="ReLU"):
    super(DeepSense, self).__init__()

    self.input_shape = input_shape
    self.nb_classes = nb_classes
    self.k_number_sensors_group = k_number_sensors_group
    self.fft_segments_length = fft_segments_length
    #self.nr_segment = int(2*input_shape[2]/fft_segments_length-1)
    self.nr_segment = int(input_shape[2]/fft_segments_length)
    batch, _, window_length, total_sensor_channel = input_shape
    assert int(total_sensor_channel/k_number_sensors_group)==d_sensor_channel

    # individual convolutional 1
    feature_dim_list = [1]
    kernel_size_list = [d_sensor_channel]
    stride_list = [d_sensor_channel]
    stride_list_0 = [2]
    for i in range(layer_nr_1):
      feature_dim_list.append(filter_nr)
      kernel_size_list.append(1)
      stride_list.append(1)
      stride_list_0.append(1)
    layers_conv = []
    for i in range(layer_nr_1):
      layers_conv.append(
          nn.Sequential(nn.Conv2d(
              in_channels = feature_dim_list[i],
              out_channels = feature_dim_list[i+1],
              kernel_size = (kernel_size_1, kernel_size_list[i]),
              stride = (stride_list_0[i],stride_list[i]),
              padding = (0,0)
              ),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(feature_dim_list[i+1])))

    self.layers_conv = nn.ModuleList(layers_conv)


    #self.flatten = nn.Flatten()

    # individual convolutional 1

    kernel_size_list_2 = [k_number_sensors_group]
    feature_dim_list_2 = [1]
    stride_list_2 = [2]
    for i in range(layer_nr_1):
      kernel_size_list_2.append(1)
      feature_dim_list_2.append(filter_nr)
      stride_list_2.append(1)

    layers_conv_2 = []
    for i in range(layer_nr_2):
      layers_conv_2.append(
          nn.Sequential(nn.Conv2d(
              in_channels = feature_dim_list_2[i],
              out_channels = feature_dim_list_2[i+1],
              kernel_size = (kernel_size_2, kernel_size_list_2[i]),
              # ------------------------ gai le ------------------- for profiling
              stride = (stride_list_2[i],1),
              padding = (0,0)
              ),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(feature_dim_list_2[i+1])))

    self.layers_conv_2 = nn.ModuleList(layers_conv_2)

    fusion_dim = self.get_the_shape()
    self.linear_proj = nn.Linear(fusion_dim,filter_nr)

    # define lstm layers
    self.gru_layers = []
    for i in range(2):
        if i == 0:
            self.gru_layers.append(nn.GRU(filter_nr, filter_nr, batch_first =True))
        else:
            self.gru_layers.append(nn.GRU(filter_nr, filter_nr, batch_first =True))
    self.gru_layers = nn.ModuleList(self.gru_layers)


    # attention
    self.linear_1 = nn.Linear(filter_nr, filter_nr)
    self.tanh = nn.Tanh()
    self.dropout_2 = nn.Dropout(0.2)
    self.linear_2 = nn.Linear(filter_nr, 1, bias=False)
    # define classifier
    self.fc = nn.Linear(filter_nr, self.nb_classes)



  def get_the_shape(self):
    x=torch.rand(1, self.fft_segments_length*2, self.nr_segment, self.input_shape[3])
    temp = x[:,:, 0:1,:].permute(0,2,1,3)
    for layer in self.layers_conv:
      temp = layer(temp)

    temp = temp.permute(0,3,2,1).reshape(1,self.k_number_sensors_group,-1)

    temp = temp.unsqueeze(1).permute(0,1,3,2)

    for layer in self.layers_conv_2:
      temp = layer(temp)

    assert temp.shape[3]==1
    temp = temp.permute(0,3,2,1).reshape(1,1,-1)
    return temp.shape[2]

  def forward(self,x):
    B,_,_,C = x.shape
    x = extract_blocks_with_overlap_torch(x, self.fft_segments_length)
    #x.shape batch, 1, fft_segments_length, sensor_channel*nr_sgments
    x = torch.cat([torch.fft.fft(x.permute(0,1,3,2), dim=-1).real, torch.fft.fft(x.permute(0,1,3,2), dim=-1).imag],dim=-1)
    #x.shape batch, 1, sensor_channel*nr_sgments,fft_segments_length*2
    x = x.reshape(B,C,self.nr_segment,-1)

    x = x.permute(0,3,2,1)


    window_length = x.shape[2]
    batch_size = x.shape[0]
    temp_list = []
    for i in range(window_length):
      temp = x[:,:, i:i+1,:].permute(0,2,1,3)

      for layer in self.layers_conv:
        temp = layer(temp)

      temp = temp.permute(0,3,2,1).reshape(batch_size,self.k_number_sensors_group,-1)

      temp = temp.unsqueeze(1).permute(0,1,3,2)

      for layer in self.layers_conv_2:
        temp = layer(temp)

      assert temp.shape[3]==1
      temp = temp.permute(0,3,2,1).reshape(batch_size,1,-1)

      temp_list.append(temp)
    x = torch.cat(temp_list,dim=1)

    x = self.linear_proj(x)

    for gru_layer in self.gru_layers:
      x, _ = gru_layer(x)



    context = x[:, :-1, :]
    out = x[:, -1, :]

    uit = self.linear_1(context)
    uit = self.tanh(uit)
    uit = self.dropout_2(uit)
    ait = self.linear_2(uit)
    attn = torch.matmul(F.softmax(ait, dim=1).transpose(-1, -2),context).squeeze(-2)

    out = self.fc(out+attn)

    return out