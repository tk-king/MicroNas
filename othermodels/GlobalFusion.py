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

class GlobalFusion(nn.Module):
  def __init__(
      self,
      input_shape,
      S_number_sensors_type,
      L_sensor_locations,
      fft_segments_length,
      nb_classes,
      d_sensor_channel = 3,
      kernel_size_1 = 5,
      kernel_size_2 = 5,
      kernel_size_3 = 5,
      layer_nr_1 = 3,
      layer_nr_2 = 3,
      layer_nr_3 = 3,
      filter_nr = 64,
      activation="ReLU"):
    super(GlobalFusion, self).__init__()
    self.input_shape = input_shape
    self.S_number_sensors_type = S_number_sensors_type
    self.L_sensor_locations = L_sensor_locations
    self.fft_segments_length = fft_segments_length
    self.nr_segment = int(input_shape[2]/fft_segments_length)
    self.nb_classes = nb_classes
    self.d_sensor_channel = d_sensor_channel
    self.kernel_size_1 = kernel_size_1
    self.kernel_size_2 = kernel_size_2
    self.kernel_size_3 = kernel_size_3
    self.layer_nr_1 = layer_nr_1
    self.layer_nr_2 = layer_nr_2
    self.layer_nr_3 = layer_nr_3
    self.filter_nr = filter_nr
    self.scale = filter_nr ** -0.5
    total_sensor_channel_nr = input_shape[3]
    assert total_sensor_channel_nr==S_number_sensors_type*L_sensor_locations*d_sensor_channel

    # individual convolutional 1
    feature_dim_list_1 = [1]
    kernel_size_list_1 = [d_sensor_channel]
    stride_list_1 = [d_sensor_channel]
    # ------------------- gaile  for profiling ----------------------
    stride_list_1_1 = [1]
    for i in range(layer_nr_1):
      feature_dim_list_1.append(filter_nr)
      kernel_size_list_1.append(1)
      stride_list_1.append(1)
      stride_list_1_1.append(1)
    layers_conv_1 = []
    for i in range(layer_nr_1):
      layers_conv_1.append(
          nn.Sequential(nn.Conv2d(
              in_channels = feature_dim_list_1[i],
              out_channels = feature_dim_list_1[i+1],
              kernel_size = (kernel_size_1, kernel_size_list_1[i]),
              stride = (stride_list_1_1[i],stride_list_1[i]),
              padding = (0,0)
              ),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(feature_dim_list_1[i+1])))

    self.layers_conv_1 = nn.ModuleList(layers_conv_1)

    self.pos_q = nn.Conv2d( in_channels = filter_nr, out_channels = filter_nr, kernel_size = (1, 1),stride = (1,1),padding = (0,0))
    self.pos_k = nn.Conv2d( in_channels = filter_nr, out_channels = filter_nr, kernel_size = (1, 1),stride = (1,1),padding = (0,0))
    self.pos_v = nn.Conv2d( in_channels = filter_nr, out_channels = filter_nr, kernel_size = (1, 1),stride = (1,1),padding = (0,0))

    # individual convolutional 2
    feature_dim_list_2 = [filter_nr]
    kernel_size_list_2 = [L_sensor_locations]
    stride_list_2 = [1]
    for i in range(layer_nr_2):
      feature_dim_list_2.append(filter_nr)
      kernel_size_list_2.append(1)
      stride_list_2.append(1)
    layers_conv_2 = []
    for i in range(layer_nr_2):
      layers_conv_2.append(
          nn.Sequential(nn.Conv2d(
              in_channels = feature_dim_list_2[i],
              out_channels = feature_dim_list_2[i+1],
              kernel_size = (kernel_size_2, kernel_size_list_2[i]),
              stride = (1,stride_list_2[i]),
              padding = (int(kernel_size_2/2),0)
              ),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(feature_dim_list_2[i+1])))

    self.layers_conv_2 = nn.ModuleList(layers_conv_2)






    self.mod_q = nn.Conv2d( in_channels = filter_nr, out_channels = filter_nr, kernel_size = (1, 1),stride = (1,1),padding = (0,0))
    self.mod_k = nn.Conv2d( in_channels = filter_nr, out_channels = filter_nr, kernel_size = (1, 1),stride = (1,1),padding = (0,0))
    self.mod_v = nn.Conv2d( in_channels = filter_nr, out_channels = filter_nr, kernel_size = (1, 1),stride = (1,1),padding = (0,0))

    # individual convolutional 2
    feature_dim_list_3 = [filter_nr]
    kernel_size_list_3 = [S_number_sensors_type]
    stride_list_3 = [1]
    for i in range(layer_nr_3):
      feature_dim_list_3.append(filter_nr)
      kernel_size_list_3.append(1)
      stride_list_3.append(1)
    layers_conv_3 = []
    for i in range(layer_nr_3):
      layers_conv_3.append(
          nn.Sequential(nn.Conv2d(
              in_channels = feature_dim_list_3[i],
              out_channels = feature_dim_list_3[i+1],
              kernel_size = (kernel_size_3, kernel_size_list_3[i]),
              stride = (1,stride_list_3[i]),
              padding = (int(kernel_size_3/2),0)
              ),
          nn.ReLU(inplace=True),
          nn.BatchNorm2d(feature_dim_list_3[i+1])))

    self.layers_conv_3 = nn.ModuleList(layers_conv_3)
    temporal_dim = self.get_the_shape()

    self.proj = nn.Linear(temporal_dim*filter_nr,filter_nr)

    # define lstm layers
    self.gru_layers = []
    for i in range(2):
        if i == 0:
            self.gru_layers.append(nn.GRU(filter_nr, filter_nr, batch_first =True))
        else:
            self.gru_layers.append(nn.GRU(filter_nr, filter_nr, batch_first =True))
    self.gru_layers = nn.ModuleList(self.gru_layers)

    self.predict = nn.Linear(self.nr_segment*filter_nr,nb_classes)

  def get_the_shape(self):
    x=torch.rand(1, self.fft_segments_length*2, self.nr_segment, self.input_shape[3])
    window_length = x.shape[2]
    batch_size = x.shape[0]
    x_sl = x[:,:, 0:0+1,:].permute(0,2,1,3)

    for layer in self.layers_conv_1:
      x_sl = layer(x_sl)


    return x_sl.shape[2]

  def forward(self,x):


    B,_,_,C = x.shape
    # print("start",x.shape)
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

      x_sl = x[:,:, i:i+1,:].permute(0,2,1,3)

      for layer in self.layers_conv_1:
        x_sl = layer(x_sl)

      x_s_list = []
      # print("x_sl.shape",x_sl.shape)
      # print("self.S_number_sensors_type :",self.S_number_sensors_type)
      # print("self.L_sensor_locations :",self.L_sensor_locations)
      for j in range(self.S_number_sensors_type):
        x_s = x_sl[:,:,:,j*self.L_sensor_locations:(j+1)*self.L_sensor_locations]
        # print("x_s.shape :",x_s.shape)  
        x_s_temp = x_s

        for layer in self.layers_conv_2:
          x_s = layer(x_s)

        # print(x_s.shape)
        x_s_q = self.pos_q(x_s)


        x_s_k = self.pos_k(x_s_temp)


        x_s_v = self.pos_v(x_s_temp)



        dots = F.softmax(torch.matmul(x_s_q.reshape(batch_size,-1,1).permute(0,2,1), x_s_k.reshape(batch_size,-1,self.L_sensor_locations))  * self.scale, dim=-1)
        # print("------")
        # print(x_s_q.shape)
        # print(x_s_k.shape)
        # print(dots.shape)
        dots = dots.unsqueeze(1)
        # print(dots.shape)
        # print(x_s_v.shape)

        weighted_x_s_v = dots * x_s_v
        weighted_x_s_v = weighted_x_s_v.sum(dim=-1, keepdim=True)

        x_s = x_s_q + weighted_x_s_v

        x_s_list.append(x_s)
      x_merge = torch.cat(x_s_list,dim=-1)


      assert x_merge.shape[-1] == self.S_number_sensors_type
      x_temp = x_merge
      for layer in self.layers_conv_3:
        x_merge = layer(x_merge)



      x_q = self.mod_q(x_merge)


      x_k = self.mod_k(x_temp)


      x_v = self.mod_v(x_temp)



      dots = F.softmax(torch.matmul(x_q.reshape(batch_size,-1,1).permute(0,2,1), x_k.reshape(batch_size,-1,self.S_number_sensors_type))  * self.scale, dim=-1)
      dots = dots.unsqueeze(1)

      weighted_x_v = dots * x_v
      weighted_x_v = weighted_x_v.sum(dim=-1, keepdim=True)


      x_final = x_q + weighted_x_v

      temp_list.append(x_final)
    x = torch.cat(temp_list,dim=-1)


    x = x.permute(0,3,2,1).reshape(batch_size,window_length,-1)

    x = self.proj(x)

    for gru_layer in self.gru_layers:
      x, _ = gru_layer(x)


    y = self.predict(x.reshape(batch_size,-1))
    return y