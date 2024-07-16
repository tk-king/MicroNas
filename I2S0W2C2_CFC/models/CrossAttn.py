import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from einops import rearrange, repeat


# Individual Convolution
class individual_convolution(nn.Module):
  def __init__(self, input_shape, hidden_dim, layer_nr, kernel_size, padding_size, stride_size, activation="ReLU"):
    super(individual_convolution, self).__init__()

    self.input_shape   = input_shape
    self.feature_dims = [input_shape[1]]
    for _ in range(layer_nr):
        self.feature_dims.append(hidden_dim)

    self.hidden_dim   = hidden_dim
    self.layer_nr     = layer_nr
    self.kernel_size   = kernel_size
    if padding_size is None:
        padding_size = int(kernel_size/2)
    self.padding_size  = padding_size
    self.stride_size   = stride_size

    self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

    stride_size_list = [1,2,1,2]
    layers_conv = []
    for i in range(layer_nr):
        layers_conv.append(
            nn.Sequential(nn.Conv2d(
                in_channels = self.feature_dims[i],
                out_channels = self.feature_dims[i+1],
                kernel_size = (kernel_size, 1),
                stride = (stride_size_list[i],1),
                padding = (padding_size,0)
                ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.feature_dims[i+1])))

    self.layers_conv = nn.ModuleList(layers_conv)

  def forward(self, x):
    for layer in self.layers_conv:
        x = layer(x)

    return x



class SelfAttention_interaction(nn.Module):

  def __init__(self, sensor_channel, n_channels):
      super(SelfAttention_interaction, self).__init__()

      self.query         = nn.Linear(n_channels, n_channels, bias=False)
      self.key          = nn.Linear(n_channels, n_channels, bias=False)
      self.value         = nn.Linear(n_channels, n_channels, bias=False)
      self.gamma         = nn.Parameter(torch.tensor([0.]))



  def forward(self, x):

      # 输入尺寸是 batch  sensor_channel feature_dim
      #print(x.shape)

      f, g, h = self.query(x), self.key(x), self.value(x)

      beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)

      o = self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta) + x.permute(0, 2, 1).contiguous()
      o = o.permute(0, 2, 1).contiguous()


      return o



class embedding_block(nn.Module):
  def __init__(self, input_shape, hidden_dim, layer_nr, kernel_size, padding_size, stride_size, activation="ReLU"):
    super(embedding_block, self).__init__()
    self.input_shape   = input_shape
    self.hidden_dim   = hidden_dim
    self.layer_nr     = layer_nr
    self.kernel_size   = kernel_size
    if padding_size is None:
      padding_size = int(kernel_size/2)
    self.padding_size  = padding_size
    self.stride_size   = stride_size

    self.individual_conv_block = individual_convolution(input_shape, hidden_dim, layer_nr, kernel_size, padding_size, stride_size, activation)
    self.cross_channel_attnetion = SelfAttention_interaction(input_shape[3], hidden_dim)
    self.channel_fusion =  nn.Linear( input_shape[3]*hidden_dim,2*hidden_dim)
    self.activation = nn.ReLU()
  def forward(self,x):

    """ =============== Individual block ==============="""
    x = self.individual_conv_block(x)
    x = x.permute(0,3,2,1)
    # ------->  B x C x L* x F*
    """ =============== cross channel interaction ==============="""

    x = torch.cat(
        [self.cross_channel_attnetion(x[:, :, t, :]).unsqueeze(3) for t in range(x.shape[2])],
        dim=-1,
    )
    # ------->  B x C x F* x L*
    """=============== cross channel fusion ==============="""

    x = x.permute(0, 3, 1, 2)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = self.activation(self.channel_fusion(x))


    return x




class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout = 0.):
    super().__init__()
    self.net = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class Attention(nn.Module):
  def __init__(self, dim, heads = 1, dim_head = 64, dropout = 0.):
    super().__init__()
    assert heads==1
    inner_dim = dim_head *  heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.norm = nn.LayerNorm(dim)

    self.attend = nn.Softmax(dim = -1)
    self.dropout = nn.Dropout(dropout)

    #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
    self.to_q = nn.Linear(dim, inner_dim , bias = False)
    self.to_k = nn.Linear(dim, inner_dim , bias = False)
    self.to_v = nn.Linear(dim, inner_dim , bias = False)

    self.to_out = nn.Sequential(
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    ) if project_out else nn.Identity()

  def forward(self, x):
    x = self.norm(x)

    #qkv = self.to_qkv(x).chunk(3, dim = -1)
    #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

    q= self.to_q(x).unsqueeze(1)
    k= self.to_k(x).unsqueeze(1)
    v= self.to_v(x).unsqueeze(1)

    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

    attn = self.attend(dots)
    attn = self.dropout(attn)

    out = torch.matmul(attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)

class Attention_block(nn.Module):
  def __init__(self, dim, heads = 1, dim_head = 64, dropout = 0.):
    super().__init__()
    self.norm_layer = nn.LayerNorm(dim)
    self.attention_layer = Attention(dim, heads, dim_head, dropout)
    self.forward_layer  = FeedForward(dim, int(dim/2), dropout)

  def forward(self,x):
    x = self.attention_layer(x) + x
    x = self.forward_layer(x) + x
    return self.norm_layer(x)




class Corss_Attention(nn.Module):
  def __init__(self, dim, heads = 1, dim_head = 64, dropout = 0.):
    super().__init__()
    assert heads==1
    inner_dim = dim_head *  heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)

    self.attend = nn.Softmax(dim = -1)
    self.dropout = nn.Dropout(dropout)

    #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
    self.to_q = nn.Linear(dim, inner_dim , bias = False)
    self.to_k = nn.Linear(dim, inner_dim , bias = False)
    self.to_v = nn.Linear(dim, inner_dim , bias = False)

    self.to_out = nn.Sequential(
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    ) if project_out else nn.Identity()

  def forward(self, x1, x2):
    x1 = self.norm1(x1)
    x2 = self.norm1(x2)


    q1= self.to_q(x1).unsqueeze(1)

    k2= self.to_k(x2).unsqueeze(1)

    v2= self.to_v(x2).unsqueeze(1)


    dots = torch.matmul(q1, k2.transpose(-1, -2)) * self.scale

    attn = self.attend(dots)
    attn = self.dropout(attn)

    out = torch.matmul(attn, v2)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)

class Corss_Attention_block(nn.Module):
  def __init__(self, dim, heads = 1, dim_head = 64, dropout = 0.):
    super().__init__()
    self.norm_layer = nn.LayerNorm(dim)
    self.attention_layer = Corss_Attention(dim, heads, dim_head, dropout)
    self.forward_layer  = FeedForward(dim, int(dim/2), dropout)

  def forward(self,x1,x2):
    x = self.attention_layer(x1, x2) + x1
    x = self.forward_layer(x) + x
    return self.norm_layer(x)


class Temporal_Weighted_Aggregation(nn.Module):
  """
  Temporal attention module
  """
  def __init__(self, hidden_dim):
    super(Temporal_Weighted_Aggregation, self).__init__()
    
    self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
    self.weighs_activation = nn.Tanh()
    self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)
    self.sm = torch.nn.Softmax(dim=1)
    self.gamma         = nn.Parameter(torch.tensor([0.]))

  def forward(self, x):

    # 输入是 batch  sensor_channel feature_dim
    #   B C F

    out = self.weighs_activation(self.fc_1(x))

    out = self.fc_2(out).squeeze(2)

    weights_att = self.sm(out).unsqueeze(2)

    context = torch.sum(weights_att * x, 1)
    context = x[:, -1, :] + self.gamma * context
    return context

class CrossAttn(nn.Module):
  def __init__(
      self,
      Ts_input_shape,
      hidden_dim,
      FFT_input_shape,

      Ts_layer_nr=4,
      Ts_kernel_size=5,
      Ts_padding_size=0,
      Ts_stride_size=2,


      FFT_layer_nr=4,
      FFT_kernel_size=3,
      FFT_padding_size=1,
      FFT_stride_size=1,

      activation="ReLU"):
    super(CrossAttn, self).__init__()
    self.Ts_input_shape = Ts_input_shape
    self.FFT_input_shape = FFT_input_shape
    self.hidden_dim = hidden_dim
    fusion_hidden_dim = 2*hidden_dim
    self.fusion_hidden_dim = fusion_hidden_dim
    self.ts_embedding_block = embedding_block(Ts_input_shape, hidden_dim, Ts_layer_nr, Ts_kernel_size, Ts_padding_size, Ts_stride_size)
    self.ts_self_attention1 = Attention_block(fusion_hidden_dim,1,fusion_hidden_dim)
    self.ts_cross_attention = Corss_Attention_block(fusion_hidden_dim,1,fusion_hidden_dim)
    self.ts_self_attention2 = Attention_block(fusion_hidden_dim,1,fusion_hidden_dim)
    self.ts_temporal_aggregation = Temporal_Weighted_Aggregation(fusion_hidden_dim)

    self.ft_embedding_block = embedding_block(FFT_input_shape, hidden_dim, FFT_layer_nr, FFT_kernel_size, FFT_padding_size, FFT_stride_size)
    self.ft_self_attention1 = Attention_block(fusion_hidden_dim,1,fusion_hidden_dim)
    self.ft_cross_attention = Corss_Attention_block(fusion_hidden_dim,1,fusion_hidden_dim)
    self.ft_self_attention2 = Attention_block(fusion_hidden_dim,1,fusion_hidden_dim)
    self.ft_temporal_aggregation = Temporal_Weighted_Aggregation(fusion_hidden_dim)

  def forward(self, x):
    x_ts = x[0]
    x_ft = x[1]
    x_ts = self.ts_embedding_block(x_ts)
    x_ts = self.ts_self_attention1(x_ts)


    x_ft = self.ft_embedding_block(x_ft)
    x_ft = self.ft_self_attention1(x_ft)

    x_ts_c = self.ts_cross_attention(x_ts,x_ft)
    x_ft_c = self.ft_cross_attention(x_ft,x_ts)

    x_ts_c = self.ts_self_attention2(x_ts_c)
    x_ft_c = self.ft_self_attention2(x_ft_c)

    x_ts_c = self.ts_temporal_aggregation(x_ts_c)
    x_ft_c = self.ft_temporal_aggregation(x_ft_c)

    return x_ts_c, x_ft_c









