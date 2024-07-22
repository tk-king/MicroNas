import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from einops import rearrange, repeat

# merge temporal -relu  norm

def extract_blocks_with_overlap_torch(input_segs, T_size,ratio = 0.5):

  B, F, T, C = input_segs.size()
  overlap = T_size*ratio
  blocks = []
  step = T_size - overlap

  for b in range(B):  # 遍历每张图片
    seg_blocks = []
    for j in range(0, C, 1):
        i = 0
        while i<=T-T_size:
        #for i in range(0, T - T_size + 1, step):
            block = input_segs[b:b+1, :, int(i):int(i)+T_size, j:j+1]
            i = i + step


            seg_blocks.append(block)
        i = i - step
        if i<T-T_size:
            # Extract the tail part if it's smaller than the window size and flag is True
            i = T-T_size
            block = input_segs[b:b+1, :, int(i):int(i)+T_size, j:j+1]
            seg_blocks.append(block)
      
        
    blocks.append(torch.cat(seg_blocks ,dim=-1 ))


  return torch.cat(blocks)

class Fembbeding_block(nn.Module):
  def __init__(
      self,
      input_shape,
      segment_length,
      feature_dim,
      temporal_flag=True,
      FFT_flag=True,
      share_flag=False,
      activation = None,
      fuse_early = False,
      oration = 0.5
  ):
    super(Fembbeding_block, self).__init__()


    B,F,T,C = input_shape
    assert F == 1
    
    #assert input_shape[2]%segment_length == 0
    # ------------------change ----------------
    #self.nr_segment = int(2*input_shape[2]/segment_length-1)
    temp = torch.rand(input_shape)
    temp_split = extract_blocks_with_overlap_torch(temp,segment_length,oration)
    self.nr_segment = int( temp_split.shape[-1]/input_shape[-1])
    print(" self.nr_segment :",self.nr_segment )
    self.temporal_flag = temporal_flag
    self.FFT_flag = FFT_flag
    self.share_flag = share_flag
    self.fuse_early = fuse_early
    self.oration = oration


    self.segment_length = segment_length
    self.feature_dim = feature_dim


    if fuse_early:
      if temporal_flag:
        #self.T_dense = nn.Linear(segment_length ,feature_dim)
        # 如何初始化
        if share_flag:
          self.T_dense_weight = nn.Parameter(torch.randn(segment_length*3, feature_dim, dtype=torch.float32))
          self.T_dense_bias  = nn.Parameter(torch.randn(feature_dim, dtype=torch.float32))
        else:
          self.T_dense_weight = nn.Parameter(torch.randn(int(C/3), segment_length*3, feature_dim, dtype=torch.float32))
          self.T_dense_bias  = nn.Parameter(torch.randn(int(C/3), feature_dim, dtype=torch.float32))

      else:
        self.T_dense_weight = None
        self.T_dense_bias = None
    else:
      if temporal_flag:
        #self.T_dense = nn.Linear(segment_length ,feature_dim)
        # 如何初始化
        if share_flag:
          self.T_dense_weight = nn.Parameter(torch.randn(segment_length, feature_dim, dtype=torch.float32))
          self.T_dense_bias  = nn.Parameter(torch.randn(feature_dim, dtype=torch.float32))
        else:
          self.T_dense_weight = nn.Parameter(torch.randn(C, segment_length, feature_dim, dtype=torch.float32))
          self.T_dense_bias  = nn.Parameter(torch.randn(C, feature_dim, dtype=torch.float32))

      else:
        self.T_dense_weight = None
        self.T_dense_bias = None



    if fuse_early:
      if FFT_flag:
        #self.FFT_dense = nn.Linear(2*segment_length ,feature_dim)

        if share_flag:
          self.F_dense_weight = nn.Parameter(torch.randn(6*segment_length, feature_dim, dtype=torch.float32))
          self.F_dense_bias  = nn.Parameter(torch.randn( feature_dim, dtype=torch.float32))
        else:
          self.F_dense_weight = nn.Parameter(torch.randn(int(C/3), 6*segment_length, feature_dim, dtype=torch.float32))
          self.F_dense_bias  = nn.Parameter(torch.randn(int(C/3), feature_dim, dtype=torch.float32))
      else:
        self.F_dense_weight = None
        self.F_dense_bias  = None

    else:

      if FFT_flag:
        #self.FFT_dense = nn.Linear(2*segment_length ,feature_dim)

        if share_flag:
          self.F_dense_weight = nn.Parameter(torch.randn(2*segment_length, feature_dim, dtype=torch.float32))
          self.F_dense_bias  = nn.Parameter(torch.randn( feature_dim, dtype=torch.float32))
        else:
          self.F_dense_weight = nn.Parameter(torch.randn(C, 2*segment_length, feature_dim, dtype=torch.float32))
          self.F_dense_bias  = nn.Parameter(torch.randn(C, feature_dim, dtype=torch.float32))
      else:
        self.F_dense_weight = None
        self.F_dense_bias  = None

    self.activation = nn.ReLU()
    self.norm_termporal = nn.LayerNorm(feature_dim*self.nr_segment)
    self.norm_FFT = nn.LayerNorm(feature_dim*self.nr_segment)
    if FFT_flag and temporal_flag:
        self.fusion_layer = nn.Linear(feature_dim*2,2*feature_dim)
    else:
        self.fusion_layer = nn.Linear(feature_dim,2*feature_dim)
    print("-----------use fusion -----" )
  def forward(self, x):
    # B F T C
    B, F, T, C  = x.shape
    assert F==1



    x_split = extract_blocks_with_overlap_torch(x, self.segment_length,self.oration)
    # B F T-->segment_length, C-->C*nr_segment




    if self.fuse_early:

      x_split_temp = x_split.permute(0,1,3,2)
      x_split_temp = x_split_temp.reshape(B,C,self.nr_segment, self.segment_length)
      x_1_temp =  x_split_temp[:,::3,:,:]
      x_2_temp =  x_split_temp[:,1::3,:,:]
      x_3_temp =  x_split_temp[:,2::3,:,:]
      x_split_temp  = torch.cat([x_1_temp,x_2_temp,x_3_temp],dim=-1)

      # B C/3 nr_segment  3*segment_length
      x_split_temp = x_split_temp.reshape(B,1,int(C/3)*self.nr_segment,self.segment_length*3)
      x_split_temp = x_split_temp.permute(0,1,3,2)


      if self.temporal_flag:
        if self.share_flag:
          x_t = torch.einsum('bict,tf->bicf', x_split_temp.permute(0,1,3,2), self.T_dense_weight)+self.T_dense_bias
          # B  F,   C-->C*nr_segment, T-->segment_length-->feature_dim,
          x_t = self.activation(self.norm_termporal(x_t.reshape(B,F,int(C/3),-1)).reshape(B,F,int(C/3)*self.nr_segment,-1))
        else:

          weight_all = self.T_dense_weight.unsqueeze(1).repeat(1, self.nr_segment, 1, 1).view(-1, self.segment_length*3,self.feature_dim)
          bias_all = self.T_dense_bias.unsqueeze(1).repeat(1, self.nr_segment,1).view(-1, int(C/3)*self.nr_segment,self.feature_dim)

          x_t = torch.einsum('bict,ctf->bicf', x_split_temp.permute(0,1,3,2), weight_all)+bias_all
          x_t = self.activation(self.norm_termporal(x_t.reshape(B,F,int(C/3),-1)).reshape(B,F,int(C/3)*self.nr_segment,-1))
      else:
        x_t = None
    else:

      if self.temporal_flag:
        if self.share_flag:
          x_t = torch.einsum('bict,tf->bicf', x_split.permute(0,1,3,2), self.T_dense_weight)+self.T_dense_bias
          # B  F,   C-->C*nr_segment, T-->segment_length-->feature_dim,
          x_t = self.activation(self.norm_termporal(x_t.reshape(B,F,C,-1)).reshape(B,F,C*self.nr_segment,-1))
        else:

          weight_all = self.T_dense_weight.unsqueeze(1).repeat(1, self.nr_segment, 1, 1).view(-1, self.segment_length,self.feature_dim)
          bias_all = self.T_dense_bias.unsqueeze(1).repeat(1, self.nr_segment,1).view(-1, C*self.nr_segment,self.feature_dim)


          x_t = torch.einsum('bict,ctf->bicf', x_split.permute(0,1,3,2), weight_all)+bias_all
          x_t = self.activation(self.norm_termporal(x_t.reshape(B,F,C,-1)).reshape(B,F,C*self.nr_segment,-1))
      else:
        x_t = None

    if self.fuse_early:
      if self.FFT_flag:

        x_f = torch.cat([torch.fft.fft(x_split.permute(0,1,3,2), dim=-1).real, torch.fft.fft(x_split.permute(0,1,3,2), dim=-1).imag],dim=-1)

        x_f = x_f.reshape(B,C,self.nr_segment, self.segment_length*2)
        x_f_1_temp =  x_f[:,::3,:,:]
        x_f_2_temp =  x_f[:,1::3,:,:]
        x_f_3_temp =  x_f[:,2::3,:,:]
        x_f  = torch.cat([x_f_1_temp,x_f_2_temp,x_f_3_temp],dim=-1)
        # B C/3 nr_segment  3*segment_length
        x_f = x_f.reshape(B,1,int(C/3)*self.nr_segment,self.segment_length*3*2)


        if self.share_flag:
          x_f = torch.einsum('bict,tf->bicf', x_f, self.F_dense_weight)+self.F_dense_bias
          x_f = self.activation(self.norm_FFT(x_f.reshape(B,F,int(C/3),-1)).reshape(B,F,int(C/3)*self.nr_segment,-1))
        else:
          weight_all = self.F_dense_weight.unsqueeze(1).repeat(1, self.nr_segment, 1, 1).view(-1, 3*2*self.segment_length,self.feature_dim)
          bias_all = self.F_dense_bias.unsqueeze(1).repeat(1, self.nr_segment,1).view(-1, int(C/3)*self.nr_segment,self.feature_dim)
          x_f = torch.einsum('bict,ctf->bicf', x_f, weight_all)+bias_all
          x_f = self.activation(self.norm_FFT(x_f.reshape(B,F,int(C/3),-1)).reshape(B,F,int(C/3)*self.nr_segment,-1))
      else:
        x_f = None
    else:
      if self.FFT_flag:
        x_f = torch.cat([torch.fft.fft(x_split.permute(0,1,3,2), dim=-1).real, torch.fft.fft(x_split.permute(0,1,3,2), dim=-1).imag],dim=-1)

        if self.share_flag:
          x_f = torch.einsum('bict,tf->bicf', x_f, self.F_dense_weight)+self.F_dense_bias
          x_f = self.activation(self.norm_FFT(x_f.reshape(B,F,C,-1)).reshape(B,F,C*self.nr_segment,-1))
        else:
          weight_all = self.F_dense_weight.unsqueeze(1).repeat(1, self.nr_segment, 1, 1).view(-1, 2*self.segment_length,self.feature_dim)
          bias_all = self.F_dense_bias.unsqueeze(1).repeat(1, self.nr_segment,1).view(-1, C*self.nr_segment,self.feature_dim)
          x_f = torch.einsum('bict,ctf->bicf', x_f, weight_all)+bias_all
          x_f = self.activation(self.norm_FFT(x_f.reshape(B,F,C,-1)).reshape(B,F,C*self.nr_segment,-1))
      else:
        x_f = None

    if self.temporal_flag and self.FFT_flag:
      x = torch.cat([x_t,x_f ],dim=-1)
    elif self.temporal_flag and not self.FFT_flag:
      x = x_t
    elif not self.temporal_flag and self.FFT_flag:
      x = x_f
    else:
      assert True==False
    if self.fuse_early:
      x = x.reshape(B,int(C/3),self.nr_segment,-1)
    else:
      x = x.reshape(B,C,self.nr_segment,-1)
    x = self.activation(self.fusion_layer(x))
    x = x.permute(0,3,2,1)

    return x


class MLP(nn.Module):
  def __init__(self, num_features, expansion_factor=1, dropout=0):
    super().__init__()
    num_hidden = int(num_features * expansion_factor)

    self.fc1 = nn.Linear(num_features, num_hidden)

    self.dropout1 = nn.Dropout(dropout)

    self.fc2 = nn.Linear(num_hidden, num_features)

    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x):
      x = self.dropout1(F.relu(self.fc1(x)))
      x = self.dropout2(self.fc2(x))

      return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, segments_nr, sensor_channel, expansion_factor, dropout):
        super().__init__()
        self.num_features = num_features
        self.segments_nr  = segments_nr
        self.sensor_channel = sensor_channel
        self.norm1 = nn.LayerNorm(num_features*segments_nr)
        self.mlp = MLP(num_features*segments_nr, expansion_factor, dropout)
        self.norm2 = nn.LayerNorm(num_features*segments_nr)

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.permute(0,3,2,1).reshape(batch_size,self.sensor_channel,-1)
        residual = x


        x = self.norm1(x)

        x = self.mlp(x)



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

        x = x.permute(0,2,1,3).reshape(batch_size,self.segments_nr,-1)

        x = self.mlp(x)

        x = x.reshape(batch_size,self.segments_nr,self.num_features,self.sensor_channel).permute(0,2,1,3)


        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, segments_nr, sensor_channel, expansion_factor, dropout):
        super().__init__()

        self.num_features = num_features
        self.segments_nr  = segments_nr
        self.sensor_channel = sensor_channel

        self.token_mixer = TokenMixer(
            num_features, segments_nr, sensor_channel, expansion_factor, dropout
        )
        self.channel_mixer = ChannelMixer(
            num_features, segments_nr, sensor_channel, expansion_factor, dropout
        )

    def forward(self, x):

        x = self.token_mixer(x)
        x = self.channel_mixer(x)

        return x


class FFTMIXER_HAR_Model(nn.Module):
  def __init__(
      self,
      input_shape,
      number_class,
      filter_num,
      fft_mixer_segments_length,
      expansion_factor,
      fft_mixer_layer_nr,
      fuse_early,
      temporal_merge,
      oration,
      model_config
      # filter_num,
      # fft_mixer_segments_length,
      # fft_mixer_share_flag,
      # fft_mixer_temporal_flag,
      # fft_mixer_FFT_flag,
      # fft_mixer_layer_nr,
      # expansion_factor
  ):
    super(FFTMIXER_HAR_Model,self).__init__()


    batch,_,window_length,sensor_channel_nr = input_shape
    if fuse_early:
      self.sensor_channel_nr = int(sensor_channel_nr/3)
    else:
      self.sensor_channel_nr = sensor_channel_nr

    #fft_mixer_segments_length = model_config["fft_mixer_segments_length"]
    #filter_num                = model_config["filter_num"]
    fft_mixer_share_flag      = model_config["fft_mixer_share_flag"]
    fft_mixer_temporal_flag      = model_config["fft_mixer_temporal_flag"]
    fft_mixer_FFT_flag      = model_config["fft_mixer_FFT_flag"]
    #fft_mixer_layer_nr      = model_config["fft_mixer_layer_nr"]
    #expansion_factor        = model_config["expansion_factor"]
    self.fuse_early = fuse_early
    temp = torch.rand(input_shape)
    temp_split = extract_blocks_with_overlap_torch(temp,fft_mixer_segments_length,oration)
    self.oration = oration
    self.nr_segment = int( temp_split.shape[-1]/input_shape[-1])
    #self.nr_segment = int(2*window_length/fft_mixer_segments_length-1)
    self.number_class            = number_class
    self.filter_num             = filter_num
    self.fft_mixer_segments_length     = fft_mixer_segments_length
    self.fft_mixer_share_flag        = fft_mixer_share_flag
    self.fft_mixer_temporal_flag       = fft_mixer_temporal_flag
    self.fft_mixer_FFT_flag         = fft_mixer_FFT_flag
    self.fft_mixer_layer_nr         = fft_mixer_layer_nr
    self.expansion_factor    = expansion_factor
    self.temporal_merge = temporal_merge

    self.fft_embedding_block = Fembbeding_block(
        input_shape    =input_shape,
        segment_length  =fft_mixer_segments_length,
        temporal_flag   =fft_mixer_temporal_flag,
        FFT_flag      =fft_mixer_FFT_flag,
        share_flag    =fft_mixer_share_flag,
        feature_dim   =filter_num,
        fuse_early = fuse_early,
        oration = oration)

    self.mixer_layer = nn.Sequential(
            *[
                MixerLayer(filter_num*2, self.nr_segment, self.sensor_channel_nr, expansion_factor=expansion_factor, dropout=0)
                for _ in range(fft_mixer_layer_nr)
            ]
        )
    if self.temporal_merge:
        self.merge = nn.Linear(filter_num*2*self.nr_segment, filter_num*2)
        self.norm = nn.LayerNorm(filter_num*2)
        self.predict = nn.Linear(filter_num*2*self.sensor_channel_nr, number_class)
    else:
        self.merge = nn.Linear(filter_num*2*self.sensor_channel_nr, filter_num*2)
        self.norm = nn.LayerNorm(filter_num*2)
        self.predict = nn.Linear(filter_num*2*self.nr_segment, number_class)

  def forward(self, x):
    batch=x.shape[0]
    x = self.fft_embedding_block(x)


    x = self.mixer_layer(x)
    # B F L C
    if self.temporal_merge:
        x = x.permute(0,3,2,1).reshape(batch, self.sensor_channel_nr, -1)
        x = F.relu(self.norm(self.merge(x)))
        x = x.reshape(batch,-1)
        y = self.predict(x)
    else:
        x = x.permute(0,2,1,3).reshape(batch, self.nr_segment, -1)
        x = F.relu(self.norm(self.merge(x)))
        x = x.reshape(batch,-1)
        y = self.predict(x)
    return y
