# ---- import models ---------------
from models.Attend import AttendDiscriminate
from models.SA_HAR import SA_HAR
from models.deepconvlstm import DeepConvLSTM
from models.deepconvlstm_attn import DeepConvLSTM_ATTN
from models.crossatten.model import Cross_TS,TSTransformer_Basic
from models.TinyHAR import TinyHAR_Model
from models.SA_HAR import SA_HAR
from models.mcnn import MCNN


from models.ALAE import ALAE_TAE
from models.GlobalFusion import GlobalFusion
from models.CrossAttn import CrossAttn
from models.deepSense import DeepSense
from models.MixerMLP import FFTMIXER_HAR_Model
from models.visionmixer import Vision_MIXER

from I2S0W2C2_CFC.dataloaders.utils import PrepareWavelets,FiltersExtention
# ------- import other packages ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Wavelet_learnable_filter(nn.Module):
    def __init__(self, args, f_in):
        super(Wavelet_learnable_filter, self).__init__()
        self.args = args
        if args.windowsize%2==1:
            self.Filter_ReplicationPad2d = nn.ReflectionPad2d((int((args.windowsize-1)/2),int((args.windowsize-1)/2),0,0))
            raw_filter = np.zeros((1,1,1,args.windowsize))
            raw_filter[0,0,0,int((args.windowsize-1)/2)] = 1
        else:
            self.Filter_ReplicationPad2d = nn.ReflectionPad2d((int(args.windowsize/2),int(args.windowsize/2),0,0))
            raw_filter = np.zeros((1,1,1,args.windowsize))
            raw_filter[0,0,0,int(args.windowsize/2)] = 1
        raw_filter = torch.tensor(raw_filter)
        SelectedWavelet = PrepareWavelets(K=args.number_wavelet_filtering, length=args.windowsize, seed=self.args.seed)
        ScaledFilter = FiltersExtention(SelectedWavelet)
        ScaledFilter = torch.cat((raw_filter,ScaledFilter),0)

        #print("debug: ", ScaledFilter.shape[0], "   ", f_in)


        self.wavelet_conv = nn.Conv2d(1, f_in, 
                                      (1,ScaledFilter.shape[3]),
                                      stride=1, bias=False, padding='valid') 
        # TODO shut down
        if not args.wavelet_filtering_learnable and f_in==ScaledFilter.shape[0]:
            print("clone the  wavefiler weight")
            self.wavelet_conv.weight.data.copy_(ScaledFilter)                                        
            self.wavelet_conv.weight.requires_grad = False
        if self.args.wavelet_filtering_layernorm:
            print("wavelet layernorm")
            self.layer_norm = nn.LayerNorm(self.args.windowsize, elementwise_affine=False)
                                                               
    def forward(self,x):
        # input shape B 1 L  C  
        x = x.permute(0,1,3,2)
        x = self.Filter_ReplicationPad2d(x)
        x = self.wavelet_conv(x)[:,:,:,:self.args.windowsize]

        if self.args.wavelet_filtering_layernorm:
            x = self.layer_norm(x)

        x = x.permute(0,1,3,2)
        return x

class model_builder(nn.Module):
    """
    
    """
    def __init__(self, args, input_f_channel = None):
        super(model_builder, self).__init__()

        self.args = args
        if input_f_channel is None:
            f_in  = self.args.f_in
        else:
            f_in  = input_f_channel




        if self.args.wavelet_filtering:
            self.wave_conv = Wavelet_learnable_filter(args, f_in)


            if self.args.wavelet_filtering_regularization:
                print("Wavelet Filtering Regularization")
                shape      = (1, f_in, 1, 1)
                p          = torch.zeros(shape)
                p[0,0,0,0] = 1
                self.register_parameter('gamma' , nn.Parameter(p))
                # self.gamma = nn.Parameter(torch.ones(shape))
                #self.register_buff
		




        if self.args.model_type == "tinyhar":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["tinyhar"]
            #print(config["filter_num"])
            self.model  = TinyHAR_Model((1,f_in, self.args.input_length, self.args.c_in ), 
                                         self.args.num_classes,
                                         filter_num = config["filter_num"],#config["filter_num"],
                                         cross_channel_interaction_type = self.args.cross_channel_interaction_type,    # attn  transformer  identity
                                         cross_channel_aggregation_type = self.args.cross_channel_aggregation_type,  # filter  naive  FC
                                         temporal_info_interaction_type = self.args.temporal_info_interaction_type,     # gru  lstm  attn  transformer  identity
                                         temporal_info_aggregation_type = self.args.temporal_info_aggregation_type)    # naive  filter  FC )
            print("Build the TinyHAR model!")

        elif self.args.model_type == "attend":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["attend"]
            self.model  = AttendDiscriminate((1,f_in, self.args.input_length, self.args.c_in ), 
                                             self.args.num_classes,
                                             self.args.filter_scaling_factor,
                                             config)
            print("Build the AttendDiscriminate model!")
        elif self.args.model_type == "mcnn":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["mcnn"]
            self.model  = MCNN((1,f_in, self.args.input_length, self.args.c_in ), 
                                self.args.num_classes,
                                self.args.filter_scaling_factor,
                                config)
            print("Build the Milti-Branch CNN model!")
        elif self.args.model_type == "deepconvlstm_attn":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["deepconvlstm_attn"]
            self.model  = DeepConvLSTM_ATTN((1,f_in, self.args.input_length, self.args.c_in ), 
                                            self.args.num_classes,
                                            self.args.filter_scaling_factor,
                                            config)
            print("Build the deepconvlstm_attn model!")

        elif self.args.model_type == "sahar":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["sahar"]
            self.model  = SA_HAR((1,f_in, self.args.input_length, self.args.c_in ), 
                                            self.args.num_classes,
                                            self.args.filter_scaling_factor,
                                            config)
            print("Build the sahar model!")

        elif self.args.model_type == "deepconvlstm":
            config_file = open('configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["deepconvlstm"]
            self.model  = DeepConvLSTM((1,f_in, self.args.input_length, self.args.c_in ), 
                                       self.args.num_classes,
                                       self.args.filter_scaling_factor,
                                       config)
            print("Build the DeepConvLSTM model!")
        elif self.args.model_type == "mixermlp":
            #config_file = open('configs/model.yaml', mode='r')
            #config = yaml.load(config_file, Loader=yaml.FullLoader)["mixermlp"]

            fft_mixer_share_flag = False
            if self.args.mixer_representation == "both":
                fft_mixer_temporal_flag  = True
                fft_mixer_FFT_flag  = True
                print("mixer use both")
            elif self.args.mixer_representation == "fft":
                fft_mixer_temporal_flag  = False
                fft_mixer_FFT_flag = True 
                print("mixer only use fft")
            elif self.args.mixer_representation == "temporal":
                fft_mixer_temporal_flag  = True
                fft_mixer_FFT_flag  = False
                print("mixer only use temporal")
            else:
                assert 1==0

            if self.args.mixer_merge =="temporal":
                temporal_merge = True
            elif self.args.mixer_merge == "channel":
                temporal_merge = False
            else:
                assert 1==0
            self.model  = FFTMIXER_HAR_Model(input_shape=(1,1, self.args.input_length, self.args.c_in ), 
                                             number_class = self.args.num_classes,
                                             filter_num = self.args.mixer_mlp_feature_mum,
                                             fft_mixer_segments_length = self.args.fft_segments_length,
                                             expansion_factor = self.args.mixer_mlp_expansion_factor,
                                             fft_mixer_layer_nr = self.args.fft_mixer_layer_nr,
                                             #self.args.filter_scaling_factor,
                                             fuse_early = self.args.mixer_fuse_early,
                                             temporal_merge= temporal_merge,
                                             oration = self.args.mixer_oration,
                                             fft_mixer_share_flag=fft_mixer_share_flag,
                                             fft_mixer_FFT_flag=fft_mixer_FFT_flag,
                                             fft_mixer_temporal_flag=fft_mixer_temporal_flag,
                                             residual_flag=self.args.residual_flag
                                            )

            print("Build the mixermlp model!")
        elif self.args.model_type == "deepsense":

            self.model  = DeepSense(input_shape=(1,1, self.args.input_length, self.args.c_in ), 
                                    k_number_sensors_group= self.args.k_number_sensors_group, 
                                    fft_segments_length = self.args.fft_segments_length,
                                    nb_classes= self.args.num_classes)

            print("Build the deepsense model!")
        elif self.args.model_type == "aele":

            self.model  = ALAE_TAE(
                input_shape=(1,1, self.args.input_length, self.args.c_in ), 
                nb_classes= self.args.num_classes)

            print("Build the ALAE_TAE model!")
        elif self.args.model_type == "globalfusion":

            self.model  = GlobalFusion(
                input_shape=(1,1, self.args.input_length, self.args.c_in ), 
                S_number_sensors_type = self.args.S_number_sensors_type,
                L_sensor_locations= self.args.L_sensor_locations,
                fft_segments_length = self.args.fft_segments_length,
                nb_classes= self.args.num_classes)

            print("Build the GlobalFusion model!")
        elif self.args.model_type == "visionmixer":
            self.model  = Vision_MIXER(
                input_shape=(1,1, self.args.input_length, self.args.c_in ), 
                patch_size = 3,
                number_class= self.args.num_classes) 

            print("Build the vision mixer model!!")
        else:
            self.model = Identity()
            print("Build the None model!")


    def forward(self,x):
        #if self.first_conv ：
        #    x = self.pre_conv(x)

        if self.args.wavelet_filtering:
            x = self.wave_conv(x)
            if self.args.wavelet_filtering_regularization:
                x = x * self.gamma
        y = self.model(x)
        return y









"""
from models.crossatten.model import Cross_TS,TSTransformer_Basic
from models.deepconvlstm import DeepConvLSTM
from models.SA_HAR import SA_HAR
from models.deepconvlstm_attn import DeepConvLSTM_ATTN
from models.Attend import AttendDiscriminate
from models.Attend_new import AttendDiscriminate_new
from models.CNN_freq import CNN_Freq_Model
from models.CNN_LSTM_FREQ import CNN_LSTM_FREQ_Model
from models.CNN_LSTM_TIME import CNN_LSTM_TIME_Model
from models.CNN_LSTM_TIME_FREQ import CNN_LSTM_CROSS_Model
from models.CFC import CFC_Model
from models.CFC_V1 import CFC_V1_Model
from models.CFC_V2 import CFC_V2_Model
from models.CFC_V3 import CFC_V3_Model
from models.CFC_V4 import CFC_V4_Model
from models.TinyHAR import TinyHAR_Model
        if self.args.model_type in ["time","freq","cross"]:
            model  = Cross_TS(self.args)
            print("Build the conv_TS model!")
        elif self.args.model_type == "basic":
            model  = TSTransformer_Basic(self.args)
            print("Build the basic TS model!")
        elif self.args.model_type == "deepconvlstm":
            model  = DeepConvLSTM(self.args.c_in, self.args.num_classes)
            print("Build the DeepConvLSTM model!")
        elif self.args.model_type == "sahar":
            model  = SA_HAR(self.args.c_in, self.args.input_length, self.args.num_classes)
            print("Build the SA_HAR model!")
        elif self.args.model_type == "deepconvlstm_attn":
            model  = DeepConvLSTM_ATTN(self.args.c_in, self.args.num_classes)
            print("Build the deepconvlstm_attn model!")
        elif self.args.model_type == "attend":
            model  = AttendDiscriminate(self.args.c_in, self.args.num_classes)
            print("Build the AttendDiscriminate model!")
        elif self.args.model_type == "attend_new":
            model  = AttendDiscriminate_new(self.args.c_in, self.args.num_classes)
            print("Build the AttendDiscriminate_new model!")
        elif self.args.model_type == "cnn_freq":
            model  = CNN_Freq_Model((1,self.args.c_in, self.args.sampling_freq, self.args.input_length ), self.args.num_classes)
            print("Build the CNN_Freq_Model model!")		
        elif self.args.model_type == "cnn_lstm_freq":
            model  = CNN_LSTM_FREQ_Model((self.args.input_length,self.args.sampling_freq ), self.args.num_classes)
            print("Build the CNN_LSTM_FREQ_Model model!")					
        elif self.args.model_type == "cnn_lstm_time":
            model  = CNN_LSTM_TIME_Model((self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CNN_LSTM_TIME_Model model!")		
        elif self.args.model_type == "cnn_lstm_cross":
            model  = CNN_LSTM_CROSS_Model((self.args.input_length, self.args.c_in ),(self.args.input_length,self.args.sampling_freq ), self.args.num_classes)
            print("Build the CNN_LSTM_CROSS_Model model!")
        elif self.args.model_type == "cfc":
            model  = CFC_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC model!")
        elif self.args.model_type == "cfcv1":
            model  = CFC_V1_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC_V1_Model model!")
        elif self.args.model_type == "cfcv2":
            model  = CFC_V2_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC_V2_Model model!")		
        elif self.args.model_type == "cfcv3":
            model  = CFC_V3_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC_V3_Model model!")
        elif self.args.model_type == "cfcv4":
            model  = CFC_V4_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC_V4_Model model!")


        elif self.args.model_type == "tinyhar":
            model  = TinyHAR_Model((1,self.args.f_in, self.args.input_length, self.args.c_in ), 
                                   self.args.num_classes,
                                   cross_channel_interaction_type = self.args.cross_channel_interaction_type,    # attn  transformer  identity
                                   cross_channel_aggregation_type = self.args.cross_channel_aggregation_type,  # filter  naive  FC
                                   temporal_info_interaction_type = self.args.temporal_info_interaction_type,     # gru  lstm  attn  transformer  identity
                                   temporal_info_aggregation_type = self.args.temporal_info_aggregation_type)    # naive  filter  FC )
            print("Build the TinyHAR_Model model!")	
			
        else:
            raise NotImplementedError
"""