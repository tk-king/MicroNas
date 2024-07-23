from othermodels.ALAE import ALAE_TAE
from othermodels.visionmixer import Vision_MIXER
from othermodels.mcnn import MCNN
from othermodels.deepconvlstm import DeepConvLSTM
from othermodels.deepconvlstm_attn import DeepConvLSTM_ATTN
from othermodels.deepSense import DeepSense
from othermodels.TinyHAR import TinyHAR_Model
from othermodels.MixerMLP import FFTMIXER_HAR_Model


config_mcnn = {
    "nb_conv_blocks": 2,
    "nb_filters": 64,
    "dilation": 1,
    "batch_norm": 0,
    "filter_width": 5,
    "drop_prob": 0.25}

config_deepconvlstm = {
    "nb_conv_blocks": 2,
    "nb_filters": 64,
    "dilation": 1,
    "batch_norm": 0,
    "filter_width": 11,
    "nb_layers_lstm": 1,
    "drop_prob": 0.2,
    "nb_units_lstm": 128}

config_deepconvlstmattn = {
    "nb_conv_blocks": 2,
    "nb_filters": 64,
    "dilation": 1,
    "batch_norm": 0,
    "filter_width": 5,
    "nb_layers_lstm": 2,
    "drop_prob": 0.5,
    "nb_units_lstm": 128}


def get_model_by_name(name, ts_len, num_sensors, num_classes):

    model_tinyhar = TinyHAR_Model((1, 1, ts_len, num_sensors),
                                  num_classes,
                                  filter_num=32,
                                  cross_channel_interaction_type="attn",    # attn  transformer  identity
                                  cross_channel_aggregation_type="FC",  # filter  naive  FC
                                  # gru  lstm  attn  transformer  identity
                                  temporal_info_interaction_type="lstm",
                                  temporal_info_aggregation_type="tnaive")    # naive  filter  FC )

    config = {}
    config["fft_mixer_share_flag"] = False
    config["fft_mixer_temporal_flag"] = True
    config["fft_mixer_FFT_flag"] = True
    fft_feature_dim = 32
    model_fft_mixer = FFTMIXER_HAR_Model(input_shape=(1, 1, ts_len, num_sensors),
                                         number_class=num_classes,
                                         filter_num=6,
                                         fft_mixer_segments_length=int(
        fft_feature_dim/2),
        expansion_factor=0.3,
        fft_mixer_layer_nr=2,
        fuse_early=False,
        temporal_merge=True,
        oration=0.25,
        model_config=config)

    model_dict = {
        "alae_tae": ALAE_TAE((1, 1, ts_len, num_sensors), num_classes),
        "vision_mixer": Vision_MIXER((1, 1, ts_len, num_sensors), patch_size=3, number_class=num_classes),
        "mcnn": MCNN((1, 1, ts_len, num_sensors), num_classes, 1, config_mcnn),
        "deepconvlstm": DeepConvLSTM((1, 1, ts_len, num_sensors), num_classes, 1, config_deepconvlstm),
        "deepconvlstmattn": DeepConvLSTM_ATTN((1, 1, ts_len, num_sensors), num_classes, 1, config_deepconvlstmattn),
        "tinyhar": model_tinyhar,
        "fftmixer": model_fft_mixer
    }
    return model_dict[name]
