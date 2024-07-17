from I2S0W2C2_CFC.dataloaders import data_dict, data_set
import argparse
import yaml
import os
from tqdm import tqdm
from InMemoryDataset import InMemoryDataset, DiskData

ROOT_PATH = "./datasets"


parser = argparse.ArgumentParser(description='Generate dataset')

parser.add_argument('--to-save-path', dest='to_save_path', default= "../Run_logs", type=str, help='Set the path to save logs')
parser.add_argument('--freq-save-path', dest='freq_save_path', default= "../Freq_data", type=str, help='Set the path to save freq images transformation')
parser.add_argument('--window-save-path', dest='window_save_path', default= "./Sliding_window", type=str, help='Set the path to save slided window file')
parser.add_argument('--root-path', dest='root_path', default= "../../../../datasets", type=str, help='Set the path to data dir')

# data normalization
parser.add_argument('--datanorm-type', dest='datanorm_type', default= "standardization", choices=[None, "minmax", "standardization"],
                    type=str, help='Set the mathod for standize the data')
parser.add_argument('--sample-wise', dest='sample_wise', action='store_true', help='weather to perform sample_wise normailization')


parser.add_argument('--drop-transition', dest='drop_transition', action='store_true', help='weather to drop the transition part')

# training config
parser.add_argument('--batch-size', dest='batch_size', default=512, type=int,  help='Batch Size')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='weather to shuffle the data')
parser.add_argument('--drop-last', dest='drop_last', action='store_true', help='weather to drop the last mini batch ')
parser.add_argument('--train-vali-quote', dest='train_vali_quote', type=float, default=0.9, help='Portion of training dataset')
parser.add_argument('--train-epochs', dest='train_epochs', default=150, type=int,  help='Total Training Epochs')
parser.add_argument('--learning-rate', dest='learning_rate', default=0.001, type=float,  help='set the initial learning rate')
parser.add_argument('--learning-rate-patience', dest='learning_rate_patience', default=7, type=int,  help='patience for adjust the learning rate')
parser.add_argument('--early-stop-patience', dest='early_stop_patience', default=15, type=int,  help='patience for stop the training')
parser.add_argument('--learning-rate-factor', dest='learning_rate_factor', default=0.1, type=float,  help='set the rate of adjusting learning rate')
#parser.add_argument('--use-gpu', dest='use_gpu', action='store_true', help='weather to use gpu ')
parser.add_argument('--use-multi-gpu', dest='use_multi_gpu', action='store_true', help='weather to use multi gpu ')
parser.add_argument('--gpu', dest='gpu', default=0, type=int,  help='gpu id')
parser.add_argument('--optimizer', dest='optimizer', default= "Adam", type=str, help='Set the Optimized Type')
parser.add_argument('--criterion', dest='criterion', default= "CrossEntropy", type=str, help='Set the Loss Type')
parser.add_argument('--seed', dest='seed', default=1, type=int,  help='training seed')

parser.add_argument('--data-name', dest='data_name', default= None, type=str, help='Set the dataset name')

parser.add_argument('--wavelet-filtering', dest='wavelet_filtering', action='store_true', help='weather to use wavelet filtering to prepare the date ')
parser.add_argument('--wavelet-filtering-regularization', dest='wavelet_filtering_regularization', action='store_true', help='weather to use wavelet filtering regularization')
parser.add_argument('--wavelet-filtering-finetuning', dest='wavelet_filtering_finetuning', action='store_true', help='weather to use wavelet filtering fine tuning')
parser.add_argument('--wavelet-filtering-finetuning-percent', dest='wavelet_filtering_finetuning_percent', default=0.5, type=float,  help='set the filtering fine tuning percentage')
parser.add_argument('--wavelet-filtering-learnable', dest='wavelet_filtering_learnable', action='store_true', help='weather to use learnable wavelet filtering')
parser.add_argument('--wavelet-filtering-layernorm', dest='wavelet_filtering_layernorm', action='store_true', help='weather to use layernorm for wavelet filtering')
parser.add_argument('--regulatization-tradeoff', dest='regulatization_tradeoff', default=0.0, type=float,  help='set the regularuization trade off parameter')
parser.add_argument('--number-wavelet-filtering', dest='number_wavelet_filtering', default=6, type=int,  help='pruning rest number of wavelet')


parser.add_argument('--difference', dest='difference', action='store_true', help='weather to use difference ')
parser.add_argument('--filtering', dest='filtering', action='store_true', help='weather to use filtering ')
parser.add_argument('--magnitude', dest='magnitude', action='store_true', help='weather to use magnitude ')
parser.add_argument('--weighted-sampler', dest='weighted_sampler', action='store_true', help='weather to use weighted_sampler ')
parser.add_argument('--load-all', dest='load_all', action='store_true', help='weather to load all freq data ')
parser.add_argument('--wavelet-function', dest='wavelet_function', default= None, type=str, help='Method to generate spectrogram')
parser.add_argument('--mixup-probability', dest='mixup_probability', default=0.5, type=float,  help='set the prob to use mixup')
parser.add_argument('--mixup-alpha', dest='mixup_alpha', default=0.5, type=float,  help='set the mixup distribution')
parser.add_argument('--mixup-argmax', dest='mixup_argmax', action='store_true', help='weather to use argmax ')
parser.add_argument('--random-augmentation-prob', dest='random_augmentation_prob', default=0.5, type=float,  help='set the prob to use random prob')
parser.add_argument('--random-augmentation-config-path', dest='random_augmentation_config_path', default="random_aug.yaml", type=str,  help='set the path to random aug config')
parser.add_argument('--max-aug', dest='max_aug', default=3, type=int,  help='max number of random aug')


parser.add_argument('--representation-type', dest='representation_type', default= "time", type=str, help='Set the data type')
parser.add_argument('--exp-mode', dest='exp_mode', default= "LOCV", type=str, help='Set the exp type')

parser.add_argument('--model-type', dest='model_type', default= None, type=str, help='Set the model type')
parser.add_argument('--output-attention', dest='output_attention', action='store_true', help='weather to print out the attention map if cross attention model is used')
parser.add_argument('--filter-scaling-factor', dest='filter_scaling_factor', default=1.0, type=float,  help='set the scaling factor for filter dimension')


parser.add_argument('--cross-channel-interaction-type', dest='cross_channel_interaction_type', default= "attn", type=str, help='Set the cross_channel_interaction_type type')
parser.add_argument('--cross-channel-aggregation-type', dest='cross_channel_aggregation_type', default= "FC", type=str, help='Set the cross_channel_aggregation_type type')
parser.add_argument('--temporal-info-interaction-type', dest='temporal_info_interaction_type', default= "lstm", type=str, help='Set the temporal_info_interaction_type type')
parser.add_argument('--temporal-info-aggregation-type', dest='temporal_info_aggregation_type', default= "tnaive", type=str, help='Set the temporal_info_aggregation_type type')



parser.add_argument('--mixer-mlp-feature-mum', dest='mixer_mlp_feature_mum', default= 4, type=int, help='Set the feature dimension for the mixermpl model')
parser.add_argument('--fft-mixer-layer-nr', dest='fft_mixer_layer_nr', default= 2, type=int, help='Set the number of layers')
parser.add_argument('--mixer-mlp-expansion-factor', dest='mixer_mlp_expansion_factor', default= 0.2, type=float, help='Set the feature dimension for the mixermpl model')
parser.add_argument('--mixer-oration', dest='mixer_oration', default= 0.5, type=float, help='Set the feature dimension for the mixermpl model')
parser.add_argument('--fft-segments-length', dest='fft_segments_length', default= 16, type=int, help='set the feature length of the segment, it should divided by the sliding window size, it depends on the dataset')
parser.add_argument('--mixer-fuse-early', dest='mixer_fuse_early', action='store_true', help='weather to use eary fusion in the mixer model')
parser.add_argument('--mixer-representation', dest='mixer_representation', default= "both", type=str, help='weather to use eary fusion in the mixer model')
parser.add_argument('--mixer-merge', dest='mixer_merge', default= "none", type=str, help='the order of merge info')
parser.add_argument('--residual-flag', dest='residual_flag', action='store_true', help='weather to use skip connection in the mixer model')

parser.add_argument("--sliding-window-factor", dest="sliding_window_factor", default=0.5)

# parser.add_argument("--data_name", type=str, help="Name of the dataset")
# parser.add_argument('--exp-mode', dest='exp_mode',
#                     default="LOCV", type=str, help='Set the exp type')
# parser.add_argument('--representation-type', dest='representation_type',
#                     default="time", type=str, help='Set the data type')

args = parser.parse_args()
if args.data_name == "skodar":
    args.exp_mode = "SOCV"

config_files = open("I2S0W2C2_CFC/configs/data.yaml", "r")
data_config = yaml.load(config_files, Loader=yaml.FullLoader)
config = data_config[args.data_name]

args.root_path = os.path.join(ROOT_PATH, config["filename"])
args.sampling_freq = config["sampling_freq"]
# print("-------------------delete num_classes-----------------")
args.num_classes = config["num_classes"]
window_seconds = config["window_seconds"]
args.windowsize = int(window_seconds * args.sampling_freq)
args.input_length = args.windowsize

args.S_number_sensors_type = config["S_number_sensors_type"]
args.L_sensor_locations = config["L_sensor_locations"]
args.k_number_sensors_group = config["k_number_sensors_group"]





args.c_in = config["num_channels"]

# Custom config
# args.load_all = False
args.pos_select       = None
args.sensor_select    = None

dataset = data_dict[args.data_name](args)

num_of_cv = dataset.num_of_cv


def get_data(args, data, flag="train", weighted_sampler = False):
    if flag == 'train':
        shuffle_flag = True 
    else:
        shuffle_flag = False
    data  = data_set(args,data,flag)
    return data

def extract_data(data):
    time_x = []
    freq_x = []
    data_y = []
    for i in tqdm(range(len(data))):
        time_x.append(data[i][0])
        freq_x.append(data[i][1])
        data_y.append(data[i][2])
    return time_x, freq_x, data_y

for iter in tqdm(range(num_of_cv)):
    dataset.update_train_val_test_keys()

    weighted_sampler = False
    train_loader = get_data(args, dataset, flag = 'train', weighted_sampler = weighted_sampler )
    val_loader = get_data(args, dataset, flag = 'vali', weighted_sampler = weighted_sampler)
    test_loader   = get_data(args, dataset, flag = 'test', weighted_sampler = weighted_sampler)

    train_dataset = InMemoryDataset(*extract_data(train_loader))
    vali_dataset = InMemoryDataset(*extract_data(val_loader))
    test_dataset = InMemoryDataset(*extract_data(test_loader))

    # train_dataset.save(f"{args.data_name}_train_{iter}.pkl")
    # vali_dataset.save(f"{args.data_name}_vali_{iter}.pkl")
    # test_dataset.save(f"{args.data_name}_test_{iter}.pkl")

    dataset_disk = DiskData(args.data_name, train_dataset, test_dataset, vali_dataset)
    dataset_disk.save()
