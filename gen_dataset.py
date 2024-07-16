from I2S0W2C2_CFC.dataloaders import data_dict, data_set
import argparse
import yaml
import os

ROOT_PATH = "./datasets"


parser = argparse.ArgumentParser(description='Generate dataset')

parser.add_argument("--data_name", type=str, help="Name of the dataset")
parser.add_argument('--exp-mode', dest='exp_mode',
                    default="LOCV", type=str, help='Set the exp type')
parser.add_argument('--representation-type', dest='representation_type',
                    default="time", type=str, help='Set the data type')

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
args.load_all = False

dataset = data_dict[args.data_name](args)

num_of_cv = dataset.num_of_cv


def get_data(self, data, flag="train", weighted_sampler = False):
    if flag == 'train':
        shuffle_flag = True 
    else:
        shuffle_flag = False

for iter in range(num_of_cv):
    dataset.update_train_val_test_keys()

    weighted_sampler = False
    train_loader = get_data(dataset, flag = 'train', weighted_sampler = weighted_sampler )
    val_loader = get_data(dataset, flag = 'vali', weighted_sampler = weighted_sampler)
    test_loader   = get_data(dataset, flag = 'test', weighted_sampler = weighted_sampler)

