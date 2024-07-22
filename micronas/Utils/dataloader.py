# Dataset stuff
from dataloaders import data_set,data_dict
from torch.utils.data import DataLoader
import numpy as np
from micronas.Utilities import DotDict
from micronas.Utils.PytorchKerasAdapter import PytorchKerasAdapter

def loadDataset(data_name="ucihar"):
    args = DotDict() 

    root_paths= {"ucihar": r"/home/king/timeseries-datasets/UCI HAR Dataset",
                 "oppo": r"C:\Users\tking\Github\timeseries-datasets\OpportunityUCIDataset/dataset",
                 "skodar": r"/Users/king/Github/timeseries-datasets/Skoda",
                 "skodal": r"C:\Users\tking\Github\timeseries-datasets\Skoda",
                 "pamap2": r"C:\Users\tking\Github\timeseries-datasets\PAMAP2_Dataset\Protocol",
                 "rw": r"C:\Users\tking\Github\timeseries-datasets\RWhar_Dataset",
                 "oppo": r"C:\Users\tking\Github\timeseries-datasets\OpportunityUCIDataset\dataset",
                 "wisdm":  r"C:\Users\tking\Github\timeseries-datasets\WISDM_ar_v1.1"}

    sampling_freqs = {
        "ucihar": 50,
        "oppo": 30,
        "skodar": 98,
        "skodal": 98,
        "pamap2": 100,
        "rw": 50,
        "oppo": 20,
        "wisdm": 20
    }

    exp_modes = {
        "ucihar": "Given",
        "rw": "Given", 
        "oppo": "FOCV",
        "skodar": "FOCV",
        "skodal": "FOCV",
        "pamap2": "Given",
        "oppo": "Given",
        "wisdm": "Given"
    }

    window_sizes = {
        "ucihar": 128,
        "oppo": 128,
        "skodar": 64,
        "skodal": 64,
        "pamap2": 128,
        "rw": 128,
        "oppo": 128,
        "wisdm": 64
    }


    args.root_path =  root_paths[data_name]
    args.freq_save_path = r"C:\Users\tking\Github\timeseries-datasets\FREQ_DATA"
    args.data_name = data_name

    args.difference = np.False_ 
    args.sampling_freq = sampling_freqs[data_name]
    args.train_vali_quote = 0.95
    # for this dataset the window size is 128
    # args.windowsize = int(2.56 * args.sampling_freq)
    args.windowsize = window_sizes[data_name]
    args.drop_long = True
    args.datanorm_type = "standardization" # None ,"standardization", "minmax"
    args.wavename = "morl"
    args.exp_mode = exp_modes[data_name]
    args.model_type = "time"

    if args.exp_mode == "FOCV":
        args.displacement =  int(1 * args.windowsize) 
    else:
        args.displacement =  int(0.5 * args.windowsize)     


    dataset = data_dict[args.data_name](args)
    print("================ {} Mode ====================".format(dataset.exp_mode))
    print("================ {} CV ======================".format(dataset.num_of_cv))


    dataset.update_train_val_test_keys()
    train_data  = data_set(args,dataset,"train", discardFreq=True)
    test_data  = data_set(args,dataset,"test", discardFreq=True)
    vali_data  = data_set(args,dataset,"vali", discardFreq=True)
    return train_data, vali_data, test_data
    

def get_dataloaders(train_data, vali_data, test_data, num_classes, batch_size=32, shuffle=True, drop_last=False, keras=True):
    train_data_loader = DataLoader(train_data,  
                                    batch_size   =  batch_size,
                                    shuffle      =  shuffle,
                                    num_workers  =  0,
                                    drop_last    =  drop_last)

    vali_data_loader = DataLoader(vali_data,  
                                    batch_size   =  batch_size,
                                    shuffle      =  shuffle,
                                    num_workers  =  0,
                                    drop_last    =  drop_last)

    test_data_loader = DataLoader(test_data,  
                                    batch_size   =  batch_size,
                                    shuffle      =  shuffle,
                                    num_workers  =  0,
                                    drop_last    =  drop_last)
    
    

    print("Data_shape: ", next(iter(train_data_loader))[0].shape)

    if keras:
        print("Using keras")
        return PytorchKerasAdapter(train_data_loader, num_classes, expand_dim=3), PytorchKerasAdapter(vali_data_loader, num_classes, expand_dim=3), PytorchKerasAdapter(test_data_loader, num_classes, expand_dim=3), num_classes
                                    

    return train_data_loader, vali_data_loader, test_data_loader, num_classes