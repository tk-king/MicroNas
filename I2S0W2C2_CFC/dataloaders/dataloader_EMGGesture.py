import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       EMG_GESTURE_DATA               =============================

class EMG_GESTURE_DATA(BASE_DATA):
    
    def __init__(self, args):

        """

        ABSTRACT 
        Electromyography (EMG) has limitations in human machine interface due to disturbances like electrode-shift, fatigue, and subject variability. 
        A potential solution to prevent model degradation is to combine multi-modal data such as EMG and electroencephalography (EEG). 
        This study presents an EMG-EEG dataset for enhancing the development of upper-limb assistive rehabilitation devices. 
        The dataset, acquired from thirty-three volunteers without neuromuscular dysfunction or disease using commercial biosensors is easily replicable and deployable. 
        The dataset consists of seven distinct gestures to maximize performance on the Toronto Rehabilitation Institute hand function test and the Jebsen-Taylor hand function test. 
        The authors aim for this dataset to benefit the research community in creating intelligent and neuro-inspired upper limb assistive rehabilitation devices.
        
        Instructions: 
        The acquisition of the electromyography data (EMG) was performed using the defunct Myo armband, consisting of 8-channels with a 200 Hz sampling frequency. 
        The EMG data, acquired from the upper-limb at maximum voluntary contraction, remains raw and unfiltered. In addition, electroencephalography (EEG) data was collected through the use of openBCI Ultracortex IV, 
        composed of 8-channels with a 250 Hz sampling frequency. The dataset is accessible both in .CSV and .MAT formats, with individual subject data in a singular directory. 
        A supervised machine learning approach can be undertaken while utilizing the naming nomenclature of the files. The file name designates the subject as S{}, repetition as R{}, and gesture as G{} respectively.
        The dataset consists of six repetitions per gesture and seven gestures in total. The gesture numbering scheme is as follows: 
        G1 represents a large diameter grasp, 
        G2 a medium diameter grasp, 
        G3 a three-finger sphere grasp, 
        G4 a prismatic pinch grasp, 
        G5 a power grasp, 
        G6 a cut grasp, 
        and G7 an open hand. 
        Detailed description of the dataset including starter code can be found here: https://github.com/HumanMachineInterface/Gest-Infer
        """
        self.used_cols = None
        
        self.col_names =  [ 
            'col_1', 'col_2', 'col_3', 'col_4',
            'col_5', 'col_6', 'col_7', 'col_8',
            'sub', 'activity_id', 'sub_id'
        ]
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None

        # 'up', 'waveIn', 'relax', 'waveOut', 'fist', 'open', 'pinch', 'down', 'left', 'right', 'forward', 'backward'
        self.label_map = [
            (0,"0"), (1,"1"), (2,"2"), (3,"3"), (4,"4"), (5,"5"), 
            (6,"6")
        ]

        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        
        self.LOCV_keys = [
            ['1', '2', '3', '4', '5'], 
            ['6', '7', '8', '9', '10'], 
            ['11', '12', '13', '14', '15'], 
            ['16', '17', '18', '19', '20'], 
            ['21', '22', '23', '24', '25'],
            ['26', '27', '28', '29', '30'],
            ['31', '32', '33']
        ]
        self.all_keys = ['1', '2', '3', '4', '5', 
                         '6', '7', '8', '9', '10', 
                         '11', '12', '13', '14', '15', 
                         '16', '17', '18', '19', '20', 
                         '21', '22', '23', '24', '25', 
                         '26', '27', '28', '29', '30', 
                         '31', '32', '33']
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        print("self.no_drop_activites :", self.no_drop_activites)
        super(EMG_GESTURE_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data ------emg.csv-------------")

        df = pd.read_csv(os.path.join(root_path,"emg.csv"))
        df.columns = self.col_names

        
        df["activity_id"] = df["activity_id"].astype(str)
        df["sub"] = df["sub"].astype(str)


        for sub in df["sub"].unique():
            temp_sub = df[df["sub"]==sub]
            self.sub_ids_of_each_sub[sub] = list(temp_sub["sub_id"].unique())
        
        df = df.set_index('sub_id')
        df = df[list(df.columns)[:-2]+["sub"]+["activity_id"]]


        label_mapping = {item[1]:item[0] for item in self.label_map}

        df["activity_id"] = df["activity_id"].map(label_mapping)
        df["activity_id"] = df["activity_id"].map(self.labelToId)
        print(" unique df[activity_id] :", df["activity_id"].unique())
        df.dropna(inplace=True)
        data_y = df.iloc[:,-1]
        data_x = df.iloc[:,:-1]



        data_x = data_x.reset_index()

        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y