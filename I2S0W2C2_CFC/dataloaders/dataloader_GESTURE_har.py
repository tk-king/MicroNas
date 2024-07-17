import pandas as pd
import numpy as np
import os

from I2S0W2C2_CFC.dataloaders.dataloader_base import BASE_DATA

# ========================================       GESTURE_HAR_DATA               =============================

class GESTURE_HAR_DATA(BASE_DATA):
    
    def __init__(self, args):
        self.used_cols = None
        
        self.col_names = ['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8', 
                          'q1', 'q2', 'q3', 'q4', 
                          'g1', 'g2', 'g3', 
                          'a1', 'a2', 'a3', 
                          'sub',
                          'activity_id', 
                          'sub_id']
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None

        # 'up', 'waveIn', 'relax', 'waveOut', 'fist', 'open', 'pinch', 'down', 'left', 'right', 'forward', 'backward'
        self.label_map = [(0, 'up'),
                          (1, "waveIn"), 
                          (2, "relax"), 
                          (3, "waveOut"),
                          (4, "fist"),
                          (5, 'open'),
                          (6, 'pinch'),
                          (7, 'down'),
                          (8, 'left'),
                          (9, 'right'),
                          (10,'forward'),
                          (11,'backward')
                         ]
        self.drop_activities = []


        self.given_train_keys   = ['training001', 'training025', 'training026', 'training027',
                             'training032', 'training033', 'training035', 'training036',
                             'training037', 'training038', 'training024', 'training039',
                             'training040', 'training041', 'training042', 'training043']
        self.given_vali_keys    = []
        self.given_test_keys    = ['test001', 'test024', 'test025', 'test026',
                             'test027', 'test028', 'test029', 'test002',
                             'test038', 'test039', 'test040', 'test041',
                             'test042', 'test003', 'test004', 'test023']
        
        self.LOCV_keys = [
            ['test001', 'test024', 'test025', 'test026',
             'test027', 'test028', 'test029', 'test002',
             'test038', 'test039', 'test040', 'test041',
             'test042', 'test003', 'test004', 'test023'],
            ['training001', 'training025', 'training026', 'training027',
             'training032', 'training033', 'training035', 'training036',
             'training037', 'training038', 'training024', 'training039',
             'training040', 'training041', 'training042', 'training043']
        ]
        self.all_keys = ['training001', 'training025', 'training026', 'training027',
                         'training032', 'training033', 'training035', 'training036',
                         'training037', 'training038', 'training024', 'training039',
                         'training040', 'training041', 'training042', 'training043',
                         'test001', 'test024', 'test025', 'test026',
                         'test027', 'test028', 'test029', 'test002',
                         'test038', 'test039', 'test040', 'test041',
                         'test042', 'test003', 'test004', 'test023']
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(GESTURE_HAR_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")
        df = pd.read_csv(os.path.join(root_path,"all.csv"))

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
        df.dropna(inplace=True)
        data_y = df.iloc[:,-1]
        data_x = df.iloc[:,:-1]


        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y