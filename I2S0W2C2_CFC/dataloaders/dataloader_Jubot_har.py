import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       Jubot_DATA               =============================

class Jubot_DATA(BASE_DATA):
    
    def __init__(self, args):

        self.used_cols = None
        
        self.col_names = [ 
            'Acc_X_1', 'Acc_Y_1', 'Acc_Z_1', 'Gyro_X_1', 'Gyro_Y_1',
            'Gyro_Z_1', 'Acc_X_2', 'Acc_Y_2', 'Acc_Z_2', 'Gyro_X_2', 'Gyro_Y_2',
            'Gyro_Z_2', 'Acc_X_3', 'Acc_Y_3', 'Acc_Z_3', 'Gyro_X_3', 'Gyro_Y_3',
            'Gyro_Z_3', 'sub', 'activity_id',  'sub_id']
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None

        # 'up', 'waveIn', 'relax', 'waveOut', 'fist', 'open', 'pinch', 'down', 'left', 'right', 'forward', 'backward'
        self.label_map = [
            (0,'drinking'), (1,'walking'), (2,'jumping'), (3,'standing')]
        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        
        self.LOCV_keys = [
            ["2"],
            ["1"]
        ]
        self.all_keys = ["1",
                         "2"]
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(Jubot_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")



        df = pd.read_csv(os.path.join(root_path,"df_all.csv"))
        df = df[self.col_names]
        df = df.interpolate(method='polynomial', order=2)

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