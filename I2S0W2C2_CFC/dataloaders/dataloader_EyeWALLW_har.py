import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       EyeWALLW_DATA               =============================

class EyeWALLW_DATA(BASE_DATA):
    
    def __init__(self, args):

        self.used_cols = None
        
        self.col_names = [
            'accel_x', 'accel_y', 'accel_z', 
            'gyro_x', 'gyro_y', 'gyro_z',
            #'norm_pos_x', 'norm_pos_y', 
            #'gaze_point_3d_x', 'gaze_point_3d_y', 'gaze_point_3d_z', 
            'activity_id', 'sub', 'sub_id']
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None


        self.label_map = label_map = [
            #(0,'noise_text'), (1,'noise_image'), (2,'noise_video'), 
            (3,'vertical_down'),(4,'near_far'), (5,'vertical_up'), (6,'far_near'),
            (7,'horizontal_left'),(8,'circle_clock'), (9,'horizontal_right'), 
            (10,'circle_anticlock'), (11,'square_anticlock'), (12,'square_clock'),
        ]
        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        # [ 0,  1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  2,  3,  4,  5,  6, 7,  8,  9]
        self.LOCV_keys = [
            ["0","1","2","3","4"],["5","6","7","8","9"],["10","11","12","13","14"],["15","16","17","18","19"],

        ]
        self.all_keys = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"]
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(EyeWALLW_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")


        df = pd.read_csv(os.path.join(root_path,"Eye_all.csv")
        df = df[self.col_names]
        

        df["sub"] = df["sub"].astype(str)
        for sub in df["sub"].unique():
            temp_sub = df[df["sub"]==sub]
            self.sub_ids_of_each_sub[sub] = list(temp_sub["sub_id"].unique())
        

        df.reset_index(drop=True,inplace=True)

        df = df.set_index('sub_id')
        df = df[list(df.columns)[:-2]+["sub"]+["activity_id"]]

        label_mapping = {item[1]:item[0] for item in self.label_map}


        df["activity_id"] = df["activity_id"].map(labelToId)
        df.dropna(inplace=True)
        data_y = df.iloc[:,-1]
        data_x = df.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y