import pandas as pd
import numpy as np
import os

from I2S0W2C2_CFC.dataloaders.dataloader_base import BASE_DATA

# ========================================       GESHOME_GESTURE_DATA               =============================

class GESHOME_GESTURE_DATA(BASE_DATA):
    
    def __init__(self, args):

        """

        ABSTRACT 
        GesHome dataset consists of 18 hand gestures from 20 non-professional subjects with various ages and occupation. 
        The participant performed 50 times for each gesture in 5 days. 
        Thus, GesHome consists of 18000 gesture samples in total. 
        Using embedded accelerometer and gyroscope, we take 3-axial linear acceleration and 3-axial angular velocity with frequency equals to 25Hz. 
        The experiments have been video-recorded to label the data manually using ELan tool.
        
        The collected files include columns in the order of 
        {datetime, accelerometer x, accelerometer y, accelerometer z, gyroscope x, gyroscope y, gyroscope z, magnetometer x, magnetometer y, magnetometer z, activity label}, 
        atetime column is set as format "yyyy-MM-dd;HH:mm:ss.SSS", while the label column format is string.
        
        Instructions: 
        GesHome dataset consists of 18 hand gestures from 20 non-professional subjects with various ages and occupation. 
        The participant performed 50 times for each gesture in 5 days. Thus, GesHome consists of 18000 gesture samples in total. 
        Using embedded accelerometer and gyroscope, we take 3-axial linear acceleration and 3-axial angular velocity with frequency equals to 25Hz. 
        The experiments have been video-recorded to label the data manually using ELan tool.
        
        The collected files include columns in the order of 
        {datetime, accelerometer x, accelerometer y, accelerometer z, gyroscope x, gyroscope y, gyroscope z, magnetometer x, magnetometer y, magnetometer z, activity label}, 
        datetime column is set as format "yyyy-MM-dd;HH:mm:ss.SSS", while the label column format is string.
        """
        self.used_cols = None
        
        self.col_names = [ 
            'acc_x', 'acc_y', 'acc_z', 
            'gyo_x', 'gyo_y', 'gyo_z', 
            'mag_x', 'mag_y', 'mag_z', 
            'sub', 'sub_id', 'activity_id']
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None

        # 'up', 'waveIn', 'relax', 'waveOut', 'fist', 'open', 'pinch', 'down', 'left', 'right', 'forward', 'backward'
        self.label_map = [(0,'Move_up;'), (1,'Move_left;'), (2,'Move_down;'), (3,'Move_right;'), (4,'Select;'),
                          (5,'Clap;'), (6,'CWCircle;'),(7, 'CCWCircle;'),
                          (8,'0;'), (9,'1;'), (10,'2;'), (11,'3;'), (12,'4;'), (13,'5;'),
                          (14,'6;'), (15,'7;'), (16,'8;'), (17,'9;')]
        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        
        self.LOCV_keys = [
            ['S1','S2', 'S3', 'S4'], 
            ['S5', 'S6', 'S7', 'S8'], 
            ['S9','S10', 'S11', 'S12'],
            ['S13', 'S14', 'S15', 'S16'],
            ['S17', 'S18', 'S19', 'S20']
        ]
        self.all_keys = ['S1','S2', 'S3', 'S4', 
                         'S5', 'S6', 'S7', 'S8', 
                         'S9','S10', 'S11', 'S12',
                         'S13', 'S14', 'S15', 'S16', 
                         'S17', 'S18', 'S19', 'S20']
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(GESHOME_GESTURE_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")


        df = pd.read_csv(os.path.join(root_path,"geshome.csv"))
        df = df[self.col_names]


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