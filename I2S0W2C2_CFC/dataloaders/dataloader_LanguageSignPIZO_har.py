import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       LANGUAGEPIZO_GESTURE_DATA               =============================

class LANGUAGEPIZO_GESTURE_DATA(BASE_DATA):
    
    def __init__(self, args):

        """

        ABSTRACT 
        The dataset contains data obtained by measuring hand movements while performing the letters of the Polish Sign Language alphabet. 
        It contains data from 16 users performing all 36 letters ten times. 
        Each single execution of a gesture is recorded in 75 samples. 
        The experiment also included data augmentation, multiplying the number of data by 200. times.

        Instructions: 
        Each single gesture execution is recorded in 75 samples. 
        The data are stored in the columns 
        exam_id, P1_1, P1_2, P2_1, P2_2, P3_1, P3_2, P4_1, P4_2, P5_1, P5_2, Euler_x, Euler_y, Euler_z, Acc_x, Acc_y, Acc_z, sign, timestamp 
        denoting in turn: subject id, finger piezoresistive sensor readings prefixed with P, gyro sensor prefixed with Euler, accelerometric sensor prefixed with Acc, 
        sign performed, timestamp of each sample.
        """
        self.used_cols = None
        
        self.col_names = [ 
            'P1_1', 'P1_2', 
            'P2_1', 'P2_2', 
            'P3_1', 'P3_2', 
            'P4_1', 'P4_2',
            'P5_1', 'P5_2', 
            # 'Euler_x', 'Euler_y', 'Euler_z', 
            # 'Acc_x', 'Acc_y', 'Acc_z', 
            'sub',
            'activity_id', 
            'sub_id']
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None

        # 'up', 'waveIn', 'relax', 'waveOut', 'fist', 'open', 'pinch', 'down', 'left', 'right', 'forward', 'backward'
        self.label_map = [
            (0,'a '), (1,'b '), (2,'c '), (3,'e '), (4,'i '), (5,'l '), (6,'m '), (7,'n '), (8,'o '), (9,'p '), (10,'r '), 
            (11,'s '), (12,'t '), (13,'u '), (14,'w '), (15,'y '), (16,'Ä… '), (17,'Ä‡ '), (18,'ch'), (19,'cz'), (20,'d '), 
            (21,'Ä™ '), (22,'f '), (23,'h '), (24,'j '), (25,'k '), (26,'Ĺ‚ '), (27,'Ĺ„ '), (28,'Ăł '), (29,'rz'), (30,'Ĺ› '),
            (31,'sz'), (32,'z '), (33,'Ĺş '), (34,'ĹĽ '), (35,'g ')]
        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        
        self.LOCV_keys = [
            ["('wd', 1)", "('jp', 2)", "('mp', 3)"],
            ["('ku', 4)", "('dd', 5)", "('sw', 6)"],
            ["('zp', 7)", "('ku', 8)", "('pp', 9)"],
            ["('mo', 10)", "('sk', 11)", "('mu', 12)"],
            ["('mk', 13)", "('mo', 14)", "('tt', 15)"],
            ["('mp', 16)", "('jP', 17)"]
        ]
        self.all_keys = ["('wd', 1)",
                         "('jp', 2)",
                         "('mp', 3)",
                         "('ku', 4)",
                         "('dd', 5)",
                         "('sw', 6)",
                         "('zp', 7)",
                         "('ku', 8)",
                         "('pp', 9)",
                         "('mo', 10)",
                         "('sk', 11)",
                         "('mu', 12)",
                         "('mk', 13)",
                         "('mo', 14)", 
                         "('tt', 15)",
                         "('mp', 16)",
                         "('jP', 17)"]
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(LANGUAGEPIZO_GESTURE_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")



        df = pd.read_csv(os.path.join(root_path,"new_df.csv"))
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