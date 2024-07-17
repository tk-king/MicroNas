import pandas as pd
import numpy as np
import os

from I2S0W2C2_CFC.dataloaders.dataloader_base import BASE_DATA

# ========================================       SENSED_GESTURE_DATA               =============================

class SENSED_GESTURE_DATA(BASE_DATA):
    
    def __init__(self, args):

        """

        Overal Description
        All the files follow the same structure and contain 9 values in each row. 
        The first field denotes the Epoch time in milliseconds that the data arrived in the application, 
        the second field includes the relative timestamp, as reported by the OS, where the data arrived, 
        the next six fields include the x, y, and z axis values of the accelerometer and the gyroscope sensors respectively, 
        while and the last field contains the label for the executed gesture. 
        The label takes values from 0 to 45, where 0 denotes the character 
        "A", 25 the character "Z", 26 the character "0", 35 the character "9" and 36 to 45 the special characters in the order that are described above.
        """
        self.used_cols = None
        
        self.col_names = [ 
            'acc_x', 'acc_y', 'acc_z', 
            'gry_x', 'gry_y', 'gry_z', 
            'activity_id',
            'sub', 'sub_id']
            
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None

        # 'up', 'waveIn', 'relax', 'waveOut', 'fist', 'open', 'pinch', 'down', 'left', 'right', 'forward', 'backward'
        self.label_map = [
            (0,"0"), (1,"1"), (2,"2"), (3,"3"), (4,"4"), (5,"5"), 
            (6,"6"), (7,"7"), (8,"8"), (9,"9"), (10,"10"), 
            (11,"11"), (12,"12"), (13,"13"), (14,"14"), (15,"15"), 
            (16,"16"), (17,"17"), (18,"18"), (19,"19"), (20,"20"), 
            (21,"21"), (22,"22"), (23,"23"), (24,"24"), (25,"25")]
        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        
        self.LOCV_keys = [
            ["1","2","3","4","5"],
            ["6","7","8","9","10"],
            ["11","12","13","14","15"],
            ["16","17","18","19","20"],
            ["21","22","23","24","25"]
        ]
        self.all_keys = [ "1","2","3","4","5","6",
                         "7","8","9","10","11","12",
                         "13","14","15","16","17",
                         "18","19","20","21",
                         "22","23","24","25"]
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(SENSED_GESTURE_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")




        df = pd.read_csv(os.path.join(root_path,"sensed.csv"))

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