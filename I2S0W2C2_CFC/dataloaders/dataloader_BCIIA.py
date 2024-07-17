import pandas as pd
import numpy as np
import os

from I2S0W2C2_CFC.dataloaders.dataloader_base import BASE_DATA

# ========================================       BCIIA_DATA               =============================

class BCIIA_DATA(BASE_DATA):
    
    def __init__(self, args):

        """


        """
        self.used_cols = None

        # ------------------- modify ----------------------
        self.col_names =  []
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None

        # 'up', 'waveIn', 'relax', 'waveOut', 'fist', 'open', 'pinch', 'down', 'left', 'right', 'forward', 'backward'
        self.label_map = [
            (1,'1'),
            (2,'2'), 
            (3,'3'),
            (4,'4'), 
                    
        ]

        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        
        self.LOCV_keys = [
            ['A01'], ['A02'], ['A03'], ['A04'], ['A05'], ['A06'], ['A07'], ['A08', 'A09']
        ]
        self.all_keys = [
            'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'
        ]
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(BCIIA_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data ----------bciiia---------")

        df = pd.read_csv(os.path.join(root_path,"bciiia.csv"))
        #df = df[self.col_names]

        
        df["activity_id"] = df["activity_id"].astype(str)
        df["sub"] = df["sub"].astype(str)


        for sub in df["sub"].unique():
            temp_sub = df[df["sub"]==sub]
            self.sub_ids_of_each_sub[sub] = list(temp_sub["sub_id"].unique())
        
        df.reset_index(drop=True,inplace=True)
        #index_list = list(np.arange(0,df.shape[0],2))
        #df = df.iloc[index_list]
        
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