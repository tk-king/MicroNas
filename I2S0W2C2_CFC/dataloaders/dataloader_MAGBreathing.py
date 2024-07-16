import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       MAG_Breathing_DATA               =============================

class MAG_Breathing_DATA(BASE_DATA):
    
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
        The 15 activities executed by the participants were: 
        standing normal breathing (STNB); 
        seated normal breathing (SNB); 
        seated guided-breathing (SGB); 
        normal/deep-alternated breathing (MIXB); 
        march (MCH); 
        squat (SQT); 
        adduction/abduction of the left/right arm (AAL/AAR); 
        adduction/abduction of the left/right leg (ALL/ALR); 
        upwards (overhead) left/right arm extension (UAL/UAR); 
        shoulder elevation (SE); 
        side stretch (SS); 
        and seated trunk rotation (TR).
        """
        self.used_cols = None

        # ------------------- modify ----------------------
        self.col_names =  [ 'Airflow (ml)' ,'sub', 'sub_id', 'activity_id']
        
        
        self.pos_filter         = None
        self.sensor_filter      = None
        self.selected_cols      = None

        # 'up', 'waveIn', 'relax', 'waveOut', 'fist', 'open', 'pinch', 'down', 'left', 'right', 'forward', 'backward'
        self.label_map = [
            (0,'AAL'),
            (1,'AAR'),
            (2,'ALL'), 
            (3,'ALR'), 
            (4,'MCH'), 
            (5,'MIXB'),
            (6,'SE'), 
            (7,'SGB'),
            (8,'SNB'),
            (9,'SQT'), 
            (10,'SS'), 
            (11,'STNB'), 
            (12,'TR'), 
            (13,'UAL'), 
            (14,'UAR')
        ]

        self.drop_activities = []


        self.train_keys   = []
        self.vali_keys    = []
        self.test_keys    = []
        
        self.LOCV_keys = [
            ['1BST', '2QWT', '7OYX'],                  
            ['83J1', '9TUL', 'D4GQ'], 
            ['EPE2', 'F9AF', 'FTD7'],
            ['G8B7', 'HAK8', 'NO15'],
            ['P4W9', 'QMQ7', 'W8Z9', 'Y6O3']
        ]
        self.all_keys = ['1BST', '2QWT', '7OYX', 
                         '83J1', '9TUL', 'D4GQ', 
                         'EPE2', 'F9AF', 'FTD7', 
                         'G8B7', 'HAK8', 'NO15', 
                         'P4W9', 'QMQ7', 'W8Z9', 'Y6O3']
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(MAG_Breathing_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data ------  MAG -------------",self.col_names)

        df = pd.read_csv(os.path.join(root_path,"MAG_Breathing.csv"))
        df = df[self.col_names]

        
        df["activity_id"] = df["activity_id"].astype(str)
        df["sub"] = df["sub"].astype(str)


        for sub in df["sub"].unique():
            temp_sub = df[df["sub"]==sub]
            self.sub_ids_of_each_sub[sub] = list(temp_sub["sub_id"].unique())

        # downsampling 2
        index_list = list(np.arange(0,df.shape[0],2))
        df = df.iloc[index_list]
        
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