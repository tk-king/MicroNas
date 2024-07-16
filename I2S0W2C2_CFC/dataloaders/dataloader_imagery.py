import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ========================================       Imagery_DATA               =============================

class Imagery_DATA(BASE_DATA):
    
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
            ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008','S009', 'S010'],
            ['S011', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018', 'S019', 'S020'],
            ['S021', 'S022', 'S023', 'S024', 'S025', 'S026', 'S027', 'S028', 'S029', 'S030'],
            ['S031', 'S032', 'S033', 'S034', 'S035', 'S036', 'S037', 'S039', 'S040', 'S041'],
            ['S042', 'S043', 'S044', 'S045', 'S046', 'S047', 'S048', 'S049', 'S050', 'S051'],
            ['S052', 'S053', 'S054', 'S055', 'S056', 'S057', 'S058', 'S059', 'S060', 'S061'],
            ['S062', 'S063', 'S064', 'S065', 'S066', 'S067', 'S068', 'S069', 'S070', 'S071'],
            ['S072', 'S073', 'S074', 'S075', 'S076', 'S077', 'S078', 'S079', 'S080', 'S081'],
            ['S083', 'S084', 'S085', 'S086', 'S087', 'S090', 'S091', 'S092', 'S093', 'S094', 'S095'],
            ['S096', 'S097', 'S098', 'S099', 'S101', 'S102', 'S103', 'S105', 'S106', 'S107', 'S108', 'S109']
        ]
        self.all_keys = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008',
       'S009', 'S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016',
       'S017', 'S018', 'S019', 'S020', 'S021', 'S022', 'S023', 'S024',
       'S025', 'S026', 'S027', 'S028', 'S029', 'S030', 'S031', 'S032',
       'S033', 'S034', 'S035', 'S036', 'S037', 'S039', 'S040', 'S041',
       'S042', 'S043', 'S044', 'S045', 'S046', 'S047', 'S048', 'S049',
       'S050', 'S051', 'S052', 'S053', 'S054', 'S055', 'S056', 'S057',
       'S058', 'S059', 'S060', 'S061', 'S062', 'S063', 'S064', 'S065',
       'S066', 'S067', 'S068', 'S069', 'S070', 'S071', 'S072', 'S073',
       'S074', 'S075', 'S076', 'S077', 'S078', 'S079', 'S080', 'S081',
       'S083', 'S084', 'S085', 'S086', 'S087', 'S090', 'S091', 'S092',
       'S093', 'S094', 'S095', 'S096', 'S097', 'S098', 'S099', 'S101',
       'S102', 'S103', 'S105', 'S106', 'S107', 'S108', 'S109']
        
        self.sub_ids_of_each_sub = {}

        self.exp_mode     = args.exp_mode
        self.split_tag    = "sub"
        
        self.file_encoding = {}  # no use 
        

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]
        super(Imagery_DATA, self).__init__(args)
        
    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -----------Imagery--------")

        df = pd.read_csv(os.path.join(root_path,"Imagery.csv"))
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