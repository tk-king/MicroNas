import os
import pickle
import torch
from torch.utils.data import Dataset

BASE_DATASET_FOLDER = "datasets_pickle"

class InMemoryDataset(Dataset):
    def __init__(self, time_x, freq_x, data_y):
        assert len(time_x) == len(data_y) == len(freq_x)
        self.time_x = [x for x in time_x]
        self.freq_x = [x for x in freq_x]
        self.data_y = [x for x in data_y]

    def __len__(self):
        return len(self.time_x)
    
    def __getitem__(self, idx):
        return self.time_x[idx], self.freq_x[idx], self.data_y[idx]
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "time_x": self.time_x,
                "freq_x": self.freq_x,
                "data_y": self.data_y}, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        time_x = torch.tensor(data["time_x"], dtype=torch.float32)
        freq_x = torch.tensor(data["freq_x"], dtype=torch.float32)
        data_y = torch.tensor(data["data_y"], dtype=torch.long)
        return InMemoryDataset(time_x, freq_x, data_y)
    

class DiskData:
    def __init__(self, dataset_name, cv, train_dataloader, test_dataloader, vali_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.vali_dataloader = vali_dataloader
        self.dataset_name = dataset_name
        self.cv = cv

    def save(self):
        path = os.path.join(BASE_DATASET_FOLDER, self.dataset_name, str(self.cv))
        if not os.path.exists(path):
            os.makedirs(path)
        train_path = os.path.join(path, "train.pkl")
        test_path = os.path.join(path, "test.pkl")
        vali_path = os.path.join(path, "vali.pkl")
        self.train_dataloader.save(train_path)
        self.test_dataloader.save(test_path)
        self.vali_dataloader.save(vali_path)
    
    @staticmethod
    def load(dataset_name, cv):
        path = os.path.join(BASE_DATASET_FOLDER, dataset_name, str(cv))
        if os.path.exists(path) == False:
            raise Exception(f"Dataset {dataset_name} not found")
        train_path = os.path.join(path, "train.pkl")
        test_path = os.path.join(path, "test.pkl")
        vali_path = os.path.join(path, "vali.pkl")
        train_dataloader = InMemoryDataset.load(train_path)
        test_dataloader = InMemoryDataset.load(test_path)
        vali_dataloader = InMemoryDataset.load(vali_path)
        return DiskData(dataset_name, cv, train_dataloader, test_dataloader, vali_dataloader)
