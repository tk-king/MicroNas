from othermodels import get_model_by_name
from InMemoryDataset import InMemoryDataset, DiskData
from micronas.config import Config
import torch

def _train(model, train_data_loader):
    
    for epoch in range(Config.retrain_epochs):
        model.train()

        optim = torch.optim.Adam(model.parameters())
        
        for i, (input_time, input_freq, target) in enumerate(train_data_loader):
            model.zero_grad()
            output = model(input_time)
            loss = model.loss(output, target)
            loss.backward()
            optim.step()
            print("Epoch: ", epoch, " / ", Config.retrain_epochs, " Loss: ", loss.item())


def train_eval(dataset, cv, model_name):
    diskData = DiskData.load(dataset, cv)

    diskData = DiskData.load(dataset, cv)
    train_dataloader = diskData.train_dataloader
    test_dataloader = diskData.test_dataloader
    vali_dataloader = diskData.vali_dataloader

    num_classes_lookup = {
        "skodar": 10
    }
    NUM_CLASSES = num_classes_lookup[dataset]

    ts_len, num_sensors = next(iter(train_dataloader))[0].shape[1:3]
    
    model = get_model_by_name(model_name, ts_len, num_sensors, NUM_CLASSES)

    _train(model, train_dataloader)