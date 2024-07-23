from othermodels import get_model_by_name
from InMemoryDataset import InMemoryDataset, DiskData
from micronas.config import Config
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from micronas.Nas.eval import eval_torch
from micronas.Nas.ModelResult import ModelResult

logger = logging.getLogger(__name__)

def _train(model, train_data_loader):
    
    for epoch in range(Config.retrain_epochs):
        model.train()

        optim = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for i, (input_time, input_freq, target) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            model.zero_grad()
            output = model(input_time)
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
        print("Epoch: ", epoch, " / ", Config.retrain_epochs, " Loss: ", loss.item())
    

def train_eval(dataset, cv, model_name):
    diskData = DiskData.load(dataset, cv)

    diskData = DiskData.load(dataset, cv)
    train_dataset = diskData.train_dataloader
    test_dataset = diskData.test_dataloader
    vali_dataset = diskData.vali_dataloader

    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    vali_dataloader = DataLoader(vali_dataset, batch_size=Config.batch_size, shuffle=False)



    num_classes_lookup = {
        "skodar": 10
    }
    NUM_CLASSES = num_classes_lookup[dataset]

    ts_len, num_sensors = next(iter(train_dataset))[0].shape[1:3]
    logger.info(f"ts_len: {ts_len}, num_sensors: {num_sensors}, num_classes: {NUM_CLASSES}")

    model = get_model_by_name(model_name, ts_len, num_sensors, NUM_CLASSES)

    _train(model, train_dataloader)
    res = eval_torch(model, train_dataloader, vali_dataloader, test_dataloader)

    model_result = ModelResult(
        keras_model=None,
        tf_lite_model=None,
        torch_Model=model,
        config=Config,
        eval=res
    )
    return model_result