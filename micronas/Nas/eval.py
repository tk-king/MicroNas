from micronas.Utils.PytorchKerasAdapter import PytorchKerasAdapter
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def eval_keras(self, model, train_dataloader, vali_dataloader, test_dataloader):

    res = {}

    loaders = [train_dataloader, vali_dataloader, test_dataloader]
    names = ["train", "vali", "test"]
    for dataloader, data_name in zip(loaders, names):
        dataloader_keras = PytorchKerasAdapter(dataloader, 10)
        preds = model.predict(dataloader_keras)
        y_true = [x[1] for x in dataloader]
        y_true = np.concatenate(y_true)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average="macro")
        res[f"acc_{data_name}"] = acc
        res[f"f1_{data_name}"] = f1
    return res

def eval_keras_quant(self, tf_lite_model, train_dataloader, vali_dataloader, test_dataloader):
    
    res = {}
    loaders = [train_dataloader, vali_dataloader, test_dataloader]
    names = ["train", "vali", "test"]

    # for dataloader, data_name in zip(loaders, names):
    #     dataloader_keras = PytorchKerasAdapter(dataloader, 10)
    #     preds = tf_lite_model.predict(dataloader_keras)
    #     y_true = [x[1] for x in dataloader]
    #     y_true = np.concatenate(y_true)
    #     acc = accuracy_score(y_true, preds)
    #     f1 = f1_score(y_true, preds, average="macro")
    #     res[f"acc_{data_name}"] = acc
    #     res[f"f1_{data_name}"] = f1

    return res