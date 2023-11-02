import torch
from torch.utils.data import DataLoader
from typing import Dict
from pydantic import BaseModel, Field

import tensorflow.keras as keras

import tensorflow_model_optimization as tfmot


from src.Utils.PytorchKerasAdapter import PytorchKerasAdapter
from src.TfLite.structure import TfLiteModel
from src.Nas.Networks.Pytorch.SearchNet import SearchNet
from src.Nas.Search import ArchSearcher
from src.config import Config
from src.Profiler.LatMemProfiler import set_ignore_latency, _lookUp
from src.Nas.Networks.Pytorch.SearchModule import InferenceType


class MicroNASConfig(BaseModel):
    train_epochs: int = Field(default=100)
    retrain_epochs : int = Field(default=100)
    


def search(dataset_train: DataLoader, dataset_vali: DataLoader, dataset_test: DataLoader, num_classes, config: Dict = {}, latency_limit=None, memory_limit=None, callback=None):

    config = MicroNASConfig(**config)

    Config.devce = "cpu"
    Config.search_epochs = 1

    set_ignore_latency(True)

    ts_len, num_sensors = next(iter(dataset_train))[0].shape[1:3]
    nas_net = SearchNet([ts_len, num_sensors], num_classes).to(Config.device)
    weights = nas_net.get_nas_weights()
    fake_input = torch.randn((1, 1, ts_len, num_sensors)).to(Config.device)
    print("output_shape: ", nas_net(fake_input))
    searcher = ArchSearcher(nas_net)

    searcher.train(dataset_train, dataset_vali, config.train_epochs, latency_limit, memory_limit, callback=callback)

    keras_model = nas_net.getKeras(getPruned=True, batch_size=None, inf_type=InferenceType.MAX_WEIGHT)
    keras_model = keras_model.get_config()
    keras_model = keras.Model.from_config(keras_model)

    quantize_model = tfmot.quantization.keras.quantize_model
    keras_model = quantize_model(keras_model)

    keras_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", Precision(), Recall(), f1_m])
    keras_model.build(input_shape=(1, 128, 9, 1))

    dataset_train_keras = PytorchKerasAdapter(dataset_train, num_classes, expand_dim=3)
    dataset_vali_keras = PytorchKerasAdapter(dataset_vali, num_classes, expand_dim=3)
    dataset_test_keas = PytorchKerasAdapter(dataset_test, num_classes, expand_dim=3)

    keras_model.fit(dataset_train_keras, epochs=config.retrain_epochs, validation_data=dataset_vali_keras)

    tflm_model = TfLiteModel(keras_model, (ts_len, num_sensors, 1),  rep_dataset=dataset_train_keras)

    tflm_model_bytes = tflm_model.byte_model()

    return keras_model, tflm_model_bytes

if __name__ == "__main__":
    search()