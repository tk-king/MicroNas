from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType
from micronas.Nas.SearchStrategy.SearchStrategy import SearchStrategy
from micronas.Utils.PytorchKerasAdapter import PytorchKerasAdapter
from micronas.Nas.ModelResult import ModelResult
from micronas.config import Config
from micronas.TfLite.structure import TfLiteModel
from micronas.Nas.eval import eval_keras, eval_keras_quant
from typing import List
import torch
import logging

class RandomSearchStrategy(SearchStrategy):
    def __init__(self, search_space, arch_tries=20):
        self.search_space = search_space
        self.arch_tries = arch_tries
        self.logger = logging.getLogger(__name__)

    def _fit_keras(self, model, train_dataloader):
        train_dataloader_keras = PytorchKerasAdapter(train_dataloader, 10)
        self.logger.info("Len keras dataloader: %s", len(train_dataloader_keras))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_dataloader_keras, epochs=Config.retrain_epochs)

    def search(self, train_dataloader, vali_dataloader, latency_limit, memory_limit, **kwargs) -> List[ModelResult]:
        ctr = 0
        models = []
        while ctr < self.arch_tries:
            fake_input = torch.randn((1, 1, 128, 9))
            lat, mem = self.search_space(fake_input, inf_type=InferenceType.RANDOM)[1:]


            if latency_limit is not None and lat > latency_limit:
                continue
            if memory_limit is not None and mem > memory_limit:
                continue
            try:
                keras_model = self.search_space.getKeras(getPruned=True, inf_type=InferenceType.MAX_WEIGHT, batch_size=None)
            except Exception as e:
                continue
            self._fit_keras(keras_model, train_dataloader)
            
            train_dataloader_keras = PytorchKerasAdapter(train_dataloader, 10)
            tf_lite = TfLiteModel(keras_model, (ts_len, num_sensors, 1), train_dataloader_keras)


            eval_keras = eval_keras(keras_model, train_dataloader, vali_dataloader, test_dataloader)
            eval_tf_lite = eval_keras_quant(tf_lite, train_dataloader, vali_dataloader, test_dataloader)


            # accuracy = self._eval_keras(keras_model, vali_dataloader)




            models.append(ModelResult(
                model=keras_model,
                accuracy=accuracy,
                latency=lat,
                memory=mem
            ))
            ctr += 1
        return models


    def extractArch():
        pass