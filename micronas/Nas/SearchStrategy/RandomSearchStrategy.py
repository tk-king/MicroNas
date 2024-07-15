from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType
from micronas.Nas.SearchStrategy.SearchStrategy import SearchStrategy
from micronas.Utils.PytorchKerasAdapter import PytorchKerasAdapter
from micronas.config import Config
import torch


class RandomSearchStrategy(SearchStrategy):
    def __init__(self, search_space, arch_tries=20):
        self.search_space = search_space
        self.arch_tries = arch_tries

    def _fit_keras(self, model, train_dataloader):
        train_dataloader_keras = PytorchKerasAdapter(train_dataloader, 6, expand_dim=3)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("shape:", next(iter(train_dataloader_keras))[0].shape)
        model.fit(train_dataloader_keras, epochs=Config.retrain_epochs)

    def _eval_keras(self, model, vali_dataloader):
        vali_dataloader_keras = PytorchKerasAdapter(vali_dataloader, 6, expand_dim=3)
        eval = model.evaluate(vali_dataloader_keras)
        accuracy = eval[1]
        return accuracy

    def search(self, train_dataloader, vali_dataloader, latency_limit, memory_limit, **kwargs):
        ctr = 0
        models = []
        while ctr < self.arch_tries:
            fake_input = torch.randn((1, 1, 128, 9))
            lat, mem = self.search_space(fake_input, inf_type=InferenceType.RANDOM)[1:]


            if latency_limit is not None and lat > latency_limit:
                continue
            if memory_limit is not None and mem > memory_limit:
                continue

            keras_model = self.search_space.getKeras(getPruned=True, inf_type=InferenceType.MAX_WEIGHT, batch_size=None)
            self._fit_keras(keras_model, train_dataloader)
            accuracy = self._eval_keras(keras_model, vali_dataloader)
            models.append({"model": keras_model, "accuracy": accuracy, "latency": lat, "memory": mem})
            ctr += 1
    
        maxModel = max(models, key=lambda x: x["accuracy"])
        print(maxModel)
        return models


    def extractArch():
        pass