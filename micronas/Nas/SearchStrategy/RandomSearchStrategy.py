from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType
from micronas.Utils.PytorchKerasAdapter import PytorchKerasAdapter
import torch


class RandomSearchStrategy():
    def __init__(self, search_space):
        self.search_space = search_space


    def _fit_keras(self, train_dataloader):
        train_dataloader_keras = PytorchKerasAdapter(train_dataloader, 6)
        next(iter(train_dataloader_keras))

    def search(self, train_dataloader, vali_dataloader, search_epochs, latency_limit, memory_limit, **kwargs):
      fake_input = torch.randn((1, 1, 128, 9))
      lat, mem = self.search_space(fake_input, inf_type=InferenceType.RANDOM)[1:]
      keras_model = self.search_space.getKeras(getPruned=True, inf_type=InferenceType.MAX_WEIGHT, batch_size=None)

      self._fit_keras(train_dataloader)

    def extractArch():
        pass