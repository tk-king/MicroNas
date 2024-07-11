from micronas.Nas.Networks.Pytorch.SearchModule import InferenceType

class RandomSearchStrategy():
    def __init__(self, search_space):
        self.search_space = search_space

    def search(self, train_dataloader, vali_dataloader, search_epochs, latency_limit, memory_limit, **kwargs):
      keras_net = self.search_space.getKeras(inf_type=InferenceType.RANDOM)
      print(keras_net)


    def extractArch():
        pass