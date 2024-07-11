from abc import ABC, abstractmethod

class SearchStrategy:
    def __init__(self, search_space):
        self.search_space = search_space
    
    @abstractmethod
    def search(train_dataloader, vali_dataloader, latency_limit, memory_limit, **kwargs):
      pass
    
    def extractArch():
        pass