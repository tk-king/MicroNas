from abc import ABC, abstractmethod
from micronas.Nas.ModelResult import ModelResult
from typing import List

class SearchStrategy:
    def __init__(self, search_space):
        self.search_space = search_space
    
    @abstractmethod
    def search(train_dataloader, vali_dataloader, latency_limit, memory_limit, **kwargs) -> List[ModelResult]:
      pass
    
    def extractArch():
        pass