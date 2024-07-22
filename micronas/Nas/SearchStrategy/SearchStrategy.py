from abc import ABC, abstractmethod
from micronas.Nas.ModelResult import ModelResult
from typing import List
import logging

class SearchStrategy:
    def __init__(self, search_space):
        self.search_space = search_space
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def search(latency_limit, memory_limit, **kwargs) -> List[ModelResult]:
      pass
    
    def extractArch():
        pass
    
    def compile(self, ts_len, num_sensors, num_classes, train_dataloader, vali_dataloader, test_dataloader):
        self.ts_len = ts_len
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.train_dataloader = train_dataloader
        self.vali_dataloader = vali_dataloader
        self.test_dataloader = test_dataloader
        self.logger.info(f"ts_len: {ts_len}, num_sensors: {num_sensors}, num_classes: {num_classes}")