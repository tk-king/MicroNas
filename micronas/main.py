from torch.utils.data import DataLoader, Dataset
from micronas.config import Config, MicroNasMCU, DefaultConfig
from micronas.Nas.Networks.Pytorch.SearchNet import SearchNet
from micronas.Profiler.LatMemProfiler import set_ignore_latency
from micronas.Nas.Search import ArchSearcher
from multipledispatch import dispatch
import logging

class MicroNas:
    def __init__(self, *args):
        if len(args) == 3:
            self.__init_two_datasets(*args)
        elif len(args) == 4:
            self.__init_three_datasets(*args)
        else:
            raise ValueError("Invalid number of arguments for MicroNas initialization")
        self.logger = logging.getLogger(__name__)


    @dispatch(Dataset, Dataset, int)
    def __init_two_datasets(self, train_dataset: Dataset, test_dataset: Dataset, num_classes: int) -> None:
        self.train_dataset = train_dataset
        self.vali_dataset = None
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.search_net = None

    @dispatch(Dataset, Dataset, Dataset, int)
    def __init_three_datasets(self, train_dataset: Dataset, vali_dataset: Dataset, test_dataset: Dataset, num_classes: int) -> None:
        self.train_dataset = train_dataset
        self.vali_dataset = vali_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.search_net = None

    def compile(self, search_space, search_strategy):

        ts_len, num_sensors = self.train_dataset[0][0].shape[1:3]
        self.logger.info(f"ts_len: {ts_len}, num_sensors: {num_sensors}")

        train_dataloader = DataLoader(self.train_dataset, batch_size=Config.batch_size, shuffle=True)
        vali_dataloader = DataLoader(self.vali_dataset, batch_size=Config.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=Config.batch_size, shuffle=True)

        self.logger.info(f"Compiling with, ts_len: {ts_len}, num_sensors: {num_sensors}, num_classes: {self.num_classes}")
        self.search_strategy = search_strategy
        self.search_strategy.compile(ts_len, num_sensors, self.num_classes, train_dataloader, vali_dataloader, test_dataloader)
        self.search_sapce = search_space
        self.search_sapce.compile(ts_len, num_sensors, self.num_classes)    



    def fit(self, target_mcu : MicroNasMCU, latency_limit, memory_limit : int, **kwargs : DefaultConfig):

        Config.mcu = target_mcu
        for key, value in kwargs.items():
            if not hasattr(Config, key):
                raise ValueError(f"Invalid configuration parameter: {key}")
            setattr(Config, key, value)

        set_ignore_latency(Config.ignore_latency)


        # test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False) if self.test_dataset else None
        # ts_len, num_sensors = dataset_shape[1:3]
        # self.search_net = SearchNet([ts_len, num_sensors], self.num_classes).to(Config.compute_unit)
        # searcher = ArchSearcher(self.search_net)
        search_res = self.search_strategy.search(latency_limit, memory_limit)
        return search_res

    def save(self, path: str, name: str) -> None:
        pass
