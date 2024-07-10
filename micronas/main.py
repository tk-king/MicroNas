from torch.utils.data import DataLoader, Dataset
from micronas.defaultConfig import DefaultConfig, MicroNasMCU
from micronas.Nas.Networks.Pytorch.SearchNet import SearchNet
from micronas.Nas.Search import ArchSearcher
from multipledispatch import dispatch

class MicroNas:
    def __init__(self, *args):
        if len(args) == 3:
            self.__init_two_datasets(*args)
        elif len(args) == 4:
            self.__init_three_datasets(*args)
        else:
            raise ValueError("Invalid number of arguments for MicroNas initialization")

    @dispatch(Dataset, Dataset, int)
    def __init_two_datasets(self, train_dataset: Dataset, test_dataset: Dataset, num_classes: int) -> None:
        self.train_dataset = train_dataset
        self.vali_dataset = None
        self.test_dataset = test_dataset
        self.num_classes = num_classes

    @dispatch(Dataset, Dataset, Dataset, int)
    def __init_three_datasets(self, train_dataset: Dataset, vali_dataset: Dataset, test_dataset: Dataset, num_classes: int) -> None:
        self.train_dataset = train_dataset
        self.vali_dataset = vali_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes

    def search(self, target_mcu : MicroNasMCU, latency_limit : int , memory_limit : int, **kwargs):
        config : DefaultConfig = DefaultConfig(**kwargs)
        batch_size = config.batch_size

        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        print(len(next(iter(train_dataloader))))
        
        vali_dataloader = DataLoader(self.vali_dataset, batch_size=batch_size, shuffle=True) if self.vali_dataset else None
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False) if self.test_dataset else None

        dataset_shape = next(iter(train_dataloader))[0].shape
        ts_len, num_sensors = dataset_shape[1:3]
        nas_net = SearchNet([ts_len, num_sensors], self.num_classes).to(config.train_device)
        searcher = ArchSearcher(nas_net)
        searcher.train(train_dataloader, vali_dataloader, config.train_epochs, latency_limit, memory_limit)
