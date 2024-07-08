from torch.utils.data import DataLoader, Dataset
from micronas.defaultConfig import DefaultConfig, MicroNasMCU
from multipledispatch import dispatch

class MicroNas:
    def __init__(self, *args):
        if len(args) == 2:
            self.__init_two_datasets(*args)
        elif len(args) == 3:
            self.__init_three_datasets(*args)
        else:
            raise ValueError("Invalid number of arguments for MicroNas initialization")

    @dispatch(Dataset, Dataset)
    def __init_two_datasets(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        self.train_dataset = train_dataset
        self.vali_dataset = None
        self.test_dataset = test_dataset

    @dispatch(Dataset, Dataset, Dataset)
    def __init_three_datasets(self, train_dataset: Dataset, vali_dataset: Dataset, test_dataset: Dataset) -> None:
        self.train_dataset = train_dataset
        self.vali_dataset = vali_dataset
        self.test_dataset = test_dataset

    def search(self, target_mcu : MicroNasMCU, latency_limit : int , memory_limit : int, **kwargs):
        config = DefaultConfig(**kwargs)
        batch_size = config.batch_size

        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        vali_dataloader = DataLoader(self.vali_dataset, batch_size=batch_size, shuffle=True) if self.vali_dataset else None
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False) if self.test_dataset else None



