from InMemoryDataset import InMemoryDataset, DiskData
from micronas import MicroNas, MicroNasMCU
from micronas.Nas.Networks.Pytorch.SearchNet import SearchNet
from micronas.Nas.SearchStrategy.DnasStrategy import DNasStrategy
from micronas.Nas.SearchStrategy.RandomSearchStrategy import RandomSearchStrategy
from micronas.Nas.SearchStrategy.DnasStrategy import DNasStrategy
import logging
from ColorLoggerFormatter import ColorLoggerFormatter
from micronas.config import Config

def micro_nas_search(dataset, cv, search_strategy, mcu, target_lat, target_mem):
    # Configure logging
    if search_strategy not in ["random", "dnas"]:
        raise ValueError("Invalid search strategy")
    
    print(mcu)
    print(MicroNasMCU.__members__)
    if mcu not in MicroNasMCU.__members__:
        raise ValueError("Invalid MCU")
    

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("main")
    ch = logging.StreamHandler()
    ch.setFormatter(ColorLoggerFormatter())
    logger.addHandler(logging.StreamHandler())

    diskData = DiskData.load(dataset, cv)
    train_dataloader = diskData.train_dataloader
    test_dataloader = diskData.test_dataloader
    vali_dataloader = diskData.vali_dataloader

    num_classes_lookup = {
        "skodar": 10
    }

    NUM_CLASSES = num_classes_lookup[dataset]

    logger.info("Length of train_dataloader: %s", len(train_dataloader))
    logger.info("Length of test_dataloader: %s", len(test_dataloader))
    logger.info("Length of vali_dataloader: %s", len(vali_dataloader))

    model = MicroNas(train_dataloader, vali_dataloader, test_dataloader, NUM_CLASSES)
    search_space = SearchNet()
    if search_strategy == "dnas":
        search_strategy = DNasStrategy(search_space)
    if search_strategy == "random":
        search_strategy = RandomSearchStrategy(search_space, arch_tries=1)
    # search_strategy = DNasStrategy(search_space)

    model.compile(search_space, search_strategy)

    models = model.fit(MicroNasMCU.NUCLEOF446RE, latency_limit=target_lat, memory_limit=target_mem, search_epochs=Config.search_epochs, retrain_epochs=Config.retrain_epochs, compute_unit="cpu")
    return models

if __name__ == '__main__':
    micro_nas_search("skodar", 0, "random", "NUCLEOF446RE", None, None)
