from dataclasses import dataclass
from enum import Enum

class MicroNasMCU(Enum):
    NiclaSense = "NiclaSense"
    # NUCLEO = "NUCLEO"
    PORTENTA = "PROTENTA"
    NUCLEOF446RE  = "NUCLEOF446RE"


@dataclass
class DefaultConfig:
    train_epochs : int = 100
    retrain_epochs : int = 100
    ignore_latency : bool = False
    ignore_memory : bool = False

    batch_size : int = 64

    train_device : str = "cpu"

