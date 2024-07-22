
import torch
from typing import TypedDict
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel

class Device(Enum):
    NICLA = "NICLA"
    NUCLEO = "NUCLEO"
    PORTENTA = "PROTENTA"
    NUCLEOF446RE  = "NUCLEOF446RE"

class ComputeUnit(Enum):
    CPU = "cpu"
    GPU = "cuda"

class MicroNasMCU(Enum):
    NiclaSense = "NiclaSense"
    # NUCLEO = "NUCLEO"
    PORTENTA = "PROTENTA"
    NUCLEOF446RE  = "NUCLEOF446RE"

@dataclass
class DefaultConfig():
    batch_size : int = 64
    compute_unit : str = "cpu"
    tensor_dtype : torch.dtype = torch.float32

    # SearchSpace Settings
    time_reduce_ch : int = 8
    time_reduce_granularity : int = 1

    ch_reduce_ch : int = 16
    ch_reduce_granularity : int = 1

    net_scale_factor : int = 0

    retrain_epochs : int = 200
    retrain_patience : int = 10

    dim_limit_time : int = 16
    dim_limit_ch :int = 5

    # ArchSearcher Settings
    search_epochs : int = 1
    # train_epochs : int = 100
    retrain_epochs : int = 100
    ignore_latency : bool = False
    ignore_memory : bool = False

    # MCU Settings
    port : str = "/dev/cu.usbmodem111403"
    mcu : MicroNasMCU = MicroNasMCU.NiclaSense



# class ConfigClass:
#     device = "cpu"

#     dataset = None
#     firmwarePath = "firmware"

#     time_reduce_ch = 8
#     time_reduce_granularity = 1

#     ch_reduce_ch = 16
#     ch_reduce_granularity = 1

#     net_scale_factor = 0

#     retrain_epochs = 200
#     retrain_patience = 10

#     dim_limit_time = 16
#     dim_limit_ch = 5

#     search_epochs = None
#     target_lat = None
#     target_mem = None
#     num_retraining = 5

#     eps = None

#     mcu = None
#     # port = "COM6" # COM3 for Nicla, COM6 for NUCLEO
#     port = "/dev/cu.usbmodem111403" 

Config = DefaultConfig()