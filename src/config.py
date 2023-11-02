
import torch
from enum import Enum
from dataclasses import dataclass

class Device(Enum):
    NICLA = "NICLA"
    NUCLEO = "NUCLEO"
    PORTENTA = "PROTENTA"
    NUCLEOF446RE  = "NUCLEOF446RE"

class ConfigClass:
    device = "cpu"

    dataset = None
    firmwarePath = "firmware"

    time_reduce_ch = 16
    time_reduce_granularity = 4

    ch_reduce_ch = 64
    ch_reduce_granularity = 8

    net_scale_factor = 2

    retrain_epochs = 200
    retrain_patience = 10

    dim_limit_time = 16
    dim_limit_ch = 5

    search_epochs = None
    target_lat = None
    target_mem = None
    num_retraining = 5

    eps = None

    mcu = None
    # port = "COM6" # COM3 for Nicla, COM6 for NUCLEO
    port = "/dev/cu.usbmodem111403" 

Config = ConfigClass()