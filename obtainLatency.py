from micronas.config import Config, Device
Config.mcu = Device.NUCLEO.name
Config.port = "/dev/cu.usbmodem111403"

from micronas.Profiler.LatMemProfiler import loadConfigs, _lookUp
import torch.nn as nn

print(len(_lookUp))

loadConfigs("profilings/002.json")