from src.config import Config
Config.mcu = "NUCLEO"
Config.port = "/dev/cu.usbmodem111403"

from src.Profiler.LatMemProfiler import loadConfigs, _lookUp
import torch.nn as nn
print(len(_lookUp))

loadConfigs("profilings/002.json")