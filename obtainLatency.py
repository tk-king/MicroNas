from src.config import Config, Device
Config.mcu = Device.NUCLEO.name
# Config.port = "/dev/cu.usbmodem0D0A24802"
Config.port = "/dev/cu.usbmodem111403"

from src.Profiler.LatMemProfiler import loadConfigs, _lookUp
import torch.nn as nn

print(len(_lookUp))

loadConfigs("profilings/002.json")