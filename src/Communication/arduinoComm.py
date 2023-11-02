import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from src.Communication.arduinoCommunicator import ArduinoCommunicator, FINISH_FLAG, ERROR_IDENTIFIER
import struct

# from TfLite.structure import TfLiteModel

def floatArrayToBytes(self, arr):
    return struct.pack("f" * len(arr), *arr)

def sendModel(tfliteModel):
    cmd = struct.pack("B", 2)
    modelLen = struct.pack("<I", len(tfliteModel))
    print(cmd)
    print(modelLen, "==", len(tfliteModel))

def readOutput():
    lines = []
    comm = ArduinoCommunicator("COM3")
    line = comm.readLineString()
    error = False
    ctr = 0
    while line != FINISH_FLAG and line != ERROR_IDENTIFIER :
        lines.append(line)
        line = comm.serial.readline()
        try:
            line = line.decode("utf-8").strip()
        except:
            pass
        ctr += 1
    comm.onComplete()
    if (line == ERROR_IDENTIFIER):
        error = True
    return lines, error

