from distutils.command.config import config
import numpy

from sklearn import neural_network
from src.Communication.flasher import flashNicla, configureFirmware, readOutput, tflmBackend
from src.TfLite.structure import TfLiteModel
from src.NeuralNetworks import BaseNetwork
from src.Utilities import RecorderError, computeLayerTimes
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import re
import json
from os.path import join as joinPaths
from pathlib import Path
from datetime import datetime
import random

from src.config import Config


class BoardRecorder():
    def __init__(self, path, board_id) -> None:
        self._board_id = board_id
        self._path = path

        Path(joinPaths(self._path, self._board_id)).mkdir(parents=True, exist_ok=True)

        # Dict to store all the data
        self._data = {"latency": {}, "memory": {}}
    
    def addLatency(self, layerName, coeff):
        self._data["latency"][layerName] = coeff


    def addMemory(self, layerName, coeff):
        self._data ["memory"][layerName] = coeff

    def save(self):
        pass

class Recorder:
    def __init__(self, save_path, firmware_path, tflm_backend : tflmBackend, name_extension="", lookup=False) -> None:
        self._save_path = save_path
        self._firmware_path = firmware_path
        self._tflm_backend = tflm_backend
        self._createTime = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._name_extension = "_" + name_extension if len(name_extension) > 0 else name_extension
        self._save_folder_name = joinPaths(save_path, self._createTime + self._name_extension)

        # Create folder to store the data
        if not lookup:
            Path(joinPaths(self._save_folder_name, "keras")).mkdir(parents=True, exist_ok=True)

        # Vars to keep the data during recording
        self._data_dict = []

    def recordLookUp(self, keras_model, config):
        tmp_record = {}
        tflmModel = TfLiteModel(keras_model, config["input_shape"])
        with open("models/latestRecorderModel.tflite", "wb") as f:
            f.write(tflmModel.byte_model)
        configureFirmware(tflmModel.byte_model, self._firmware_path, Config.mcu, self._tflm_backend)
        flashOutput, error = flashNicla(self._firmware_path)

        # Only read the output if no error has occured
        if error == RecorderError.NOERROR:
            execOutput, error, errorText = readOutput()
            if (error != RecorderError.NOERROR):
                tmp_record["errorText"] = errorText

        print(error.name)
        if error == RecorderError.FLASHERROR:
            print(flashOutput)
        tmp_record["modelConfig"] = config
        tmp_record["tflmBackend"] = self._tflm_backend.name
        tmp_record["flashSize"] = tflmModel.flash_size
        tmp_record["memoryEstimate"] = tflmModel.memory
        tmp_record["flopsEstimate"] = tflmModel.flops

        if error is not RecorderError.NOERROR:
            tmp_record["error"] = error.name
        if error != RecorderError.NOERROR:
            self._data_dict.append(tmp_record)
            self._writeJsonFile()
            return


        for line in execOutput:
            if not isinstance(line, str):
                continue
            if line.startswith("TIMING"):
                tmp_record["timing"] = self._processTimingInformation(line)
            if line.startswith("MEMORY"):
                tmp_record["memory"] = self._processMemoryInformation(line)
        self._data_dict.append(tmp_record)
        self._writeJsonFile()


    def recordModel(self, neural_network: BaseNetwork, inputShape):
        tmp_record = {}
        model_random_data = random.getrandbits(64)
        keras_model = neural_network.toKerasModel()
        tflmModel = TfLiteModel(keras_model, inputShape)
        with open("models/latestRecorderModel.tflite", "wb") as f:
            f.write(tflmModel.byte_model)
        configureFirmware(tflmModel.byte_model, self._firmware_path, self._tflm_backend)
        flashOutput, error = flashNicla(self._firmware_path)
        if error == RecorderError.FLASHERROR:
            print(flashOutput)
            
        # Only read the output if no error has occured
        if error == RecorderError.NOERROR:
            execOutput, error, errorText = readOutput()
            if (error != RecorderError.NOERROR):
                tmp_record["errorText"] = errorText

        print(error.name)

        # If no error has occured, parse and record the data comming from the MCU
        tmp_record["modelConfig"] = neural_network.config
        tmp_record["tflmBackend"] = self._tflm_backend.name
        tmp_record["flashSize"] = tflmModel.flash_size
        tmp_record["memoryEstimate"] = tflmModel.memory
        tmp_record["flopsEstimate"] = tflmModel.flops
        tmp_record["assets"] = model_random_data # Use random string to associate assets with a record
        
        # Save the assets to the disk
        keras_model.save(joinPaths(self._save_folder_name, "keras", f'kerasModel_{model_random_data}.h5'))
        tflmModel.saveModel(joinPaths(self._save_folder_name, "tflite", f'tfliteModel_{model_random_data}.tflite'))

        if error is not RecorderError.NOERROR:
            tmp_record["error"] = error.name

        for line in execOutput:
            if not isinstance(line, str):
                continue
            if line.startswith("TIMING"):
                tmp_record["timing"] = self._processTimingInformation(line)
            if line.startswith("MEMORY"):
                tmp_record["memory"] = self._processMemoryInformation(line)
        self._data_dict.append(tmp_record)
        self._writeJsonFile()


    def _writeJsonFile(self):
        def convert(o):
            if isinstance(o, numpy.int64): return int(o)  
            if isinstance(o, numpy.int32): return int(o)
            if isinstance(o, numpy.int): return int(o)
            if isinstance(o, int): return int(o)
            return o
        
        with open(joinPaths(self._save_folder_name, "records.json"), 'w') as fp:
            json.dump(self._data_dict, fp, indent=4, sort_keys=True, default=convert)

    def finalize(self):
        self._writeJsonFile()
        self.data_dict = []


    def _processTimingInformation(self, line):
        timings = computeLayerTimes(line)
        tmpTimingDict = []
        for [type, cpuCycles, millis] in timings:
            tmpTimingDict.append({"type": type, "cpuCycles": cpuCycles, "millis": millis})
        return tmpTimingDict

    def _processMemoryInformation(self, line):
        memory = re.findall("\(.*?\)", line)
        tmpMemDict = {}
        for elm in memory:
            info, value = elm.replace("(", "").replace(")", "").split(",")
            tmpMemDict[info] = value
        return tmpMemDict
