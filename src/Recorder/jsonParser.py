import json
from ntpath import join
import numpy as np
from os.path import join as joinPaths
from typing import List
import multiprocessing
from src.Utilities import tflmBackend

from src.TfLite.structure import TfLiteModel

from tensorflow import keras as keras
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Flatten

op2Name = {
    3: "Conv2D",
    9: "Dense"
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Timing:
    def __init__(self, record) -> None:
        self._timings = record["timing"]
        self._modelConfig = record["modelConfig"]

    @property
    def wholeModel(self):
        return dotdict(next(x for x in self._timings if x["type"] == 255))

    def layer(self, layer_id):
        layerName = op2Name[layer_id]
        return dotdict(next(x for x in self._modelConfig if x["name"] == layerName)), dotdict(next(x for x in self._timings if x["type"] == layer_id))


class JsonParser:

    def __init__(self, filePath, filterErrors=True) -> None:
        self._data = self._extractJson(filePath, filterErrors)
        self._filterErrors = filterErrors
        self._filePaths = [filePath] * len(self._data)

    def filterBackend(self, backend: tflmBackend):
        print("Sum: ", sum([1 for x in self._data if x["tflmBackend"] == backend.name]))

    def _extractJson(self, filePath, filterErrors):
        recordPath = joinPaths(filePath, "records.json").replace("\\", "/")
        with open(recordPath, 'r') as f:
            jsonData = json.load(f)
            return [x for x in jsonData if "error" not in x and filterErrors]

    def addJson(self, filePath):
        newData = self._extractJson(filePath, self._filterErrors)
        self._data = [*self._data, *newData]
        self._filePaths.extend([filePath] * len(newData))

    @ property
    def tensorAreaUsed(self):
        return [x["memory"]["kTensorAreaUsed"] for x in self._data]

    @ property
    def memoryEstimate(self):
        memoryEst = [x["memoryEstimate"] for x in self._data]
        est = list(map(list, zip(*memoryEst)))
        return est[0], est[1]

    @ property
    def timings(self):
        return [Timing(x) for x in self._data]

    @ property
    def flops(self):
        return np.asarray([int(x["flopsEstimate"]) for x in self._data])

    @ property
    def numRecords(self):
        return len(self._data)

    @ property
    def rnnInput(self):
        maxLen = 0
        input_data = []
        for record in self._data:
            layer_data = []
            for layer in record["modelConfig"]:
                if layer["name"] == "Conv2D":
                    inputX, inputY, inputC = layer["input"]
                    filterX, filterY, filterC = layer["filter"]
                    outputX, outputY, outputC = layer["filter"]
                    conv2dData = np.asarray(
                        [0, inputX, inputY, inputC, filterX, filterY, filterC])
                    maxLen = max(maxLen, len(conv2dData))
                    layer_data.append(conv2dData)  # 0 for Conv2D
                elif layer["name"] == "Dense":
                    num_input = layer["input"]
                    num_output = layer["output"]
                    denseData = np.asarray([1, num_input, num_output])
                    maxLen = max(maxLen, len(denseData))
                    layer_data.append(denseData)  # 1 for Dense
            input_data.append(np.asarray(layer_data))

        print(len(input_data))
        # Now pad the input data
        for record in input_data:
            for layer in record:
                layer = np.pad(layer, (0, maxLen - len(layer)),
                               mode='constant')

        return input_data

    # This operation takes quite some time
    def toTfLite(self) -> List[TfLiteModel]:
        tflite = []
        for record, path in zip(self._data, self._filePaths):
            asset = record["assets"]
            inputShape = tuple(record["modelConfig"][0]["input"])
            kerasFileName = f'kerasModel_{asset}.h5'
            filePath = joinPaths(
                path, "keras", kerasFileName).replace("\\", "/")
            kerasModel = keras.models.load_model(filePath)
            tflite.append(TfLiteModel(kerasModel, inputShape))

        return tflite

    def toKeras(self):
        kerasModels = []
        for record, path in zip(self._data, self._filePaths):
            asset = record["assets"]
            kerasFileName = f'kerasModel_{asset}.h5'
            filePath = joinPaths(
                path, "keras", kerasFileName).replace("\\", "/")
            kerasModels.append(keras.models.load_model(filePath))
        return kerasModels

    def getRecord(self, index):
        return self._data[index], self._filePaths[index]
