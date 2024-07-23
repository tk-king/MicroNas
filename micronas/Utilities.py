import numpy as np
from torch.autograd import Variable
import torch.nn as nn
# from prettytable import PrettyTable
from ast import literal_eval
from enum import Enum
import torch

import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from torch.utils.data import DataLoader
from collections import OrderedDict
from collections.abc import Mapping

import tensorflow as tf

from micronas.config import Config, MicroNasMCU


class RecorderError(Enum):
    NOERROR = 0
    RUNTIMEERROR = 1
    FLASHERROR = 2


class tflmBackend(Enum):
    STOCK = 1
    CMSIS = 0


class DotDict(OrderedDict):
    '''
    Quick and dirty implementation of a dot-able dict, which allows access and
    assignment via object properties rather than dict indexing.
    '''

    def __init__(self, *args, **kwargs):
        # we could just call super(DotDict, self).__init__(*args, **kwargs)
        # but that won't get us nested dotdict objects
        od = OrderedDict(*args, **kwargs)
        for key, val in od.items():
            if isinstance(val, Mapping):
                value = DotDict(val)
            else:
                value = val
            self[key] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as ex:
            raise AttributeError(f"No attribute called: {name}") from ex

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as ex:
            raise AttributeError(f"No attribute called: {k}") from ex

    __setattr__ = OrderedDict.__setitem__

def torch_to_one_channel(shape):
    shape = list(shape)
    shape[0] = 1
    return shape

def torch_to_keras_shape(shape):
    shape = list(shape)
    return [shape[0]] + shape[2:4] + [shape[1]] 

def cyclesToMillis(cycles, mcu=None):

    used_mcu = mcu if mcu is not None else Config.mcu
    freq = None
    if used_mcu == MicroNasMCU.NiclaSense:
        freq = 64000000
    if used_mcu == MicroNasMCU.NUCLEOL552ZEQ:
        freq = 80000000
    if used_mcu == MicroNasMCU.NUCLEOF446RE:
        freq = 180000000
    return (cycles / freq) * 1000  # in Millis


def computeLayerTimes(timingStr, mcu=None):
    timingStr = timingStr.replace("TIMING", "").replace(" ", "")
    timingArray = list(literal_eval(timingStr))
    print("t_arr: ", len(timingArray))
    def compute(timingArray):
        if len(timingArray) == 0:
            return []
        serachElm = timingArray[0]
        searchArr = timingArray[1:]
        ret = [serachElm[0], 0, 0]
        for elm in searchArr:
            if elm[0] == serachElm[0]:
                ret[1] = abs(serachElm[1] - elm[1])
                ret[2] = cyclesToMillis(abs(serachElm[1] - elm[1]), mcu)
                searchArr.remove(elm)
                return [ret] + compute(searchArr)
        print(f"Elm not found: {serachElm[0]}")
        return []
    return compute(timingArray)


def convertKerasToTfLite(model_keras, path="model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    data_bytes = converter.convert()
    with open(path, "wb") as f:
        f.write(data_bytes)
    return data_bytes


# def torchToKeras(torchModel, inputShape):
#     x = torch.randn((1, *inputShape), requires_grad=True)
#     torch.onnx.export(torchModel, x, "torch_onnx.onnx", verbose=True, input_names=[
#                       'input'], output_names=['output'])

#     onnx_model = onnx.load("./torch_onnx.onnx")  # load onnx model
#     tf_rep = prepare(onnx_model)  # prepare tf representation
#     tf_rep.export_graph("tfModel.pb")  # export the model

#     # This is how one would load the model
#     tf_model = tf.saved_model.load("tfModel.pb")
#     converter = tf.lite.TFLiteConverter.from_saved_model("tfModel.pb")
#     data_bytes = converter.convert()
#     with open("onnx_model.tflite", "wb") as f:
#         f.write(data_bytes)


# def torchToKeras2(torchModel, inputShape):
#     x = torch.randn((1, *inputShape), requires_grad=True)
#     torch.onnx.export(torchModel, x, "torchToOnnx.onnx", verbose=True, input_names=[
#                       'input'], output_names=['output'])

#     onnx_model = onnx.load("./torchToOnnx.onnx")  # load onnx model
#     tf_rep = prepare(onnx_model)  # prepare tf representation
#     tf_rep.export_graph("tfModel.pb")  # export the model

#     keras_model = tf.keras.models.load_model('tfModel.pb')
#     print(keras_model)