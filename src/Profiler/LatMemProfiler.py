import json
import os

from torch import conv2d, multiply
from src.Utilities import DotDict, tflmBackend, torch_to_keras_shape, torch_to_one_channel, cyclesToMillis

from src.Recorder.recorder import Recorder, RecorderError

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, GlobalAveragePooling2D, Softmax, BatchNormalization, Add, Multiply
from tensorflow.keras import Sequential
from tqdm import tqdm
import copy
import numpy as np

import torch
import torch.nn as nn
from src.config import Config, Device


def build_model_from_config(config):
    cfg = json.loads(json.dumps(config))
    input_shape = cfg["input_shape"]
    output_shape = cfg["output_shape"]
    cfg.pop("input_shape", None)
    cfg.pop("output_shape", None)
    layer = None
    l_name = cfg["name"]
    if "conv2d" in l_name:
        layer = Conv2D
    if "max_pooling2d" in l_name:
        layer = MaxPool2D
    if "global_average_pooling2d" in l_name:
        layer = GlobalAveragePooling2D
    if "dense" in l_name:
        layer = Dense
    if "softmax" in l_name:
        layer = Softmax
    if "batch_normalization" in l_name:
        layer = BatchNormalization
    if "add" in l_name:
        layer = Add()
    if "multiply" in l_name:
        layer = Multiply()

    if layer is None:
        print("Not found: ", l_name)
        raise NotImplementedError()
    layer = layer.from_config(cfg)

    if "multiply" in l_name or "add" in l_name:
        input_001 = Input(shape=input_shape[0][1:])
        input_002 = Input(shape=input_shape[1][1:])
        out = layer([input_001, input_002])
        model = Model(inputs=[input_001, input_002], outputs=out)
    else:
        model = Sequential()
        model.add(layer)
        model.build(input_shape=input_shape)

    return model, input_shape, output_shape


# old add / multiply : "tmp_look_up/20220405-211942/records.json"
# paths = ["tmp_look_up/20220402-221302/records.json", "tmp_look_up/20220402-231313/records.json",
#          "tmp_look_up/20220404-202744/records.json", "tmp_look_up/20220405-162714/records.json",
#          "tmp_look_up/20220405-182820/records.json", "tmp_look_up/20220405-233132/records.json",
#          "tmp_look_up/20220406-002132/records.json"]

paths = []
# nicla_paths = ["tmp_look_up/20220412-121323/records.json", 
#         "tmp_look_up/20220419-205859/records.json", 
#         "tmp_look_up/20220420-085100/records.json",
#         "tmp_look_up/20220420-100433/records.json",
#         "tmp_look_up/20220424-101405/records.json",
#         "tmp_look_up/20220426-172434/records.json",
#         "tmp_look_up/20220426-174546/records.json",
#         "tmp_look_up/20220426-193145/records.json",
#         "tmp_look_up/20220502-184252/records.json",
#         "tmp_look_up/20220502-204114/records.json",
#         "tmp_look_up/20220503-124816/records.json",
#         "tmp_look_up/20220503-140405/records.json",
#         "tmp_look_up/20220503-144814/records.json",
#         "tmp_look_up/20220503-155051/records.json",
#         "tmp_look_up/20220503-165122/records.json",
#         "tmp_look_up/20220505-104207/records.json",
#         "tmp_look_up/20220509-160658/records.json",
#         "tmp_look_up/20220516-111720/records.json",
#         "tmp_look_up/20220516-163316/records.json",
#         "tmp_look_up/20220516-165212/records.json",
#         "tmp_look_up/20220522-090242/records.json",
#         "tmp_look_up/20220522-091140/records.json",
#         "tmp_look_up/20220522-101156/records.json",
#         "tmp_look_up/20220524-161205/records.json",
#         "tmp_look_up/20220524-174821/records.json",
#         "tmp_look_up/20220528-094744/records.json",
#         "tmp_look_up/20220528-142405/records.json",
#         "tmp_look_up/20220528-151308/records.json",
#         "tmp_look_up/20220530-084845/records.json",
#         "tmp_look_up/20220530-114636/records.json",
#         "tmp_look_up/20220530-140859/records.json",
#         "tmp_look_up/20220530-141352/records.json",
#         "tmp_look_up/20220530-154456/records.json",
#         "tmp_look_up/20220601-092546/records.json"]

print(f"Profiler set for device {Config.mcu}")
paths = []
for subdir, dirs, files in os.walk(f"lookUp/{Config.mcu}"):
    for file in files:
        print("Loading file: ", file)
        f_path = os.path.join(subdir, file)
        print(f_path)
        if f_path.endswith(".json"):
            paths.append(f_path)
print(paths)

_lookUp = {}

ignoreLat = False
ignoreMem = False


def set_ignore_latency(ignoreLatency=False, ignoreMemory=False):
    global ignoreLat
    global ignoreMem
    ignoreLat = ignoreLatency
    ignoreMem = ignoreMemory


def buildKey(rec):
    l_name = rec["modelConfig"]["name"].lower()
    key = None
    if "separableconv2d" in l_name:
        print("choose sepconv2d")
        key = f'name: separableConv2D, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape} filter: {rec.modelConfig.kernel_size} padding: {rec.modelConfig.padding} strides: {rec.modelConfig.strides} activation: {rec.modelConfig.activation}'
    if "conv2d" in l_name:
        key = f'name: conv2d, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape} filter: {rec.modelConfig.kernel_size} padding: {rec.modelConfig.padding} strides: {rec.modelConfig.strides} activation: {rec.modelConfig.activation}'
    if "max_pooling2d" in l_name or "maxpool2d" in l_name:
        key = f'name: maxpool2d, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape} filter: {rec.modelConfig.pool_size}'
    if "global_average_pooling2d" in l_name:
        key = f'name: gap2d, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape}'
    if "dense" in l_name:
        key = f'name: dense, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape}'
    if "softmax" in l_name:
        key = f'name: softmax, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape}'
    if "batch_normalization" in l_name or "batchnorm2d" in l_name:
        key = f'name: batchNorm, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape}'
    if "multiply" in l_name:
        key = f'name: multiply, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape}'
    if "add" in l_name:
        key = f'name: add, input: {rec.modelConfig.input_shape} output: {rec.modelConfig.output_shape}'

    if key is None:
        print("Not found: ", l_name)
        raise NotImplementedError()
    return key


for p in paths:
    with open(p, "r") as f:
        estimates = json.loads(f.read())
        estimates = [DotDict(x) for x in estimates]
    for est in estimates:
        layer_name = est["modelConfig"]["name"]
        if hasattr(est, "error") and est.error != RecorderError.NOERROR.name:
            _lookUp[buildKey(est)] = 50000000000;
            continue;
        if "batch_normalization" in layer_name:
            _lookUp[buildKey(est)] = est.timing[2]["cpuCycles"] + \
                est.timing[3]["cpuCycles"]
            continue
        if len(est.timing) == 5:
            _lookUp[buildKey(est)] = sum([l["cpuCycles"]
                                          for l in est.timing[-3:]])
            continue
        if len(est.timing) == 4 and "dense" in layer_name:
            _lookUp[buildKey(est)] = sum([l["cpuCycles"]
                                          for l in est.timing[-2:]])
            continue
        if len(est.timing) == 4 and "conv2d" in layer_name:
            _lookUp[buildKey(est)] = sum([l["cpuCycles"]
                                          for l in est.timing[-2:]])
            continue
        if len(est.timing) != 3:
            print("not included: ", layer_name)
            continue

        _lookUp[buildKey(est)] = est.timing[2]["cpuCycles"]

print(f"Loaded LatencyPredictor with {len(_lookUp)} samples")


def calcMemory(type, input_shape, output_shape, kernel_shape=None, only_outputs=False):
    if isinstance(input_shape[0], list):
        in_size = np.sum([np.prod(i) for i in input_shape])
    else:
        in_size = np.prod(input_shape)
    out_size = np.prod(output_shape)

    mem = 0
    if type in ["conv2d", "dyn_conv2d"]:
        mem += 4 * np.prod(kernel_shape) * input_shape[-1]
    if type.lower() in ["separableconv2d"]:
        intermediate_size = np.prod(output_shape[:-1]) * input_shape[-1]
        depth_conv = in_size + intermediate_size + (4 * np.prod(kernel_shape) + 4)
        point_conv = intermediate_size + out_size
        mem = max(depth_conv, point_conv) if not only_outputs else point_conv
        return torch.tensor(mem, dtype=float)
        

    # print("type: ", type, "input_shape: ", input_shape, "output_shape: ",
    #       output_shape, "arm: ", mem, "kernel_size: ", kernel_shape, "in_s: ", in_size, "out_s: ", out_size, "mem: ", mem + in_size + out_size)
    if only_outputs:
        return torch.tensor(mem + out_size, dtype=float)
    return torch.tensor(mem + in_size + out_size, dtype=float)

tmp = []

def lookup_torch(layer, input_shape, output_shape, input_shape_002=None, only_outputs=False, options={}):
    input_shape = torch_to_keras_shape(torch_to_one_channel(input_shape))
    output_shape = torch_to_keras_shape(torch_to_one_channel(output_shape))

    if type(layer) is str:
        kernel_size = None
        layer_name = layer
        if layer == "add" or layer == "multiply":
            if input_shape_002:
                input_shape_002 = torch_to_keras_shape(
                    torch_to_one_channel(input_shape_002))
            else:
                input_shape_002 = input_shape
            if input_shape_002[1] == 1 and input_shape_002[2] == 1:
                input_shape_002 = [input_shape_002[0], input_shape_002[3]]
            input_shape = [input_shape_002, input_shape]
            # input_shape = [input_shape, input_shape_002] # if len(input_shape) <= len(input_shape_002) and input_shape[] else [input_shape_002, input_shape]
    else:
        kernel_size = list(layer.kernel_size) if hasattr(layer, "kernel_size") else None

        layer_name = layer._get_name().lower()
    layer_dict = {
        "modelConfig": {
            "name": layer_name,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "kernel_size": kernel_size,
            "pool_size": kernel_size,
            **options
        }
    }

    res_lat= torch.tensor(0)
    res_mem = torch.tensor(0)

    if not ignoreLat:
        try:
            key = buildKey(DotDict(layer_dict))
            val = _lookUp[key]
        except:
            layer_dict["modelConfig"]["input_shape"] = list(reversed(layer_dict["modelConfig"]["input_shape"]))
            key = buildKey(DotDict(layer_dict))
            val = _lookUp[key]


        tmp.append([key, ": ", cyclesToMillis(val)])
        
        res_lat = torch.tensor(cyclesToMillis(_lookUp[buildKey(DotDict(layer_dict))]))

    if not ignoreMem:
        res_mem = calcMemory(layer_name, layer_dict["modelConfig"]["input_shape"], layer_dict["modelConfig"]["output_shape"], layer_dict["modelConfig"]["kernel_size"], only_outputs=only_outputs)

    return res_lat.to(Config.device), res_mem.to(Config.device)

def lookUp_keras(layer, input_shape, output_shape):
    pass


def loadConfigs(path):
    rec = Recorder("lookUp/" + Config.mcu, "firmware", tflmBackend.CMSIS)
    with open(path, "r") as f:
        configs = json.loads(f.read())
        configs = [DotDict(x) for x in configs]
    print("Number of configs: ", len(configs))
    keys_profiled = []
    for cfg in tqdm(configs):
        # Check if this operation has already been profiled
        key = buildKey(DotDict({"modelConfig": cfg}))
        if key in _lookUp or key in keys_profiled:
            continue
        print(key)
        model, _, _ = build_model_from_config(cfg)
        rec.recordLookUp(model, cfg)
        keys_profiled.append(key)
    rec.finalize()


def printDict():
    for k, v in _lookUp.items():
        print(f"{k}: {v}")
