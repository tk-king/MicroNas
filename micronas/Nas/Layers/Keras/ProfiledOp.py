import tensorflow.keras as keras
from tensorflow.keras.layers import BatchNormalization

import json
from pathlib import Path
import os

import numpy as np


profiling = False
configs = []


class NetworkProfiler(object):
    def __init__(self, path, kerasShape):
        self._path = path
        self._kerasShape = kerasShape
    
    def __enter__(self):
        fake_input = np.random.randn(self._kerasShape)

        enableProfiling()

    def __exit__(self):
        saveProfiling(self._path)
        disableProfling()

def profileNetwork(network, path, kerasShape):
    enableProfiling()
    # cutShape = [x[1:] for x in kerasShape]
    keras_net = network.getKeras()
    saveProfiling(path)
    disableProfling()


def enableProfiling():
    global profiling
    profiling = True

def disableProfling():
    global profiling, configs
    profiling = False
    configs = []


def saveProfiling(path):
    jsonStr = json.dumps(configs, indent=4, sort_keys=True)
    folder, _ = os.path.split(path)
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(jsonStr)
    print(f"Saved {len(configs)} operations for profiling")


def profiledOp(base: keras.layers.Layer, *args, **kwargs):
    base_call = base.call
    def new_call(inputs, training=None, *args, **kwargs):
        if profiling:
            config = base.get_config()
            if isinstance(inputs, list): # For mul or add
                config["input_shape"] = [i.shape.as_list() for i in inputs]
            else:
                config["input_shape"] = inputs.shape.as_list()

            if isinstance(base, BatchNormalization):
                outputs = base_call(inputs, training=training)
            else:
                outputs = base_call(inputs)
            config["output_shape"] = outputs.shape.as_list()
            configs.append(config)

            return outputs
        else:
            return base_call(inputs)
    base.call = new_call

    return base