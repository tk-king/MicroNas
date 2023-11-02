
import itertools
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, DepthwiseConv2D
from enum import Enum
import tensorflow as tf
import numpy as np
from pathlib import Path
import tflite
import tflite.TensorType
from os.path import dirname
# from keras_flops import get_flops


class TfLiteModel():

    def __init__(self, kerasModel, inputShape, rep_dataset=None) -> None:
        self._rep_dataset = rep_dataset
        self._kerasModel = kerasModel
        self._inputShape = tuple(inputShape)
        if len(self._inputShape) == 4:
            self._inputShape = self._inputShape[1:]
        self.byte_model = self._toTfLiteModel()

        # Graph stuff
        self.graph_model = tflite.Model.GetRootAsModel(self.byte_model, 0)
        self.graph = self.graph_model.Subgraphs(0)
        self.inputs = self.graph.InputsAsNumpy()
        self.outputs = self.graph.OutputsAsNumpy()

        self.tensors = [TflmTensor(self.graph.Tensors(i), idx=i)
                        for i in range(self.graph.TensorsLength())]

        # Collect all the operations in the graph
        self.operations = []
        for idx in range(self.graph.OperatorsLength()):
            simpleOp = self.graph.Operators(idx)
            inputs = [self.tensors[i] for i in simpleOp.InputsAsNumpy()]
            outputs = [self.tensors[i] for i in simpleOp.OutputsAsNumpy()]
            tflmOp = TflmOp(self.graph_model.OperatorCodes(
                simpleOp.OpcodeIndex()).BuiltinCode(), idx, inputs, outputs)
            self.operations.append(tflmOp)

        # Check which tensors are stored in flash or ram
        # If a tensor was not prduced by an operation it is stored in flash
        for tensor in self.tensors:
            if tensor not in itertools.chain.from_iterable([op.outputs for op in self.operations]) and tensor.idx not in self.inputs:
                tensor.storage = TensorStorage.FLASH

    @property
    def memory(self):
        def _tensorUsage(tensor):
            tensorUsed_first = [
                op.idx for op in self.operations if tensor in op.outputs]
            tensorUsed_last = [
                op.idx for op in self.operations if tensor in op.inputs]
            firstUsed = min(tensorUsed_first) if tensorUsed_first else 0
            lastUsed = max(tensorUsed_last) if tensorUsed_last else len(
                self.tensors)
            return firstUsed, lastUsed

        def _between(num, tuple):
            return tuple[0] <= num <= tuple[1]

        def _calcOpMemory(op):
            return sum([tensor.space for tensor in self.tensors if _between(op.idx, _tensorUsage(tensor)) and tensor.storage is TensorStorage.RAM])

        peak_mem = max([_calcOpMemory(op) for op in self.operations])
        mem_sum = sum([_calcOpMemory(op) for op in self.operations])

        return peak_mem, mem_sum

    @property
    def flops(self):
        return sum([op.flops for op in self.operations])

    @property
    def keras_flops(self):
        return -1
        # return get_flops(self._kerasModel, batch_size=1)

    @property
    def flash_size(self):
        return len(self.byte_model)

    @property
    def flash_size_tensors(self):
        return sum([t.space for t in self.tensors])

    def saveModel(self, filePath):
        Path(dirname(filePath)).mkdir(parents=True, exist_ok=True)
        with open(filePath, "wb") as f:
            f.write(self.byte_model)

    def _toTfLiteModel(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self._kerasModel)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # if len(self._inputShape) > 1:
        if isinstance(self._inputShape[0], list):
            def representative_dataset():
                i_shape_001 = self._inputShape[0]
                i_shape_002 = self._inputShape[1]
                if len(self._inputShape[0]) == len(self._inputShape[1]):
                    i_shape_001 = self._inputShape[0][1:]
                    i_shape_002 = self._inputShape[1][1:]
                
                for _ in range(100):
                    data1, data2 = np.random.random(i_shape_001), np.random.random(i_shape_002)
                    yield [data1.astype(np.float32), data2.astype(np.float32)]

            converter.representative_dataset = representative_dataset
        else:
            converter.representative_dataset = \
                lambda: [
                    [np.random.random((1,) + self._inputShape).astype("float32")] for _ in range(5)]
            print(np.asarray(converter.representative_dataset()).shape)
        if self._rep_dataset is not None:
            itr = iter(self._rep_dataset)
            converter.representative_dataset = lambda: [[x[0]] for x in self._rep_dataset]
            print(np.asarray(converter.representative_dataset()).shape)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        byte_model = converter.convert()
        return byte_model

# This is used to represent tensors in the graph


class TensorStorage(Enum):
    FLASH = 0
    RAM = 1


class TflmTensor:

    def __init__(self, tensor: tflite.Tensor, idx, predecessors=[], successors=[], storage=TensorStorage.RAM) -> None:
        self.idx = idx  # Id in the tflite graph
        self.shape = tensor.ShapeAsNumpy()  # Shape of the tensor
        self.type = tensor.Type()  # Datatype of the tensor
        self.storage = storage  # Storage type: Flash or Ram
        self.name = tensor.Name().decode("ascii")

    def _storageScale(self):
        typeSizeMap = {
            tflite.TensorType.UINT8: 1,
            tflite.TensorType.INT8: 1,
            tflite.TensorType.INT16: 2,
            tflite.TensorType.INT32: 4,
            tflite.TensorType.FLOAT16: 2,
            tflite.TensorType.FLOAT32: 4,
            tflite.TensorType.FLOAT64: 8
        }
        return typeSizeMap[self.type]

    @property
    def space(self):
        return np.prod(self.shape) * self._storageScale()

    def __hash__(self) -> int:
        return self.idx

    def __repr__(self) -> str:
        return f'TFLMTensor: {self.name} ({self.idx}) :: shape: {self.shape}, type: {self.type}'


class TflmOp:
    def __init__(self, op, idx, inputs, output) -> None:
        self.op = op  # Buildin opcode in tflm
        self.inputs = inputs  # Inputs needed for this operation
        self.outputs = output  # Outputs generated for this operation
        self.idx = idx  # index in the tflm-file

    @property
    def get_dict(self):
        input_shapes = [x.shape for x in self.inputs]
        output_shapes = [x.shape for x in self.outputs]
        return {"op": self.op, "inputs": input_shapes, "outputs": output_shapes, "idx": self.idx}

    @property
    def flops(self):
        return computeFlops(self.op, self.inputs, self.outputs)

    def __hash__(self) -> int:
        return self.idx

    def __repr__(self) -> str:
        return f'TFLMOp: {self.op} :: idx: {self.idx}'


def computeFlops(op, inputs, outputs):
    flops = 0
    if op == tflite.BuiltinOperator.CONV_2D:
        input, kernel, bias = inputs
        [output] = outputs
        in_batch, in_X, in_Y, in_C = input.shape
        k_oC, k_X, k_Y, k_iC = kernel.shape
        out_batch, out_X, out_Y, out_C = output.shape
        flops += out_batch * out_X * out_Y * k_X * k_Y * k_iC * k_oC
        if bias is not None:
            flops += out_batch * out_X * out_Y * out_C
    elif op == tflite.BuiltinOperator.FULLY_CONNECTED:
        input, kernel, bias = inputs
        [output] = outputs
        in_batch, in_len = input.shape
        out_batch, out_len = output.shape
        flops += in_batch * in_len * out_len
        if bias is not None:
            flops += out_batch * out_len
    elif op == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
        input, kernel, bias = inputs
        [output] = outputs
        k_oC, k_X, k_Y, k_iC = kernel.shape
        out_batch, out_X, out_Y, out_C = output.shape
        flops += out_batch * k_iC * out_X * out_Y * k_X * k_Y
        if bias is not None:
            flops += out_batch * out_X * out_Y * out_C
    else:
        pass

    return 2 * flops
