from src.Layers.DepthWiseCnnLayer import DepthWiseCnnLayer
from src.TfLite.structure import TfLiteModel
from src.Layers.DenseLayer import DenseLayer
from src.Layers.Conv2dLayer import Conv2Dlayer
from src.NeuralNetworks.CNN import CNN
import numpy as np
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# Configuration for the random cnn generator
MIN_CNN_LAYERS = 1
MAX_CNN_LAYERS = 10

MIN_CONV_KERNEL_SIZE = 1
MAX_CONV_KERNEL_SIZE = 8
MIN_CONV_NUM_KERNELS = 1
MAX_CONV_NUM_KERNELS = 16

MIN_DENSE_LAYERS = 1
MAX_DENSE_LAYERS = 4
MIN_DENSE_SIZE = 4
MAX_DENSE_SIZE = 32

RAM_DEFAULT = 20_000
FLASH_DEFAULT = 100_000

cnnRandomConfig_default = (MIN_CONV_KERNEL_SIZE, MAX_CONV_KERNEL_SIZE,
                           MIN_CONV_NUM_KERNELS, MAX_CONV_NUM_KERNELS)
denseRandomConfig_default = (MIN_DENSE_SIZE, MAX_DENSE_SIZE)


INPUT_SHAPE = (20, 20, 3)
NUM_CLASSES = 6


class RandomNetwork:

    def __init__(self, cnnRandomConfig=cnnRandomConfig_default, denseRandomConfig=denseRandomConfig_default, ram_constraint=RAM_DEFAULT, flash_constraint=FLASH_DEFAULT) -> None:
        self._cnnRandomConfig = cnnRandomConfig
        self._denseRandomConfig = denseRandomConfig
        self._ramConstraints = ram_constraint
        self._flashConstraints = flash_constraint

    # A CNN consists for several CNN and maxpooling layers
    # After the CNN layers, we use several (or one) dense layers for the final classification
    def generateRandomCnn(self, inputShape, num_classes, useDepthWiseCNN=False):

        cnnNetwork = CNN(inputShape, num_classes)

        num_cnn_layers = np.random.randint(
            low=MIN_CNN_LAYERS, high=MAX_CNN_LAYERS + 1)
        num_dense_layers = np.random.randint(
            low=MIN_DENSE_LAYERS, high=MAX_DENSE_LAYERS + 1)

        # Generate the first layer
        if useDepthWiseCNN:
            convLayer, previousShape = DepthWiseCnnLayer().configureRandom(self._cnnRandomConfig, inputShape)
            cnnNetwork.addLayer(convLayer)
            convLayer, previousShape = Conv2Dlayer().configureRandom1x1CNN(self._cnnRandomConfig, previousShape)
            cnnNetwork.addLayer(convLayer)
        else:
            convLayer, previousShape = Conv2Dlayer().configureRandom(self._cnnRandomConfig, inputShape)
            cnnNetwork.addLayer(convLayer)
        for _ in range(num_cnn_layers - 1):
            if useDepthWiseCNN:
                # DepthConv followed by 1x1 conv to controll output shape
                convLayer, previousShape = DepthWiseCnnLayer().configureRandom(self._cnnRandomConfig, previousShape)
                cnnNetwork.addLayer(convLayer)
                convLayer, previousShape = Conv2Dlayer().configureRandom1x1CNN(self._cnnRandomConfig, previousShape)
                cnnNetwork.addLayer(convLayer)
            else:
                convLayer, previousShape = Conv2Dlayer().configureRandom(self._cnnRandomConfig, previousShape)
                cnnNetwork.addLayer(convLayer)

        for _ in range(num_dense_layers - 1):
            denseLayer, previousShape = DenseLayer().configureRandom(
                self._denseRandomConfig, previousShape)
            cnnNetwork.addLayer(denseLayer)
        lastLayer = DenseLayer(num_classes)
        lastLayer.build(previousShape)
        cnnNetwork.addLayer(lastLayer)
        cnnNetwork.build()
        return cnnNetwork

    def generateRandom_Constraints(self, inputShape, num_classes, useDepthWiseCNN=False):
        cnn = self.generateRandomCnn(inputShape, num_classes, useDepthWiseCNN)
        ctr = 0
        while not self._checkRequirements(cnn, inputShape):
            cnn = self.generateRandomCnn(inputShape, num_classes, useDepthWiseCNN)
            ctr += 1
        return cnn

    def _checkRequirements(self, network, inputShape):
        tflmModel = TfLiteModel(network.toKerasModel(), inputShape)
        if self._flashConstraints is not None and tflmModel.flash_size > self._flashConstraints:
            return False
        if self._ramConstraints is not None and tflmModel.memory[1] > self._ramConstraints:
            return False
        return True