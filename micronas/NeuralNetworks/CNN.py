from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten

from micronas.NeuralNetworks.BaseNetwork import BaseNetwork
from micronas.Layers.BaseLayer import Layer


class CNN(BaseNetwork):

    def __init__(self, inputShape, numClasses):
        super().__init__(inputShape, numClasses)

    def toKerasModel(self) -> Sequential:
        model = Sequential()

        # Add layers to network
        # Also add Flatten between Conv and Dense layers
        for (layer, next_layer) in zip(self.graph, self.graph[1:]):
            model.add(layer.getKerasLayer())
            if layer.type == Layer.CONV2D and next_layer.type == Layer.DENSE:
                model.add(Flatten())

        # Process the last layer
        model.add(self.graph[-1].getKerasLayer())
        model.build((1, *self.inputShape))
        model.compile(optimizer='adam', loss="categorical_crossentropy")
        return model