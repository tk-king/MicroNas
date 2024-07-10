import numpy as np
from torch.utils.data import DataLoader
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical

import torch

class PytorchKerasAdapter(keras.utils.Sequence):
    def __init__(self, dataloader: DataLoader, num_classes, time=True, freq=False, expand_dim=None):
        self.gen = dataloader
        self.iter = iter(dataloader)

        self.num_classes = num_classes
        self.expand_dim = expand_dim
        self.time = time
        self.freq = freq

    def __len__(self):
        return len(self.gen)

    def __getitem__(self, index):

        # Try to obtain data
        # When dataloader is done, generate a new one
        try:
            input_time, target = next(self.iter)
        except StopIteration:
            self.iter = iter(self.gen)
            input_time, target = next(self.iter)

        # Convert pytorch format to keras format
        # Keras: (batch, length, channels)
        # Pytorch: (batch, channels, length)

        input_time = input_time.numpy().astype(np.float32)

        if self.expand_dim:
            input_time = np.expand_dims(input_time, self.expand_dim)

        # if self.freq:
        #     input_freq = torch.moveaxis(input_freq, 1, -1).numpy().astype(np.float32)
        target = to_categorical(
            target.numpy(), num_classes=self.num_classes, dtype=Config.tensor_dtype)

        result = ()
        if self.time:
            result += (input_time, )

        # if self.freq:
        #     result += (input_freq, )

        result += (target, )

        assert not np.any(np.isnan(input_time))
        assert not np.any(np.isinf(input_time))
        assert not np.any(np.isneginf(input_time))

        assert not np.any(np.isnan(target))
        assert not np.any(np.isinf(target))
        assert not np.any(np.isneginf(target))

        return result