import numpy as np
from torch.utils.data import DataLoader
import tensorflow.keras as keras
import logging
from tensorflow.keras.utils import to_categorical

import torch


logger = logging.getLogger("TEST")

class PytorchKerasAdapter(keras.utils.Sequence):
    def __init__(self, dataloader: DataLoader, num_classes, time=True, freq=False):
        self.gen = dataloader
        self.iter = iter(dataloader)

        self.num_classes = num_classes
        self.time = time
        self.freq = freq


    def __len__(self):
        return len(self.gen)

    def __getitem__(self, index):
        try:
            input_time, input_freq, target = next(self.iter)
        except StopIteration:
            self.iter = iter(self.gen)
            input_time, input_freq, target = next(self.iter)

        input_time = input_time.numpy().astype(np.float32)

        input_time = np.swapaxes(input_time, 1, 3)

        input_time = input_time.transpose(0, 2, 1, 3)

        # if self.expand_dim:
        #     input_time = np.expand_dims(input_time, self.expand_dim)

        target = to_categorical(target.numpy(), num_classes=self.num_classes)

        result = ()
        if self.time:
            result += (input_time, )

        result += (target, )

        assert not np.any(np.isnan(input_time))
        assert not np.any(np.isinf(input_time))
        assert not np.any(np.isneginf(input_time))

        assert not np.any(np.isnan(target))
        assert not np.any(np.isinf(target))
        assert not np.any(np.isneginf(target))

        return result