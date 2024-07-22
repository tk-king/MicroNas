import numpy as np
from torch.utils.data import DataLoader
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical

class PytorchKerasAdapter(keras.utils.Sequence):
    def __init__(self, dataloader: DataLoader, num_classes, time=True, freq=False):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.time = time
        self.freq = freq
        self.dataset = list(dataloader)  # Convert DataLoader to a list of batches
        self.current_index = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input_time, input_freq, target = self.dataset[index]

        input_time = input_time.numpy().astype(np.float32)
        input_time = input_time.transpose(0, 2, 3, 1)

        target = to_categorical(target.numpy(), num_classes=self.num_classes)

        result = ()
        if self.time:
            result += (input_time,)

        result += (target,)

        assert not np.any(np.isnan(input_time))
        assert not np.any(np.isinf(input_time))
        assert not np.any(np.isneginf(input_time))

        assert not np.any(np.isnan(target))
        assert not np.any(np.isinf(target))
        assert not np.any(np.isneginf(target))

        return result

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        item = self[self.current_index]
        self.current_index += 1
        return item