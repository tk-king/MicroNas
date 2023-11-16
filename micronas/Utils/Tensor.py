import numpy as np

# This class represents tensors which are also present in the tflm framework
class Tensor:
    
    def __init__(self, tensorShape, dataType=np.uint8) -> None:
        self.dataType = np.dtype(dataType)
        self.tensorShape = tensorShape # Tuple to describe the shape of the tensor

    @property
    def size(self):
        return np.prod(self.tensorShape) 

    def __repr__(self) -> str:
        return f'Shape: {self.tensorShape}, dataType: {self.dataType} (Size: {self.dataType.itemsize})'