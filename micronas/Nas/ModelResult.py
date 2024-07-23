from dataclasses import dataclass, field
from keras.models import Model
from typing import Dict
import json
import os
from micronas.Utils.JsonEncoder import JsonEncoder
from torch.nn import Module
import torch

@dataclass
class ModelResult:
    keras_model: Model = None
    torch_Model: Module = None
    tf_lite_model: object = None

    latency: float = None
    memory: float = None

    config: Dict = field(default_factory=dict)

    eval: Dict = field(default_factory=dict)
    eval_tf_lite: Dict = field(default_factory=dict)
    

    def save(self, path: str, idx: int):
        path = os.path.join(path, f"idx={idx}")
        # Ensure the directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save the Keras model
        if self.keras_model is not None:
            model_path = os.path.join(path, 'model.h5')
            self.keras_model.save(model_path)

        # Save the TFLite model
        if self.tf_lite_model is not None:
            tflite_path = os.path.join(path, 'model.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(self.tf_lite_model)
        
        if self.torch_Model is not None:
            torch_path = os.path.join(path, 'model.pth')
            torch.save(self.torch_Model, torch_path)

        # Save the other attributes
        result_data = {
            'latency': self.latency,
            'memory': self.memory,
            'eval': self.eval,
            'eval_tf_lite': self.eval_tf_lite,
            'config': self.config
        }

        for key, value in result_data.items():
            print(type(value))
        
        # Save the attributes as a JSON file
        json_path = os.path.join(path, 'result.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=4, cls=JsonEncoder)
