from dataclasses import dataclass
from keras.models import Model
from typing import Dict
import json
import os

@dataclass
class ModelResult:
    keras_model: Model
    tf_lite_model: object

    latency: float
    memory: float

    accuracy: float
    f1: float
    
    accuracy_quant: float
    f1_quant: float
    
    config: Dict

    def save(self, path: str):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the Keras model
        model_path = path + '_model.h5'
        self.keras_model.save(model_path)

        # Save the TFLite model
        tflite_path = path + '_tflite.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(self.tf_lite_model)
        
        # Save the other attributes
        result_data = {
            'latency': self.latency,
            'memory': self.memory,
            'accuracy': self.accuracy,
            'f1': self.f1,
            'accuracy_quant': self.accuracy_quant,
            'f1_quant': self.f1_quant,
            'config': self.config
        }
        
        # Save the attributes as a JSON file
        json_path = path + '_result.json'
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=4)
